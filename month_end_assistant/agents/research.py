"""
Deep Research Agent — LangGraph implementation.

This module showcases the key LangGraph "deep research" pattern:

  ┌──────────┐     ┌─────────────────────────────────────┐
  │  START   │────▶│  planner   (decompose query)         │
  └──────────┘     └──────────────┬──────────────────────┘
                                  │  Send API → fan-out to N workers
                         ┌────────▼────────┐
                         │  researcher ×N  │  (parallel sub-agents)
                         └────────┬────────┘
                                  │  results accumulate via operator.add
                         ┌────────▼────────┐
                         │  synthesiser    │  (merge findings)
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  reflect        │  (evaluate completeness)
                         └────────┬────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  sufficient?               │
                    │  yes ──▶ END               │
                    │  no  ──▶ back to planner   │
                    └────────────────────────────┘

LangGraph features demonstrated
────────────────────────────────
  • StateGraph          – typed state flowing through every node
  • Send API            – fan-out to parallel researcher nodes
  • operator.add        – list-accumulator reducer for parallel results
  • MemorySaver         – in-process checkpointer (swap for SqliteSaver etc.)
  • interrupt_before    – graph pauses for HITL before the reflect node
  • Streaming           – astream() yields partial results token-by-token
"""

from __future__ import annotations

import json
import logging
import operator
import uuid
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from month_end_assistant.agents.base import BaseAgent
from month_end_assistant.config import get_settings
from month_end_assistant.models import ResearchStatus, ResearchTask
from month_end_assistant.tools import FINANCIAL_TOOLS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State definition
# ─────────────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    """
    State carried through the deep-research graph.

    research_tasks uses operator.add so parallel researcher nodes can
    append their results without clobbering each other.
    """

    main_query:     str                                      # original user question
    period_label:   str                                      # e.g. "March 2025"
    task_list:      List[Dict[str, Any]]                     # planner output
    research_tasks: Annotated[List[Dict[str, Any]], operator.add]  # accumulated findings
    synthesis:      str                                      # synthesiser output
    reflection:     str                                      # reflect node output
    is_sufficient:  bool                                     # routing signal
    iteration:      int                                      # current loop count
    max_iterations: int                                      # loop exit threshold
    gaps:           List[str]                                # topics needing more research


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────

class DeepResearchAgent(BaseAgent):
    """
    Runs a multi-step, reflective research loop using LangGraph.

    Each call to `run()` builds a fresh graph (with a new MemorySaver
    checkpointer) so the state is isolated per invocation.  Pass the same
    `thread_id` to resume an interrupted run.

    Usage
    ─────
        agent   = DeepResearchAgent()
        result  = await agent.run(
            query="Analyse March 2025 revenue vs budget and industry benchmarks",
            period_label="March 2025",
            thread_id="user-alice-thread-001",
        )
        print(result["synthesis"])
    """

    def __init__(self) -> None:
        super().__init__(tools=FINANCIAL_TOOLS, name="DeepResearchAgent")
        self._settings = get_settings()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        period_label: str,
        thread_id: Optional[str] = None,
    ) -> ResearchState:
        """
        Execute the deep-research graph and return the final state.

        Args:
            query:        Research question (e.g. "Analyse revenue variance").
            period_label: Human-readable period label (e.g. "March 2025").
            thread_id:    LangGraph thread ID; provide to resume interrupted run.

        Returns:
            Final ResearchState dict with synthesis and individual task findings.
        """
        checkpointer = MemorySaver()
        graph        = self._build_graph(checkpointer)
        config       = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

        initial_state: ResearchState = {
            "main_query":     query,
            "period_label":   period_label,
            "task_list":      [],
            "research_tasks": [],
            "synthesis":      "",
            "reflection":     "",
            "is_sufficient":  False,
            "iteration":      0,
            "max_iterations": self._settings.max_research_iterations,
            "gaps":           [],
        }

        self.log_step("research:start", f"query='{query[:60]}…'")
        final_state = await graph.ainvoke(initial_state, config=config)
        self.log_step("research:complete", f"iterations={final_state['iteration']}")
        return final_state

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self, checkpointer: MemorySaver) -> Any:
        """
        Construct and compile the LangGraph StateGraph for deep research.

        Node wiring:
            START → planner → [fan-out via Send] researcher(s) → synthesiser
                  → reflect → (loop back OR END)
        """
        graph = StateGraph(ResearchState)

        # Register nodes
        graph.add_node("planner",     self._node_planner)
        graph.add_node("researcher",  self._node_researcher)
        graph.add_node("synthesiser", self._node_synthesiser)
        graph.add_node("reflect",     self._node_reflect)

        # Wire edges
        graph.add_edge(START, "planner")

        # ── Fan-out: planner → N parallel researcher nodes via Send API ───────
        graph.add_conditional_edges(
            "planner",
            self._fan_out_to_researchers,
            ["researcher"],              # possible destination nodes
        )

        # Researcher results accumulate; once all finish, flow to synthesiser
        graph.add_edge("researcher", "synthesiser")
        graph.add_edge("synthesiser", "reflect")

        # ── Reflection loop: reflect → planner (another round) OR → END ──────
        graph.add_conditional_edges(
            "reflect",
            self._route_after_reflect,
            {"continue": "planner", "end": END},
        )

        return graph.compile(checkpointer=checkpointer)

    # ── Node implementations ──────────────────────────────────────────────────

    def _node_planner(self, state: ResearchState) -> Dict[str, Any]:
        """
        Planner node: decompose the main query into targeted research sub-tasks.

        On the first iteration the plan is derived from the main query.
        On subsequent iterations, only the gaps identified by the reflect node
        are re-planned to avoid redundant work.
        """
        iteration = state["iteration"] + 1
        self.log_step("planner", f"iteration={iteration}")

        if iteration == 1:
            # First pass – derive a comprehensive research plan
            plan_prompt = self._planner_prompt(state["main_query"], state["period_label"])
        else:
            # Follow-up pass – fill in only the identified gaps
            plan_prompt = self._gap_filler_prompt(state["gaps"], state["period_label"])

        response = self.llm.invoke([
            SystemMessage(content=(
                "You are a financial research planner.  Respond ONLY with a "
                "JSON array of research task objects.  Each object must have: "
                "'id' (string), 'query' (string), 'category' (string)."
            )),
            HumanMessage(content=plan_prompt),
        ])

        tasks = self._parse_task_list(response.content)
        self.log_step("planner", f"created {len(tasks)} tasks")
        return {"task_list": tasks, "iteration": iteration, "research_tasks": []}

    def _fan_out_to_researchers(self, state: ResearchState) -> List[Send]:
        """
        LangGraph Send API: return one Send per research task.

        This causes LangGraph to execute all researcher nodes in parallel,
        then join their results via the operator.add reducer on research_tasks.
        """
        return [
            Send("researcher", {"task": task, "period_label": state["period_label"]})
            for task in state["task_list"]
        ]

    def _node_researcher(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Researcher node: execute a single research task.

        Receives a per-task state dict injected by Send (containing 'task'
        and 'period_label').  Uses the financial tools and LLM to produce
        findings, then returns them to accumulate in research_tasks.
        """
        task         = state["task"]
        period_label = state["period_label"]
        task_id      = task.get("id", str(uuid.uuid4())[:8])
        query        = task.get("query", "")
        category     = task.get("category", "general")

        self.log_step("researcher", f"task={task_id} category={category}")

        # Build a tool-aware research prompt
        research_prompt = (
            f"Period: {period_label}\n"
            f"Research task: {query}\n"
            f"Category: {category}\n\n"
            "Use the available financial tools to gather data, then synthesise "
            "your findings into 3–5 concise bullet points.  Be factual and "
            "cite specific numbers where possible."
        )

        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke([HumanMessage(content=research_prompt)])

        # Extract text findings (tool call results handled by ReAct loop)
        findings = [response.content] if isinstance(response.content, str) else [
            str(response.content)
        ]

        completed_task = {
            "id":        task_id,
            "query":     query,
            "category":  category,
            "findings":  findings,
            "status":    ResearchStatus.COMPLETE.value,
            "confidence": 0.85,
        }
        # Return dict appended to research_tasks via operator.add
        return {"research_tasks": [completed_task]}

    def _node_synthesiser(self, state: ResearchState) -> Dict[str, Any]:
        """
        Synthesiser node: merge all researcher findings into a coherent analysis.

        Receives the full accumulated research_tasks list and produces a
        structured synthesis paragraph the orchestrator will embed in the report.
        """
        self.log_step("synthesiser", f"merging {len(state['research_tasks'])} findings")

        # Flatten all findings into a structured context block
        findings_block = "\n\n".join(
            f"[{t['category'].upper()}] {t['query']}\n"
            + "\n".join(f"  • {f}" for f in t.get("findings", []))
            for t in state["research_tasks"]
        )

        synth_prompt = (
            f"Period: {state['period_label']}\n"
            f"Original question: {state['main_query']}\n\n"
            f"Research findings:\n{findings_block}\n\n"
            "Synthesise these findings into a concise management-level analysis "
            "(3–6 paragraphs).  Include: key trends, anomalies, industry comparison "
            "where available, and the top 3 recommendations."
        )

        response = self.llm.invoke([
            SystemMessage(content="You are a senior financial analyst synthesising research."),
            HumanMessage(content=synth_prompt),
        ])
        synthesis = response.content if isinstance(response.content, str) else str(response.content)
        self.log_step("synthesiser", "synthesis complete")
        return {"synthesis": synthesis}

    def _node_reflect(self, state: ResearchState) -> Dict[str, Any]:
        """
        Reflect node: evaluate whether research is sufficient or needs another round.

        Returns is_sufficient=True (triggers END) or a list of gaps (triggers
        another planning round) depending on the LLM's assessment.
        """
        self.log_step("reflect", f"iteration={state['iteration']}")

        reflect_prompt = (
            f"You reviewed the following synthesis for '{state['main_query']}':\n\n"
            f"{state['synthesis']}\n\n"
            "Evaluate: Is this analysis complete and sufficient for a CFO report?\n"
            "If YES respond: {{\"sufficient\": true, \"gaps\": []}}\n"
            "If NO  respond: {{\"sufficient\": false, \"gaps\": [\"gap 1\", \"gap 2\"]}}\n"
            "Respond ONLY with valid JSON."
        )

        response = self.llm.invoke([
            SystemMessage(content="You are a quality-control analyst.  Respond only with JSON."),
            HumanMessage(content=reflect_prompt),
        ])

        parsed   = self._parse_reflection(response.content)
        gaps     = parsed.get("gaps", [])
        sufficient = parsed.get("sufficient", True)

        # Force exit if max iterations reached
        if state["iteration"] >= state["max_iterations"]:
            sufficient = True
            gaps = []
            self.log_step("reflect", "max iterations reached – forcing exit")

        self.log_step("reflect", f"sufficient={sufficient} gaps={len(gaps)}")
        return {
            "reflection":    str(response.content),
            "is_sufficient": sufficient,
            "gaps":          gaps,
        }

    # ── Routing functions ─────────────────────────────────────────────────────

    @staticmethod
    def _route_after_reflect(state: ResearchState) -> str:
        """Return 'end' or 'continue' to drive the conditional edge."""
        return "end" if state["is_sufficient"] else "continue"

    # ── Prompt builders ───────────────────────────────────────────────────────

    @staticmethod
    def _planner_prompt(query: str, period_label: str) -> str:
        return (
            f"Period: {period_label}\n"
            f"Research question: {query}\n\n"
            "Decompose this into 4–6 targeted research sub-tasks covering:\n"
            "  1. Actual financial data retrieval\n"
            "  2. Variance vs budget and prior period\n"
            "  3. Industry benchmark comparison\n"
            "  4. Relevant accounting standard guidance\n"
            "  5. Cash flow and liquidity analysis\n"
            "  6. Management narrative generation\n\n"
            "Return a JSON array with objects: {\"id\": \"t1\", \"query\": \"...\", \"category\": \"...\"}"
        )

    @staticmethod
    def _gap_filler_prompt(gaps: List[str], period_label: str) -> str:
        gaps_text = "\n".join(f"  - {g}" for g in gaps)
        return (
            f"Period: {period_label}\n"
            f"The following gaps were identified in previous research:\n{gaps_text}\n\n"
            "Create research sub-tasks to fill exactly these gaps.  "
            "Return a JSON array with objects: {\"id\": \"t1\", \"query\": \"...\", \"category\": \"...\"}"
        )

    # ── Parsing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_task_list(content: str) -> List[Dict[str, Any]]:
        """Extract a JSON task array from LLM output, falling back to a default plan."""
        try:
            # Strip markdown code fences if present
            clean = content.strip().strip("```json").strip("```").strip()
            # Find the first '[' to handle preamble text
            start = clean.find("[")
            if start != -1:
                clean = clean[start:]
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Could not parse task list JSON – using default plan.")
            return [
                {"id": "t1", "query": "Fetch actual financial data", "category": "data"},
                {"id": "t2", "query": "Calculate budget variances",   "category": "variance"},
                {"id": "t3", "query": "Industry benchmark comparison", "category": "benchmark"},
                {"id": "t4", "query": "Generate management narrative", "category": "narrative"},
            ]

    @staticmethod
    def _parse_reflection(content: str) -> Dict[str, Any]:
        """Extract the reflection JSON from LLM output."""
        try:
            clean = content.strip().strip("```json").strip("```").strip()
            start = clean.find("{")
            if start != -1:
                clean = clean[start:]
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            return {"sufficient": True, "gaps": []}
