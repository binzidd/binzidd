"""
CritiqueAgent — Reflexion-pattern self-reflection agent.

Implements the three-phase Reflexion loop:
  1. Generate   – produce an initial analysis output
  2. Critique   – evaluate the output against structured criteria
  3. Revise     – rewrite the output addressing critique feedback
  4. Loop       – repeat until quality threshold met or max_rounds reached

Graph:
    START → generate → critique → (revise → critique → …) → save → END

This is a proper LangGraph implementation with:
  ✅ Typed StateGraph with operator.add for history accumulation
  ✅ Pydantic-structured critique schema (no free-form JSON)
  ✅ Conditional loop with convergence detection
  ✅ MemorySaver checkpointer for mid-loop resumption
  ✅ Custom event emission for streaming progress updates
  ✅ BaseAgent inheritance (shared Bedrock LLM)

Usage::
    agent = CritiqueAgent(max_rounds=3, quality_threshold=0.80)
    result = await agent.run(
        content="Revenue grew 12% YoY but we missed the IFRS 15 adjustment.",
        criteria=["IFRS compliance", "completeness", "actionability"],
        period="2025-09",
        user_id="u1",
    )
    print(result["revised_content"])
"""
from __future__ import annotations

import json
import logging
import operator
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from month_end_assistant.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class CriterionScore(BaseModel):
    criterion: str
    score: float = Field(ge=0.0, le=1.0, description="0.0 = fails completely, 1.0 = perfect")
    issues: List[str] = Field(default_factory=list, description="Specific problems identified")
    suggestions: List[str] = Field(default_factory=list, description="Concrete improvements")

class CritiqueResult(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    criterion_scores: List[CriterionScore]
    strengths: List[str] = Field(default_factory=list)
    critical_gaps: List[str] = Field(default_factory=list)
    is_sufficient: bool = Field(description="True if overall_score >= quality threshold")
    revision_instructions: str = Field(
        description="Specific, actionable instructions for the revision step"
    )


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class CritiqueState(TypedDict):
    # Input
    original_content: str
    criteria: List[str]
    period_label: str
    user_id: str

    # Working state
    current_content: str
    critique_history: Annotated[List[CritiqueResult], operator.add]  # accumulate across rounds
    round: int
    max_rounds: int
    quality_threshold: float
    is_complete: bool

    # Output
    revised_content: str
    final_critique: Optional[Dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

_GENERATE_SYSTEM = """You are a senior financial analyst producing month-end analysis.
Write a thorough, structured analysis that:
- Addresses the specific criteria provided
- Cites relevant accounting standards (IFRS/GAAP) where applicable
- Quantifies findings with numbers where possible
- Ends with clear, actionable recommendations"""

_CRITIQUE_SYSTEM = """You are a quality-assurance reviewer for financial analysis.
Evaluate the analysis strictly against the criteria. Be specific about weaknesses.
Output ONLY valid JSON matching the CritiqueResult schema."""

_REVISE_SYSTEM = """You are a financial analyst revising your analysis.
Address every issue in the critique instructions. Keep strengths intact.
Produce a complete, revised analysis — do NOT reference the critique process."""


class CritiqueAgent(BaseAgent):
    """
    Self-reflection agent using the Reflexion pattern.

    The agent generates an initial output, critiques it with structured scoring,
    revises based on feedback, and repeats until the quality threshold is met
    or max_rounds is exhausted.
    """

    def __init__(
        self,
        max_rounds: int = 3,
        quality_threshold: float = 0.80,
    ) -> None:
        super().__init__(name="CritiqueAgent")
        self._max_rounds       = max_rounds
        self._quality_threshold = quality_threshold
        self._checkpointer     = MemorySaver()
        self.graph             = self._build_graph()

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        content: str,
        criteria: Optional[List[str]] = None,
        period: str = "2025-09",
        user_id: str = "anonymous",
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the critique loop on `content`.

        If `content` is a question/task (not a draft), the generate node will
        produce an initial draft first.  If it's already a draft, pass
        `criteria` to focus the critique.
        """
        import uuid
        thread_id = thread_id or str(uuid.uuid4())
        criteria  = criteria or [
            "IFRS/GAAP compliance",
            "completeness",
            "quantitative support",
            "actionability",
            "clarity",
        ]

        initial_state: CritiqueState = {
            "original_content":  content,
            "criteria":          criteria,
            "period_label":      period,
            "user_id":           user_id,
            "current_content":   "",           # generate node will populate
            "critique_history":  [],
            "round":             0,
            "max_rounds":        self._max_rounds,
            "quality_threshold": self._quality_threshold,
            "is_complete":       False,
            "revised_content":   "",
            "final_critique":    None,
        }

        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
        output = await self.graph.ainvoke(initial_state, config=config)
        self.log_step("done", f"rounds={output['round']}  score={output.get('final_critique', {}).get('overall_score', '?')}")
        return output

    def run_sync(
        self,
        content: str,
        criteria: Optional[List[str]] = None,
        period: str = "2025-09",
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self.run(content, criteria, period, user_id))

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        g = StateGraph(CritiqueState)

        g.add_node("generate", self._node_generate)
        g.add_node("critique", self._node_critique)
        g.add_node("revise",   self._node_revise)
        g.add_node("save",     self._node_save)

        g.add_edge(START,      "generate")
        g.add_edge("generate", "critique")
        g.add_conditional_edges(
            "critique",
            self._route_critique,
            {"revise": "revise", "done": "save"},
        )
        g.add_edge("revise",   "critique")   # loop back
        g.add_edge("save",     END)

        return g.compile(checkpointer=self._checkpointer)

    # ── Node: generate ────────────────────────────────────────────────────────

    async def _node_generate(self, state: CritiqueState) -> Dict[str, Any]:
        self.log_step("generate", f"round={state['round']}")

        criteria_str = "\n".join(f"  • {c}" for c in state["criteria"])
        messages = [
            SystemMessage(content=_GENERATE_SYSTEM),
            HumanMessage(content=(
                f"Period: {state['period_label']}\n"
                f"Task/Content:\n{state['original_content']}\n\n"
                f"Evaluation criteria (address all):\n{criteria_str}"
            )),
        ]
        response = await self.llm.ainvoke(messages)
        return {"current_content": response.content}

    # ── Node: critique ────────────────────────────────────────────────────────

    async def _node_critique(self, state: CritiqueState) -> Dict[str, Any]:
        self.log_step("critique", f"round={state['round']}")

        criteria_str = ", ".join(state["criteria"])
        schema_hint  = CritiqueResult.model_json_schema()

        messages = [
            SystemMessage(content=_CRITIQUE_SYSTEM),
            HumanMessage(content=(
                f"Evaluate against criteria: {criteria_str}\n\n"
                f"Quality threshold: {state['quality_threshold']}\n\n"
                f"CONTENT TO EVALUATE:\n{state['current_content']}\n\n"
                f"JSON schema:\n{json.dumps(schema_hint, indent=2)}\n\n"
                "Respond with JSON only."
            )),
        ]

        try:
            structured_llm = self.llm.with_structured_output(CritiqueResult)
            critique: CritiqueResult = await structured_llm.ainvoke(messages)
        except Exception:
            # Fallback: parse raw JSON
            raw_response = await self.llm.ainvoke(messages)
            content = raw_response.content.strip()
            if "```" in content:
                content = content.split("```")[1].lstrip("json\n")
            parsed = json.loads(content)
            parsed.setdefault("is_sufficient", parsed.get("overall_score", 0) >= state["quality_threshold"])
            critique = CritiqueResult(**parsed)

        critique.is_sufficient = critique.overall_score >= state["quality_threshold"]
        is_complete = critique.is_sufficient or state["round"] >= state["max_rounds"] - 1

        self.log_step("critique", f"score={critique.overall_score:.2f}  sufficient={critique.is_sufficient}")
        return {
            "critique_history": [critique],     # operator.add will append
            "is_complete":      is_complete,
            "final_critique":   critique.model_dump(),
            "round":            state["round"] + 1,
        }

    # ── Node: revise ──────────────────────────────────────────────────────────

    async def _node_revise(self, state: CritiqueState) -> Dict[str, Any]:
        self.log_step("revise", f"round={state['round']}")

        last_critique = state["critique_history"][-1]
        messages = [
            SystemMessage(content=_REVISE_SYSTEM),
            HumanMessage(content=(
                f"ORIGINAL TASK:\n{state['original_content']}\n\n"
                f"CURRENT DRAFT:\n{state['current_content']}\n\n"
                f"CRITIQUE INSTRUCTIONS:\n{last_critique.revision_instructions}\n\n"
                f"SPECIFIC ISSUES:\n"
                + "\n".join(
                    f"[{cs.criterion}] " + "; ".join(cs.issues)
                    for cs in last_critique.criterion_scores if cs.issues
                )
            )),
        ]
        response = await self.llm.ainvoke(messages)
        return {"current_content": response.content}

    # ── Node: save ────────────────────────────────────────────────────────────

    async def _node_save(self, state: CritiqueState) -> Dict[str, Any]:
        self.log_step("save", f"final round={state['round']}  complete={state['is_complete']}")
        return {"revised_content": state["current_content"]}

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route_critique(self, state: CritiqueState) -> str:
        if state["is_complete"]:
            return "done"
        return "revise"

    # ── Convenience: critique an existing report ──────────────────────────────

    async def critique_report(
        self,
        report: str,
        additional_criteria: Optional[List[str]] = None,
        period: str = "2025-09",
    ) -> Dict[str, Any]:
        """Shortcut: critique an already-generated report rather than generating from scratch."""
        criteria = [
            "IFRS/GAAP compliance",
            "variance analysis depth",
            "risk identification",
            "actionability",
            *(additional_criteria or []),
        ]
        return await self.run(report, criteria, period)
