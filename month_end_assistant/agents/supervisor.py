"""
LangGraph Multi-Agent Supervisor.

Implements the canonical LangGraph supervisor pattern where a central
"router" LLM dynamically decides which specialist worker agent to invoke
next based on the current task and conversation state.

Architecture
────────────

  ┌──────────────────────────────────────────────────────────────────────┐
  │                        Supervisor Graph                              │
  │                                                                      │
  │  User query                                                          │
  │      │                                                               │
  │      ▼                                                               │
  │  ┌─────────────────┐                                                 │
  │  │   SUPERVISOR    │ ◄────────────────────────────────────────┐      │
  │  │  (router LLM)   │                                          │      │
  │  └───────┬─────────┘                                          │      │
  │          │  Command(goto=worker_name)                         │      │
  │    ┌─────▼──────────────────────────────────────────────┐    │      │
  │    │               WORKER AGENTS                         │    │      │
  │    │                                                      │    │      │
  │    │  ┌──────────┐ ┌──────────┐ ┌─────────┐            │    │      │
  │    │  │  Data    │ │ Research │ │ Anomaly │            │    │      │
  │    │  │  Worker  │ │  Worker  │ │ Worker  │            │    │      │
  │    │  └────┬─────┘ └────┬─────┘ └────┬────┘            │    │      │
  │    │       │             │             │                  │    │      │
  │    │  ┌────▼─────┐ ┌────▼─────┐                         │    │      │
  │    │  │  Recon   │ │  Risk    │                         │    │      │
  │    │  │  Worker  │ │  Worker  │                         │    │      │
  │    │  └──────────┘ └──────────┘                         │    │      │
  │    └──────────────────────────────┬─────────────────────┘    │      │
  │                                   │  Command(goto="supervisor")     │
  │                                   └────────────────────────────────┘│
  │                                                                      │
  │  Supervisor sees accumulated results, decides END or next worker     │
  └──────────────────────────────────────────────────────────────────────┘

LangGraph features demonstrated
────────────────────────────────
  • StateGraph with MessagesState  – message-based state shared by all nodes
  • create_react_agent             – each worker is a ReAct agent with tools
  • Command(goto=…)                – supervisor routes dynamically via Command
  • Conditional edge on "__end__"  – supervisor can terminate the graph
  • Parallel-capable design        – multiple workers can be called in sequence
  • MemorySaver checkpointer       – state is persisted across turns
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from month_end_assistant.agents.base import BaseAgent
from month_end_assistant.models import AgentState
from month_end_assistant.tools.financial import FINANCIAL_TOOLS
from month_end_assistant.tools.reporting import REPORTING_TOOLS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Worker names  (also used as graph node names and routing keys)
# ─────────────────────────────────────────────────────────────────────────────

WORKERS = Literal[
    "data_worker",
    "research_worker",
    "anomaly_worker",
    "recon_worker",
    "risk_worker",
    "__end__",
]


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor Agent
# ─────────────────────────────────────────────────────────────────────────────

class MonthEndSupervisor(BaseAgent):
    """
    LangGraph supervisor that routes between five specialist worker agents.

    Each worker is built with `create_react_agent` (a prebuilt ReAct loop),
    giving it access to a curated tool set.  The supervisor LLM decides which
    worker to invoke next using structured JSON output, then emits a LangGraph
    `Command` to route the graph accordingly.

    Workers
    ───────
      data_worker     – fetches actuals, calculates variances, generates narrative
      research_worker – deep-dives into accounting standards, benchmarks
      anomaly_worker  – scans for unusual transactions and patterns
      recon_worker    – performs account reconciliations
      risk_worker     – assesses and scores financial risks

    Usage
    ─────
        supervisor = MonthEndSupervisor()
        result     = await supervisor.run(
            query="Find all anomalies, then reconcile the bank account, "
                  "then assess credit risk for March 2025.",
            period_label="March 2025",
        )
    """

    _SUPERVISOR_SYSTEM = (
        "You are a month-end close supervisor orchestrating a team of specialist agents.\n\n"
        "Available workers:\n"
        "  data_worker     – fetch financial actuals, calculate variances, generate narratives\n"
        "  research_worker – lookup accounting standards (IFRS/GAAP), industry benchmarks\n"
        "  anomaly_worker  – detect unusual transactions, duplicate entries, threshold breaches\n"
        "  recon_worker    – reconcile balance-sheet accounts (bank, AR, AP, inventory)\n"
        "  risk_worker     – assess financial risks (credit, liquidity, FX, compliance)\n\n"
        "For each turn, decide which worker to invoke next (or '__end__' if the task is done).\n"
        "Respond ONLY with valid JSON: {\"next\": \"<worker_name>\", \"reason\": \"<why>\"}."
    )

    def __init__(self) -> None:
        super().__init__(name="MonthEndSupervisor")
        self._checkpointer = MemorySaver()
        self._graph        = self._build_graph()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        period_label: str = "current period",
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Route a query through the worker team and return the accumulated result.

        Args:
            query:        Task for the supervisor to orchestrate.
            period_label: Human-readable period label.
            thread_id:    LangGraph thread ID for checkpointing.
        """
        import uuid
        thread_id = thread_id or str(uuid.uuid4())
        config    = {"configurable": {"thread_id": thread_id}}

        initial = {
            "messages": [
                SystemMessage(content=(
                    f"You are analysing the month-end close for {period_label}. "
                    "Complete the requested tasks using the available specialist workers."
                )),
                HumanMessage(content=query),
            ]
        }

        self.log_step("supervisor:run", f"query='{query[:60]}…'")
        result = await self._graph.ainvoke(initial, config=config)
        return result

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        """
        Wire the supervisor and worker nodes into a StateGraph.

        Each worker is a `create_react_agent` node that returns control to
        the supervisor via `Command(goto="supervisor")`.  The supervisor
        then decides the next step.
        """
        from langgraph.graph.message import MessagesState

        graph = StateGraph(MessagesState)

        # ── Register supervisor node ──────────────────────────────────────────
        graph.add_node("supervisor", self._supervisor_node)

        # ── Register worker nodes (each is a ReAct agent) ─────────────────────
        graph.add_node("data_worker",     self._make_worker_node(
            name="Data Worker",
            tools=FINANCIAL_TOOLS,
            system=(
                "You are a financial data analyst. Use your tools to fetch actuals, "
                "compute variances, and generate management narratives. "
                "When done, summarise your findings concisely."
            ),
        ))

        graph.add_node("research_worker", self._make_worker_node(
            name="Research Worker",
            tools=[t for t in FINANCIAL_TOOLS
                   if t.name in ("lookup_accounting_standard", "get_industry_benchmarks")],
            system=(
                "You are an accounting-standards researcher. Use your tools to look up "
                "IFRS / GAAP guidance and industry benchmarks. Cite the specific standard."
            ),
        ))

        graph.add_node("anomaly_worker",  self._make_worker_node(
            name="Anomaly Worker",
            tools=[t for t in REPORTING_TOOLS if t.name == "detect_anomalies"],
            system=(
                "You are a forensic financial analyst. Run the anomaly detection tool "
                "and explain each finding clearly, noting severity and recommended action."
            ),
        ))

        graph.add_node("recon_worker",    self._make_worker_node(
            name="Reconciliation Worker",
            tools=[t for t in REPORTING_TOOLS if t.name == "reconcile_account"],
            system=(
                "You are an account reconciliation specialist. Reconcile the requested "
                "accounts and flag any exceptions with proposed clearing actions."
            ),
        ))

        graph.add_node("risk_worker",     self._make_worker_node(
            name="Risk Worker",
            tools=[t for t in REPORTING_TOOLS if t.name == "assess_financial_risk"],
            system=(
                "You are a financial risk manager. Assess and score financial risks, "
                "then recommend mitigations ranked by risk score."
            ),
        ))

        # ── Edges ─────────────────────────────────────────────────────────────
        graph.add_edge(START, "supervisor")

        # All workers return to supervisor via Command – no static edges needed
        # The supervisor node itself emits Command(goto=worker | END)

        return graph.compile(checkpointer=self._checkpointer)

    # ── Supervisor node ───────────────────────────────────────────────────────

    def _supervisor_node(self, state: Dict[str, Any]) -> Command:
        """
        The supervisor LLM reads the current message history and decides which
        worker to dispatch to next (or '__end__' if the task is complete).

        Returns a LangGraph `Command` that re-routes the graph.
        """
        messages = state.get("messages", [])

        supervisor_prompt = [
            SystemMessage(content=self._SUPERVISOR_SYSTEM),
            *messages,
        ]

        response = self.llm.invoke(supervisor_prompt)

        # Parse the routing decision from the LLM output
        decision = self._parse_routing(response.content)
        next_node = decision.get("next", "__end__")
        reason    = decision.get("reason", "")

        self.log_step("supervisor", f"→ {next_node}  ({reason[:60]})")

        if next_node == "__end__":
            return Command(goto=END)

        return Command(
            goto=next_node,
            update={"messages": [AIMessage(content=f"[Supervisor → {next_node}] {reason}")]},
        )

    # ── Worker node factory ────────────────────────────────────────────────────

    def _make_worker_node(
        self,
        name: str,
        tools: List[Any],
        system: str,
    ):
        """
        Return a node function wrapping a `create_react_agent` worker.

        The worker runs its ReAct loop, then emits `Command(goto="supervisor")`
        to hand control back.

        create_react_agent builds a full ReAct (Reason + Act) loop:
          1. LLM decides which tool to call
          2. Tool executes
          3. LLM evaluates result and decides next tool or final answer
        """
        worker_agent = create_react_agent(
            model=self.llm,
            tools=tools,
            state_modifier=SystemMessage(content=system),
        )

        async def node_fn(state: Dict[str, Any]) -> Command:
            self.log_step(f"worker:{name.lower().replace(' ', '_')}")
            result = await worker_agent.ainvoke(state)
            worker_message = result["messages"][-1]
            return Command(
                goto="supervisor",
                update={"messages": [
                    AIMessage(content=f"[{name}]\n{worker_message.content}")
                ]},
            )

        node_fn.__name__ = name.lower().replace(" ", "_")
        return node_fn

    # ── Parsing helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_routing(content: str) -> Dict[str, str]:
        """Extract JSON routing decision from LLM output."""
        try:
            clean = content.strip().strip("```json").strip("```").strip()
            start = clean.find("{")
            if start != -1:
                clean = clean[start:]
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            # If parsing fails, end the chain rather than loop forever
            logger.warning("Could not parse supervisor routing – terminating.")
            return {"next": "__end__", "reason": "Parse error – stopping safely."}
