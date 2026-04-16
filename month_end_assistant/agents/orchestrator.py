"""
Month-End Orchestrator — the top-level LangGraph supervisor graph.

This is the main graph the application invokes.  It coordinates every
subsystem in a linear pipeline with a HITL approval gate:

  ┌─────────────────────────────────────────────────────────────────┐
  │                    Orchestrator Graph                           │
  │                                                                 │
  │  START                                                          │
  │   │                                                             │
  │   ▼                                                             │
  │  load_session      ── load user context from AgentCore Memory   │
  │   │                                                             │
  │   ▼                                                             │
  │  deep_research     ── LangGraph deep-research sub-graph         │
  │   │                                                             │
  │   ▼                                                             │
  │  analyse_finances  ── fetch actuals, calculate variances        │
  │   │                                                             │
  │   ▼                                                             │
  │  sandbox_deepdive  ── run custom calculations in the sandbox    │
  │   │                                                             │
  │   ▼                                                             │
  │  build_report      ── assemble MonthEndReport draft             │
  │   │                                                             │
  │   ▼                                                             │
  │  ◆ hitl_checkpoint  ── interrupt() + Teams/Slack notifications  │
  │   │         (graph pauses here until human responds)           │
  │   ▼                                                             │
  │  route_approval    ── approved → publish  |  rejected → revise  │
  │   │                                                             │
  │   ▼                                                             │
  │  publish_report    ── save report, update session memory        │
  │   │                                                             │
  │   ▼                                                             │
  │  END                                                            │
  └─────────────────────────────────────────────────────────────────┘

LangGraph features demonstrated here
──────────────────────────────────────
  • StateGraph          – AgentState flows through every node
  • interrupt()         – graph pauses at hitl_checkpoint for human input
  • Command(resume=…)   – operator resumes the graph with approval decision
  • MemorySaver         – thread-level checkpointing for graph state
  • Conditional edges   – route_approval branches on HITLResponse.status
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from month_end_assistant.agents.base import BaseAgent
from month_end_assistant.agents.research import DeepResearchAgent
from month_end_assistant.aws.agentcore import AgentCoreClient
from month_end_assistant.config import get_settings
from month_end_assistant.hitl import HITLManager
from month_end_assistant.memory import SessionManager
from month_end_assistant.models import (
    AgentState,
    ApprovalStatus,
    FinancialMetrics,
    HITLRequest,
    HITLResponse,
    MonthEndPeriod,
    MonthEndReport,
    ReportStatus,
    UserSession,
    VarianceReport,
)
from month_end_assistant.sandbox import SandboxExecutor
from month_end_assistant.tools import calculate_variances, fetch_financial_data

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator class
# ─────────────────────────────────────────────────────────────────────────────

class MonthEndOrchestrator(BaseAgent):
    """
    Supervisor agent that owns the full month-end close pipeline.

    Instantiate once and call `run()` for each month-end cycle.  Pass the
    same thread_id to resume from a HITL checkpoint.

    Usage
    ─────
        orch   = MonthEndOrchestrator()
        # First run – will pause at HITL checkpoint
        state  = await orch.run(period, session, thread_id="abc123")

        # After human approves in Teams/Slack
        state  = await orch.resume(
            thread_id="abc123",
            response=HITLResponse(request_id=..., status=ApprovalStatus.APPROVED)
        )
    """

    def __init__(self) -> None:
        super().__init__(name="MonthEndOrchestrator")
        self._research   = DeepResearchAgent()
        self._hitl       = HITLManager()
        self._session_mgr = SessionManager()
        self._agentcore  = AgentCoreClient()
        self._sandbox    = SandboxExecutor()
        self._checkpointer = MemorySaver()
        self._graph      = self._build_graph()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(
        self,
        period: MonthEndPeriod,
        session: UserSession,
        thread_id: Optional[str] = None,
    ) -> AgentState:
        """
        Start (or continue) the month-end pipeline for *period*.

        The graph will pause at the HITL checkpoint and return the interrupted
        state.  Call `resume()` once the human has responded.
        """
        thread_id  = thread_id or session.thread_id
        config     = {"configurable": {"thread_id": thread_id}}

        initial: AgentState = {
            "messages":      [HumanMessage(content=(
                f"Run month-end close for {period.label}. "
                f"User: {session.user_id}, Company: {session.company_id}."
            ))],
            "research_tasks": [],
            "report":         None,
            "hitl_request":   None,
            "hitl_response":  None,
            "session":        json.loads(session.model_dump_json()),
            "iteration":      0,
            "next_step":      "load_session",
            "error":          "",
        }

        self.log_step("orchestrator:run", f"period={period.label} thread={thread_id}")
        return await self._graph.ainvoke(initial, config=config)

    async def resume(
        self,
        thread_id: str,
        response: HITLResponse,
    ) -> AgentState:
        """
        Resume the graph after a human responds to the HITL approval request.

        Wraps the response in a LangGraph Command so the interrupt() call in
        `_node_hitl_checkpoint` receives it as its return value.
        """
        self._hitl.record_response(response)
        config  = {"configurable": {"thread_id": thread_id}}
        command = Command(resume=response.model_dump())
        self.log_step("orchestrator:resume", f"status={response.status}")
        return await self._graph.ainvoke(command, config=config)

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        """
        Wire all nodes and edges into a compiled StateGraph.

        The graph is compiled with:
          • checkpointer  – MemorySaver for thread-level state persistence
          • interrupt_before=["hitl_checkpoint"]  – auto-pause before HITL node
            (alternative approach to explicit interrupt() inside the node)
        """
        graph = StateGraph(AgentState)

        # Register nodes
        graph.add_node("load_session",      self._node_load_session)
        graph.add_node("deep_research",     self._node_deep_research)
        graph.add_node("analyse_finances",  self._node_analyse_finances)
        graph.add_node("sandbox_deepdive",  self._node_sandbox_deepdive)
        graph.add_node("build_report",      self._node_build_report)
        graph.add_node("hitl_checkpoint",   self._node_hitl_checkpoint)
        graph.add_node("publish_report",    self._node_publish_report)
        graph.add_node("revise_report",     self._node_revise_report)

        # Linear pipeline
        graph.add_edge(START,              "load_session")
        graph.add_edge("load_session",     "deep_research")
        graph.add_edge("deep_research",    "analyse_finances")
        graph.add_edge("analyse_finances", "sandbox_deepdive")
        graph.add_edge("sandbox_deepdive", "build_report")
        graph.add_edge("build_report",     "hitl_checkpoint")

        # ── Conditional branch after HITL approval ────────────────────────────
        graph.add_conditional_edges(
            "hitl_checkpoint",
            self._route_approval,
            {
                "publish":  "publish_report",
                "revise":   "revise_report",
                "escalate": "hitl_checkpoint",  # loop back for re-approval
            },
        )
        graph.add_edge("publish_report", END)
        graph.add_edge("revise_report",  "build_report")   # rebuild → re-approve

        return graph.compile(checkpointer=self._checkpointer)

    # ── Node implementations ──────────────────────────────────────────────────

    async def _node_load_session(self, state: AgentState) -> Dict[str, Any]:
        """
        Load user session and retrieve relevant memories from AgentCore Memory.

        Injects historical context (prior period summaries, user preferences,
        custom rules) into the message history so every downstream node
        benefits from accumulated institutional knowledge.
        """
        self.log_step("load_session")
        session_dict = state["session"]
        user_id      = session_dict.get("user_id", "unknown")
        period_label = session_dict.get("active_period", {}).get("label", "current period") \
                       if session_dict.get("active_period") else "current period"

        # Retrieve relevant memories from AgentCore
        memories = await self._agentcore.retrieve_memories(
            query=f"month-end close {period_label} revenue expenses variances",
            user_id=user_id,
            top_k=5,
        )

        memory_context = ""
        if memories:
            memory_context = "Relevant memories from prior sessions:\n" + "\n".join(
                f"  • {m.content}" for m in memories
            )
            self.log_step("load_session", f"loaded {len(memories)} memories")

        system_msg = SystemMessage(content=(
            "You are an expert CFO assistant running a month-end financial close. "
            "Be precise, data-driven, and flag any anomalies immediately.\n\n"
            + memory_context
        ))

        return {
            "messages":  [system_msg],
            "next_step": "deep_research",
        }

    async def _node_deep_research(self, state: AgentState) -> Dict[str, Any]:
        """
        Invoke the LangGraph deep-research sub-graph.

        Runs the multi-iteration research loop (plan → parallel research →
        synthesise → reflect) and injects the synthesis into the message
        history for downstream nodes.
        """
        self.log_step("deep_research")
        session_dict = state["session"]
        period_data  = session_dict.get("active_period") or {}
        year         = period_data.get("year", datetime.utcnow().year)
        month        = period_data.get("month", datetime.utcnow().month)
        period_label = f"{_MONTH_NAMES[month - 1]} {year}"

        query = (
            f"Perform a comprehensive analysis of the month-end financials for "
            f"{period_label}.  Include: revenue variance vs budget, cost analysis, "
            f"industry benchmarks, cash-flow commentary, and top recommendations."
        )

        research_result = await self._research.run(
            query=query,
            period_label=period_label,
            thread_id=f"research-{session_dict.get('thread_id', uuid.uuid4())}",
        )

        research_tasks = research_result.get("research_tasks", [])
        synthesis      = research_result.get("synthesis", "")

        return {
            "messages":      [AIMessage(content=f"Research complete:\n{synthesis}")],
            "research_tasks": research_tasks,
            "next_step":     "analyse_finances",
        }

    async def _node_analyse_finances(self, state: AgentState) -> Dict[str, Any]:
        """
        Fetch actuals and compute variances using the financial tools.

        Populates the state with a FinancialMetrics snapshot and the full
        variance list so the report builder can use them directly.
        """
        self.log_step("analyse_finances")
        session_dict = state["session"]
        period_data  = session_dict.get("active_period") or {}
        year  = period_data.get("year",  datetime.utcnow().year)
        month = period_data.get("month", datetime.utcnow().month)

        # Use LangChain tools directly (no LLM hop needed for data fetch)
        actuals   = fetch_financial_data.invoke({"year": year, "month": month})
        variances = calculate_variances.invoke({"year": year, "month": month})

        # Identify material variances for the HITL summary
        material = [v for v in variances if v["is_material"]]
        summary  = (
            f"Financials for {_MONTH_NAMES[month-1]} {year}: "
            f"Revenue ${actuals['revenue']:,.0f}, "
            f"Net Income ${actuals['net_income']:,.0f}. "
            f"Material variances: {len(material)}."
        )

        return {
            "messages":  [AIMessage(content=summary)],
            "report":    {
                "actuals":   actuals,
                "variances": variances,
                "year":      year,
                "month":     month,
            },
            "next_step": "sandbox_deepdive",
        }

    async def _node_sandbox_deepdive(self, state: AgentState) -> Dict[str, Any]:
        """
        Run custom financial calculations inside the sandboxed executor.

        The agent generates Python code for bespoke metrics (DSO, DPO,
        working capital ratio) and executes them safely.  Output is
        appended to the report state.
        """
        self.log_step("sandbox_deepdive")
        report_data = state.get("report") or {}
        actuals     = report_data.get("actuals", {})

        # Financial calculations to run in the sandbox
        code = """
# Days Sales Outstanding (DSO)
dso = (accounts_receivable / revenue) * 30 if revenue else 0

# Days Payable Outstanding (DPO)
dpo = (accounts_payable / cost_of_goods_sold) * 30 if cost_of_goods_sold else 0

# Working Capital Ratio
working_capital = accounts_receivable - accounts_payable
working_capital_ratio = (accounts_receivable / accounts_payable) if accounts_payable else 0

# Gross Margin %
gross_margin_pct = ((revenue - cost_of_goods_sold) / revenue * 100) if revenue else 0

# Net Margin %
net_margin_pct = (net_income / revenue * 100) if revenue else 0

result = {
    "dso_days":             round(dso, 1),
    "dpo_days":             round(dpo, 1),
    "working_capital":      round(working_capital, 2),
    "working_capital_ratio": round(working_capital_ratio, 2),
    "gross_margin_pct":     round(gross_margin_pct, 1),
    "net_margin_pct":       round(net_margin_pct, 1),
}
print(f"DSO: {result['dso_days']} days | DPO: {result['dpo_days']} days")
print(f"Gross Margin: {result['gross_margin_pct']}% | Net Margin: {result['net_margin_pct']}%")
"""

        sandbox_result = self._sandbox.run(
            code=code,
            context={
                "revenue":            actuals.get("revenue", 0),
                "cost_of_goods_sold": actuals.get("cost_of_goods_sold", 0),
                "net_income":         actuals.get("net_income", 0),
                "accounts_receivable": actuals.get("accounts_receivable", 0),
                "accounts_payable":    actuals.get("accounts_payable", 0),
            },
        )

        sandbox_outputs = []
        if sandbox_result.success:
            sandbox_outputs = [
                sandbox_result.stdout,
                f"Calculated metrics: {json.dumps(sandbox_result.return_value, indent=2)}",
            ]
            self.log_step("sandbox_deepdive", f"DSO={sandbox_result.return_value.get('dso_days')} days")
        else:
            logger.error("Sandbox execution failed: %s", sandbox_result.stderr)
            sandbox_outputs = [f"Sandbox error: {sandbox_result.stderr}"]

        # Merge sandbox outputs into report state
        existing_report = state.get("report") or {}
        existing_report["sandbox_outputs"] = sandbox_outputs
        existing_report["derived_metrics"] = sandbox_result.return_value or {}

        return {
            "messages": [AIMessage(content=f"Sandbox analysis:\n" + "\n".join(sandbox_outputs))],
            "report":   existing_report,
        }

    async def _node_build_report(self, state: AgentState) -> Dict[str, Any]:
        """
        Assemble the draft MonthEndReport from all accumulated state.

        Combines research synthesis, financial actuals, variance analysis,
        and sandbox-derived metrics into a structured report object.
        """
        self.log_step("build_report")
        report_data  = state.get("report") or {}
        session_dict = state["session"]
        actuals      = report_data.get("actuals", {})
        variances    = report_data.get("variances", [])
        year         = report_data.get("year",  datetime.utcnow().year)
        month        = report_data.get("month", datetime.utcnow().month)
        derived      = report_data.get("derived_metrics", {})
        sandbox_outs = report_data.get("sandbox_outputs", [])

        # Extract research synthesis from messages
        research_synthesis = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, AIMessage) and "Research complete" in m.content),
            "Research synthesis not available.",
        )

        # Build variance objects
        variance_models = [
            {
                "metric_name":   v["metric"],
                "actual":        v["actual"],
                "budget":        v["budget"],
                "prior_period":  v["prior_period"],
                "vs_budget_pct": v["vs_budget_pct"],
                "is_material":   v["is_material"],
            }
            for v in variances
        ]

        insights = [
            f"Revenue of ${actuals.get('revenue', 0):,.0f} "
            f"vs budget ${next((v['budget'] for v in variances if v['metric'] == 'revenue'), 0):,.0f}.",
            f"Net margin: {derived.get('net_margin_pct', 0):.1f}% | "
            f"Gross margin: {derived.get('gross_margin_pct', 0):.1f}%",
            f"DSO: {derived.get('dso_days', 0)} days | DPO: {derived.get('dpo_days', 0)} days",
        ]
        material_items = [v for v in variances if v["is_material"]]
        if material_items:
            names = ", ".join(v["metric"].replace("_", " ") for v in material_items)
            insights.append(f"Material variances in: {names} – investigation recommended.")

        report_dict = {
            "id":                 str(uuid.uuid4()),
            "period":             {"year": year, "month": month},
            "actuals":            actuals,
            "variances":          variance_models,
            "research_synthesis": research_synthesis,
            "insights":           insights,
            "recommendations": [
                "Review adverse variance drivers with department heads.",
                "Monitor DSO trend over the next 30 days.",
                "Compare gross margin to industry median and document gap.",
            ],
            "sandbox_outputs":    sandbox_outs,
            "status":             ReportStatus.IN_REVIEW.value,
            "generated_at":       datetime.utcnow().isoformat(),
        }

        has_material = any(v["is_material"] for v in variances)
        self.log_step("build_report", f"material_variances={has_material}")

        return {
            "messages": [AIMessage(content=f"Report draft assembled. Material variances: {has_material}.")],
            "report":   report_dict,
        }

    async def _node_hitl_checkpoint(self, state: AgentState) -> Dict[str, Any]:
        """
        HITL checkpoint — the graph pauses here for human approval.

        Step 1: Dispatch notifications to Teams and Slack.
        Step 2: Call interrupt() — LangGraph serialises state and suspends
                the graph until Command(resume=...) is called.
        Step 3: Process the resume value (human decision) and route.

        This node is the centrepiece of the HITL pattern.  It demonstrates:
          • Sending rich notifications before pausing
          • interrupt() for true async human-in-the-loop
          • Reading the approval decision from the resumed Command value
        """
        self.log_step("hitl_checkpoint", "dispatching notifications")
        report_data  = state.get("report") or {}
        session_dict = state["session"]
        variances    = report_data.get("variances", [])
        year         = report_data.get("year",  datetime.utcnow().year)
        month        = report_data.get("month", datetime.utcnow().month)
        period_label = f"{_MONTH_NAMES[month - 1]} {year}"

        # Build material variance summary for the notification
        material_variances = [v for v in variances if v.get("is_material")]
        context = {
            "Period":         period_label,
            "Report ID":      report_data.get("id", "?"),
            "Revenue":        f"${report_data.get('actuals', {}).get('revenue', 0):,.0f}",
            "Net Income":     f"${report_data.get('actuals', {}).get('net_income', 0):,.0f}",
            "Material Items": str(len(material_variances)),
        }

        hitl_request = HITLRequest(
            title=f"Month-End Report Approval Required – {period_label}",
            summary=(
                f"The {period_label} month-end close report is ready for review. "
                f"{len(material_variances)} material variance(s) detected. "
                "Finance controller approval required."
            ),
            detail=(
                "Please review the report and approve or reject.  "
                "Rejection will trigger a revision cycle.  "
                "Escalation routes to the CFO."
            ),
            context=context,
            requires_approval=True,
        )

        # Dispatch to Teams + Slack (fire-and-forget – don't block on delivery)
        dispatched = await self._hitl.request_approval(hitl_request)
        self.log_step("hitl_checkpoint", f"request_id={dispatched.id}")

        # ── LangGraph interrupt() – suspends the graph here ──────────────────
        # The dict passed to interrupt() is surfaced to the operator so they
        # know what decision is being awaited.
        raw_response: dict = interrupt({
            "request_id":   dispatched.id,
            "title":        hitl_request.title,
            "period":       period_label,
            "awaiting":     "finance_controller_approval",
        })

        # Graph resumes here after Command(resume=response_dict) is called
        response = HITLResponse(**raw_response)
        self.log_step("hitl_checkpoint", f"decision={response.status}")

        return {
            "messages":     [AIMessage(content=f"HITL response: {response.status.value} by {response.reviewer}.")],
            "hitl_request": dispatched.model_dump(),
            "hitl_response": response.model_dump(),
        }

    async def _node_publish_report(self, state: AgentState) -> Dict[str, Any]:
        """
        Publish the approved report.

        Marks status as PUBLISHED, stores the report summary in AgentCore
        Memory so future sessions can reference it, and updates the session.
        """
        self.log_step("publish_report")
        report_data  = state.get("report") or {}
        session_dict = state["session"]
        year         = report_data.get("year",  datetime.utcnow().year)
        month        = report_data.get("month", datetime.utcnow().month)
        period_label = f"{_MONTH_NAMES[month - 1]} {year}"
        report_id    = report_data.get("id", str(uuid.uuid4()))

        report_data["status"] = ReportStatus.PUBLISHED.value
        report_data["published_at"] = datetime.utcnow().isoformat()

        # Persist a summary to AgentCore Memory for future sessions
        memory_content = (
            f"Month-end report for {period_label} (ID: {report_id}) was APPROVED and published. "
            f"Revenue: {report_data.get('actuals', {}).get('revenue', 0):,.0f} USD. "
            f"Insights: {'; '.join(report_data.get('insights', [])[:2])}"
        )
        await self._agentcore.store_memory(
            user_id=session_dict.get("user_id", "unknown"),
            content=memory_content,
            session_id=session_dict.get("memory_session_id"),
        )

        # Send a completion notification to Teams + Slack
        await self._hitl.send_info_notification(
            title=f"Month-End Report Published – {period_label}",
            summary=f"Report {report_id} has been approved and published.",
            context={"Report ID": report_id, "Period": period_label, "Status": "Published"},
        )

        self.log_step("publish_report", f"report_id={report_id} published")
        return {
            "messages": [AIMessage(content=f"Report {report_id} published successfully.")],
            "report":   report_data,
        }

    async def _node_revise_report(self, state: AgentState) -> Dict[str, Any]:
        """
        Revision node – triggered when the HITL reviewer rejects the report.

        Logs the rejection reason, notifies the finance team, and loops back
        to build_report for a revised draft.
        """
        self.log_step("revise_report")
        response_data = state.get("hitl_response") or {}
        comment       = response_data.get("comment", "No reason given.")
        reviewer      = response_data.get("reviewer", "unknown")

        await self._hitl.send_info_notification(
            title="Month-End Report Revision Required",
            summary=f"Report was rejected by {reviewer}: {comment}",
            context={"Rejected By": reviewer, "Reason": comment},
        )

        return {
            "messages": [AIMessage(content=f"Report rejected by {reviewer}. Revising: {comment}")],
            "hitl_response": None,   # clear so the next HITL cycle starts fresh
        }

    # ── Routing function ──────────────────────────────────────────────────────

    @staticmethod
    def _route_approval(state: AgentState) -> str:
        """
        Read the HITL response from state and return a routing key.

          approved  → 'publish'
          rejected  → 'revise'
          escalated → 'escalate' (loops back to hitl_checkpoint)
          timed_out → 'escalate' (treat as needing re-approval)
        """
        response = state.get("hitl_response") or {}
        status   = response.get("status", ApprovalStatus.PENDING.value)

        routing = {
            ApprovalStatus.APPROVED.value:  "publish",
            ApprovalStatus.REJECTED.value:  "revise",
            ApprovalStatus.ESCALATED.value: "escalate",
            ApprovalStatus.TIMED_OUT.value: "escalate",
        }
        return routing.get(status, "escalate")


# ─────────────────────────────────────────────────────────────────────────────
# Factory function (used by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_month_end_graph() -> MonthEndOrchestrator:
    """Create and return a fully-wired MonthEndOrchestrator."""
    return MonthEndOrchestrator()


# ── Module-level constant ─────────────────────────────────────────────────────

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
