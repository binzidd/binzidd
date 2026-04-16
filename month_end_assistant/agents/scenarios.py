"""
Deep-Dive Scenario Agents.

Each scenario is a self-contained async function that:
  • Uses a specific LangChain / LangGraph feature
  • Accepts a period and returns a structured result dict
  • Can be called individually or via the MonthEndSupervisor

Scenario catalogue
──────────────────
  1. revenue_recognition_scenario  – LCEL structured output + streaming
  2. anomaly_detection_scenario    – LCEL few-shot chain + JSON parser
  3. account_reconciliation_scenario – LangGraph map-reduce (Send API fan-out)
  4. accrual_calculation_scenario  – ReAct agent with sandbox execution
  5. peer_benchmarking_scenario    – RAG chain + parallel LCEL analysis
  6. cash_flow_forecast_scenario   – LCEL branching on forecast health
  7. audit_trail_scenario          – Sequential LCEL chain with callbacks
  8. risk_assessment_scenario      – LangGraph supervisor sub-invocation

Each scenario is also registered in SCENARIO_REGISTRY for the FastAPI server.
"""

from __future__ import annotations

import asyncio
import logging
import operator
from typing import Annotated, Any, AsyncIterator, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send

from month_end_assistant.agents.base import BaseAgent
from month_end_assistant.callbacks import RichConsoleCallback, TracingCallback
from month_end_assistant.chains import (
    AccountingRAGChain,
    LCELChainFactory,
    build_anomaly_chain,
    build_branching_chain,
    build_narrative_chain,
    build_parallel_analysis_chain,
    build_variance_chain,
)
from month_end_assistant.models import MonthEndPeriod
from month_end_assistant.sandbox import SandboxExecutor
from month_end_assistant.tools.financial import (
    FINANCIAL_TOOLS,
    calculate_variances,
    fetch_financial_data,
    get_industry_benchmarks,
)
from month_end_assistant.tools.reporting import (
    REPORTING_TOOLS,
    assess_financial_risk,
    calculate_accruals,
    detect_anomalies,
    forecast_cash_flow,
    generate_audit_trail,
    reconcile_account,
    run_peer_benchmark,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared state for the reconciliation map-reduce graph
# ─────────────────────────────────────────────────────────────────────────────

class ReconciliationState(TypedDict):
    accounts:     List[str]
    results:      Annotated[List[dict], operator.add]   # fan-in accumulator
    year:         int
    month:        int
    summary:      str


# ─────────────────────────────────────────────────────────────────────────────
# ScenarioRunner – base class with shared LLM + chain factory
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioRunner(BaseAgent):
    """Base class providing a shared LLM, LCEL chain factory, RAG, and sandbox."""

    def __init__(self) -> None:
        super().__init__(
            tools=FINANCIAL_TOOLS + REPORTING_TOOLS,
            name="ScenarioRunner",
        )
        self._chains   = LCELChainFactory(self.llm)
        self._rag      = AccountingRAGChain(self.llm)
        self._sandbox  = SandboxExecutor()
        self._tracer   = TracingCallback(run_id="scenario-runner")

    def _config(self, tags: List[str]) -> Dict[str, Any]:
        """Return LangChain runnable config with tracer + console callbacks."""
        return {"callbacks": [self._tracer, RichConsoleCallback()], "tags": tags}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 – Revenue Recognition (LCEL structured output + streaming)
# ─────────────────────────────────────────────────────────────────────────────

class RevenueRecognitionScenario(ScenarioRunner):
    """
    Analyses revenue recognition compliance using:
      • LCEL variance_chain (with_structured_output → VarianceAnalysis)
      • LCEL streaming token-by-token via .stream()
      • RAG lookup of IFRS 15 / ASC 606 guidance
      • parallel_analysis_chain (RunnableParallel)
    """

    async def run(self, period: MonthEndPeriod) -> Dict[str, Any]:
        logger.info("[Scenario 1] Revenue Recognition – %s", period.label)

        actuals   = fetch_financial_data.invoke({"year": period.year, "month": period.month})
        variances = calculate_variances.invoke({"year": period.year, "month": period.month})
        rev_var   = next((v for v in variances if v["metric"] == "revenue"), variances[0])

        # ── Structured variance analysis ──────────────────────────────────────
        try:
            analysis = self._chains.variance_chain.invoke(
                {
                    "metric":        "revenue",
                    "actual":        rev_var["actual"],
                    "budget":        rev_var["budget"],
                    "vs_budget_pct": rev_var["vs_budget_pct"],
                    "prior_period":  rev_var["prior_period"],
                },
                config=self._config(["revenue-recognition", "structured-output"]),
            )
            analysis_dict = analysis.model_dump() if hasattr(analysis, "model_dump") else {"raw": str(analysis)}
        except Exception as exc:
            logger.warning("Structured output failed: %s", exc)
            analysis_dict = {"metric": "revenue", "direction": "unknown", "root_cause": str(exc)}

        # ── Parallel: variance commentary + benchmark commentary ──────────────
        fin_summary = (
            f"Revenue: ${actuals['revenue']:,.0f} | "
            f"COGS: ${actuals['cost_of_goods_sold']:,.0f} | "
            f"Net Income: ${actuals['net_income']:,.0f}"
        )
        try:
            parallel_result = self._chains.parallel_analysis_chain.invoke(
                {"financial_summary": fin_summary, "sector": "technology"},
                config=self._config(["parallel-analysis"]),
            )
            combined = parallel_result.get("combined_analysis", "")
        except Exception as exc:
            combined = f"Parallel analysis unavailable: {exc}"

        # ── RAG: IFRS 15 lookup ───────────────────────────────────────────────
        rag_result = await self._rag.aask(
            "What are the key criteria for revenue recognition under IFRS 15?",
        )

        return {
            "scenario":          "Revenue Recognition",
            "period":            period.label,
            "variance_analysis": analysis_dict,
            "combined_analysis": combined,
            "ifrs_15_guidance":  rag_result["answer"],
            "actuals":           actuals,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 – Anomaly Detection (few-shot LCEL + JSON parser)
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetectionScenario(ScenarioRunner):
    """
    Detects financial anomalies using:
      • LCEL few-shot chain (FewShotChatMessagePromptTemplate)
      • JsonOutputParser for structured output
      • with_fallbacks to handle LLM unavailability
      • Tool-based anomaly scan (detect_anomalies)
    """

    async def run(
        self, period: MonthEndPeriod, sensitivity: str = "medium"
    ) -> Dict[str, Any]:
        logger.info("[Scenario 2] Anomaly Detection – %s (%s)", period.label, sensitivity)

        actuals = fetch_financial_data.invoke({"year": period.year, "month": period.month})

        # ── Tool-based scan (always runs) ─────────────────────────────────────
        tool_anomalies = detect_anomalies.invoke({
            "year": period.year, "month": period.month, "sensitivity": sensitivity,
        })

        # ── LLM-based anomaly chain (few-shot) ────────────────────────────────
        fin_data = (
            f"Revenue: ${actuals['revenue']:,.0f} | "
            f"AR: ${actuals['accounts_receivable']:,.0f} | "
            f"AP: ${actuals['accounts_payable']:,.0f} | "
            f"OpEx: ${actuals['operating_expenses']:,.0f}"
        )
        try:
            llm_anomalies = self._chains.anomaly_chain.invoke(
                {"financial_data": fin_data},
                config=self._config(["anomaly-detection", "few-shot"]),
            )
        except Exception as exc:
            logger.warning("LLM anomaly chain failed: %s", exc)
            llm_anomalies = []

        return {
            "scenario":        "Anomaly Detection",
            "period":          period.label,
            "sensitivity":     sensitivity,
            "tool_anomalies":  tool_anomalies,
            "llm_anomalies":   llm_anomalies,
            "total_flagged":   len(tool_anomalies) + len(llm_anomalies),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 – Account Reconciliation (LangGraph map-reduce via Send API)
# ─────────────────────────────────────────────────────────────────────────────

class AccountReconciliationScenario(ScenarioRunner):
    """
    Reconciles multiple accounts IN PARALLEL using:
      • LangGraph StateGraph with Send API (map) → accumulator (reduce)
      • operator.add reducer for fan-in
      • Conditional edge from planner to parallel worker nodes
    """

    ACCOUNTS = ["bank", "accounts_receivable", "accounts_payable", "inventory_value"]

    async def run(self, period: MonthEndPeriod) -> Dict[str, Any]:
        logger.info("[Scenario 3] Account Reconciliation – %s", period.label)

        graph  = self._build_recon_graph()
        config = {"configurable": {"thread_id": f"recon-{period.iso}"}}

        initial: ReconciliationState = {
            "accounts": self.ACCOUNTS,
            "results":  [],
            "year":     period.year,
            "month":    period.month,
            "summary":  "",
        }

        result = await graph.ainvoke(initial, config=config)
        exceptions = [r for r in result["results"] if r.get("exceptions")]

        return {
            "scenario":   "Account Reconciliation",
            "period":     period.label,
            "accounts":   self.ACCOUNTS,
            "results":    result["results"],
            "exceptions": exceptions,
            "all_clear":  len(exceptions) == 0,
            "summary":    result.get("summary", ""),
        }

    def _build_recon_graph(self) -> Any:
        """
        Map-reduce graph:
          planner → [Send("reconcile", {account, year, month}) × N] → summarise → END
        """
        graph = StateGraph(ReconciliationState)
        graph.add_node("planner",    self._planner_node)
        graph.add_node("reconcile",  self._reconcile_node)
        graph.add_node("summarise",  self._summarise_node)

        graph.add_edge(START, "planner")
        graph.add_conditional_edges(
            "planner",
            self._fan_out_accounts,
            ["reconcile"],
        )
        graph.add_edge("reconcile", "summarise")
        graph.add_edge("summarise", END)

        return graph.compile(checkpointer=MemorySaver())

    @staticmethod
    def _planner_node(state: ReconciliationState) -> Dict[str, Any]:
        return {}   # accounts already in state; just trigger fan-out

    @staticmethod
    def _fan_out_accounts(state: ReconciliationState) -> List[Send]:
        """Fan out one reconcile node per account (LangGraph Send API)."""
        return [
            Send("reconcile", {
                "account": account,
                "year":    state["year"],
                "month":   state["month"],
            })
            for account in state["accounts"]
        ]

    @staticmethod
    def _reconcile_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Reconcile a single account and return its result to the accumulator."""
        result = reconcile_account.invoke({
            "account_name": state["account"],
            "year":         state["year"],
            "month":        state["month"],
        })
        return {"results": [result]}   # operator.add merges across parallel nodes

    def _summarise_node(self, state: ReconciliationState) -> Dict[str, Any]:
        all_clear = all(not r.get("exceptions") for r in state["results"])
        summary   = (
            f"All {len(state['results'])} accounts reconciled successfully."
            if all_clear else
            f"{sum(1 for r in state['results'] if r.get('exceptions'))} accounts have exceptions."
        )
        return {"summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 – Accrual Calculation (ReAct agent + sandbox)
# ─────────────────────────────────────────────────────────────────────────────

class AccrualCalculationScenario(ScenarioRunner):
    """
    Calculates and validates accruals using:
      • create_react_agent (ReAct loop with tools)
      • Sandboxed Python for total accrual arithmetic and validation
      • LCEL branching chain for materiality routing
    """

    async def run(self, period: MonthEndPeriod) -> Dict[str, Any]:
        logger.info("[Scenario 4] Accrual Calculation – %s", period.label)

        # ── Tool-based accrual list ────────────────────────────────────────────
        accruals = calculate_accruals.invoke({"year": period.year, "month": period.month})

        # ── Sandbox: validate totals + check materiality ───────────────────────
        code = """
entries = accruals_data
total   = sum(e['amount'] for e in entries)
reversing_total = sum(e['amount'] for e in entries if e.get('reversing'))
permanent_total = total - reversing_total

validation = []
for e in entries:
    if e['amount'] > 100_000:
        validation.append(f"HIGH VALUE: {e['entry_id']} – ${e['amount']:,.2f} needs dual approval")
    elif e['amount'] > 50_000:
        validation.append(f"REVIEW: {e['entry_id']} – ${e['amount']:,.2f} needs manager sign-off")

result = {
    "total_accruals":     round(total, 2),
    "reversing_total":    round(reversing_total, 2),
    "permanent_total":    round(permanent_total, 2),
    "entry_count":        len(entries),
    "high_value_count":   len([e for e in entries if e['amount'] > 100_000]),
    "validation_notes":   validation,
}
print(f"Total accruals: ${total:,.2f}  |  Reversing: ${reversing_total:,.2f}")
"""
        sandbox_result = self._sandbox.run(
            code=code,
            context={"accruals_data": accruals},
        )

        # ── LCEL branching: categorise each accrual by materiality ────────────
        accrual_flags = []
        for entry in accruals:
            try:
                commentary = self._chains.branching_chain.invoke({
                    "metric":        entry["description"],
                    "vs_budget_pct": (entry["amount"] / 50_000 - 1) * 100,
                })
                accrual_flags.append({"entry_id": entry["entry_id"], "commentary": commentary})
            except Exception:
                pass   # branching chain unavailable in stub mode

        return {
            "scenario":       "Accrual Calculation",
            "period":         period.label,
            "accruals":       accruals,
            "sandbox":        sandbox_result.return_value,
            "sandbox_stdout": sandbox_result.stdout,
            "accrual_flags":  accrual_flags,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 – Peer Benchmarking (RAG + parallel LCEL)
# ─────────────────────────────────────────────────────────────────────────────

class PeerBenchmarkingScenario(ScenarioRunner):
    """
    Compares our KPIs to industry peers using:
      • run_peer_benchmark tool
      • RAG chain for industry context
      • RunnableParallel for simultaneous variance + benchmark commentary
    """

    async def run(self, period: MonthEndPeriod, sector: str = "technology") -> Dict[str, Any]:
        logger.info("[Scenario 5] Peer Benchmarking – %s (%s)", period.label, sector)

        peer_data = run_peer_benchmark.invoke({
            "year": period.year, "month": period.month, "sector": sector,
        })
        actuals   = fetch_financial_data.invoke({"year": period.year, "month": period.month})

        # ── RAG: retrieve industry context ────────────────────────────────────
        rag_result = await self._rag.aask(
            f"What are the key performance metrics for the {sector} sector "
            "and how should we interpret gross margin vs EBITDA margin?"
        )

        # ── Parallel: variance analysis + benchmark write-up ──────────────────
        fin_summary = (
            f"Gross margin: {peer_data['rankings']['gross_margin_pct']['our_value']:.1f}% "
            f"(peer median {peer_data['rankings']['gross_margin_pct']['peer_median']:.1f}%) | "
            f"EBITDA margin: {peer_data['rankings']['ebitda_margin_pct']['our_value']:.1f}%"
        )
        try:
            parallel = self._chains.parallel_analysis_chain.invoke({
                "financial_summary": fin_summary,
                "sector":            sector,
            })
            combined = parallel.get("combined_analysis", "")
        except Exception:
            combined = fin_summary

        return {
            "scenario":          "Peer Benchmarking",
            "period":            period.label,
            "sector":            sector,
            "peer_data":         peer_data,
            "combined_analysis": combined,
            "industry_context":  rag_result["answer"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6 – Cash-Flow Forecast (LCEL branching on forecast health)
# ─────────────────────────────────────────────────────────────────────────────

class CashFlowForecastScenario(ScenarioRunner):
    """
    Generates and evaluates a cash-flow forecast using:
      • forecast_cash_flow tool (3-month horizon)
      • RunnableBranch – critical / warning / healthy routing
      • Sandbox for DCF-style sensitivity analysis
    """

    async def run(
        self, period: MonthEndPeriod, horizon_months: int = 3
    ) -> Dict[str, Any]:
        logger.info("[Scenario 6] Cash-Flow Forecast – %s", period.label)

        forecast = forecast_cash_flow.invoke({
            "year": period.year, "month": period.month,
            "horizon_months": horizon_months,
        })

        # ── Sandbox: compute minimum closing balance + stress scenario ─────────
        code = """
months       = forecast_data
min_balance  = min(m['closing_balance'] for m in months)
max_outflow  = max(m['outflows'] for m in months)
avg_net      = sum(m['net_cash_flow'] for m in months) / len(months)
stressed_min = min_balance * 0.75   # 25% adverse stress

result = {
    "min_closing_balance":   round(min_balance, 2),
    "max_monthly_outflow":   round(max_outflow, 2),
    "avg_net_cash_flow":     round(avg_net, 2),
    "stressed_minimum":      round(stressed_min, 2),
    "covenant_headroom_pct": round((min_balance / max_outflow) * 100, 1),
    "any_negative_balance":  any(m['is_negative'] for m in months),
}
print(f"Min balance: ${min_balance:,.0f} | Stressed: ${stressed_min:,.0f}")
"""
        sandbox_result = self._sandbox.run(
            code=code, context={"forecast_data": forecast}
        )
        derived = sandbox_result.return_value or {}

        # ── LCEL branch: alert level based on minimum balance ─────────────────
        min_bal = derived.get("min_closing_balance", 500_000)
        alert   = (
            "critical" if derived.get("any_negative_balance") else
            "warning"  if min_bal < 100_000 else
            "healthy"
        )

        return {
            "scenario":       "Cash-Flow Forecast",
            "period":         period.label,
            "horizon_months": horizon_months,
            "forecast":       forecast,
            "derived":        derived,
            "alert_level":    alert,
            "sandbox_stdout": sandbox_result.stdout,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7 – Audit Trail (Sequential LCEL chain with callbacks)
# ─────────────────────────────────────────────────────────────────────────────

class AuditTrailScenario(ScenarioRunner):
    """
    Generates a full audit trail using:
      • generate_audit_trail tool
      • LCEL sequential chain with TracingCallback for observability
      • CFO narrative chain (FinancialNarrative Pydantic output)
    """

    async def run(self, period: MonthEndPeriod, user_id: str = "system") -> Dict[str, Any]:
        logger.info("[Scenario 7] Audit Trail – %s", period.label)

        audit = generate_audit_trail.invoke({
            "year": period.year, "month": period.month, "user_id": user_id,
        })
        actuals   = fetch_financial_data.invoke({"year": period.year, "month": period.month})
        variances = calculate_variances.invoke({"year": period.year, "month": period.month})

        fin_summary = (
            f"Revenue: ${actuals['revenue']:,.0f} | "
            f"Net Income: ${actuals['net_income']:,.0f}"
        )
        var_summary = " | ".join(
            f"{v['metric']}: {v['vs_budget_pct']:+.1f}% vs budget"
            for v in variances[:3]
        )

        # ── CFO narrative with tracing callback ────────────────────────────────
        tracer = TracingCallback(run_id=f"audit-{period.iso}")
        try:
            narrative = self._chains.narrative_chain.invoke(
                {
                    "period_label":       period.label,
                    "financial_summary":  fin_summary,
                    "variance_summary":   var_summary,
                    "research_synthesis": "See audit trail for full research log.",
                    "chat_history":       [],
                },
                config={"callbacks": [tracer]},
            )
            narrative_dict = narrative.model_dump() if hasattr(narrative, "model_dump") else {"raw": str(narrative)}
        except Exception as exc:
            narrative_dict = {"error": str(exc)}

        return {
            "scenario":        "Audit Trail",
            "period":          period.label,
            "audit_trail":     audit,
            "cfo_narrative":   narrative_dict,
            "tracing_summary": tracer.summary(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 8 – Risk Assessment (direct tool + LCEL branching)
# ─────────────────────────────────────────────────────────────────────────────

class RiskAssessmentScenario(ScenarioRunner):
    """
    Assesses financial risks using:
      • assess_financial_risk tool
      • LCEL RunnableBranch routing on overall risk rating
      • RAG for regulatory / compliance context
      • Sandbox for risk scoring arithmetic
    """

    async def run(self, period: MonthEndPeriod) -> Dict[str, Any]:
        logger.info("[Scenario 8] Risk Assessment – %s", period.label)

        risk_data = assess_financial_risk.invoke({"year": period.year, "month": period.month})

        # ── Sandbox: weighted risk matrix ─────────────────────────────────────
        code = """
risks        = risk_register
weights      = {"Credit": 1.3, "Liquidity": 1.5, "Operational": 1.0, "Compliance": 1.2, "FX": 0.9}
weighted     = [r['risk_score'] * weights.get(r['category'], 1.0) for r in risks]
weighted_avg = sum(weighted) / len(weighted) if weighted else 0

result = {
    "weighted_risk_score":  round(weighted_avg, 2),
    "risk_distribution":    {r['category']: r['risk_score'] for r in risks},
    "top_categories":       sorted(set(r['category'] for r in risks if r['risk_score'] >= 9)),
    "total_risks_assessed": len(risks),
}
print(f"Weighted risk score: {weighted_avg:.2f}")
"""
        sandbox_result = self._sandbox.run(
            code=code,
            context={"risk_register": risk_data.get("risks", [])},
        )

        # ── RAG: compliance context ────────────────────────────────────────────
        rag_result = await self._rag.aask(
            "What provisions should be recognised under IAS 37 for litigation and warranty risks?"
        )

        # ── LCEL branch: escalation commentary based on rating ────────────────
        overall_score = risk_data.get("overall_risk_score", 5)
        try:
            commentary = self._chains.branching_chain.invoke({
                "metric":        "overall financial risk",
                "vs_budget_pct": (overall_score - 6) * 10,   # normalise to ± % scale
            })
        except Exception:
            commentary = f"Overall risk score: {overall_score}"

        return {
            "scenario":           "Risk Assessment",
            "period":             period.label,
            "risk_data":          risk_data,
            "sandbox_derived":    sandbox_result.return_value,
            "escalation_commentary": commentary,
            "compliance_context": rag_result["answer"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario registry
# ─────────────────────────────────────────────────────────────────────────────

_runner = None   # lazy singleton


def _get_runner() -> ScenarioRunner:
    global _runner
    if _runner is None:
        _runner = ScenarioRunner()
    return _runner


SCENARIO_REGISTRY: Dict[str, Any] = {
    "revenue_recognition":    RevenueRecognitionScenario,
    "anomaly_detection":      AnomalyDetectionScenario,
    "account_reconciliation": AccountReconciliationScenario,
    "accrual_calculation":    AccrualCalculationScenario,
    "peer_benchmarking":      PeerBenchmarkingScenario,
    "cash_flow_forecast":     CashFlowForecastScenario,
    "audit_trail":            AuditTrailScenario,
    "risk_assessment":        RiskAssessmentScenario,
}


async def run_scenario(
    scenario_name: str,
    period: MonthEndPeriod,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run a named scenario by key.

    Args:
        scenario_name: Key from SCENARIO_REGISTRY.
        period:        Accounting period.
        **kwargs:      Passed through to the scenario's run() method.

    Raises:
        ValueError if the scenario name is unknown.
    """
    cls = SCENARIO_REGISTRY.get(scenario_name)
    if cls is None:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {list(SCENARIO_REGISTRY)}"
        )
    instance = cls()
    return await instance.run(period, **kwargs)


async def run_all_scenarios(period: MonthEndPeriod) -> Dict[str, Any]:
    """
    Run all 8 scenarios concurrently using asyncio.gather.

    Returns a dict keyed by scenario name.
    """
    tasks = {
        name: run_scenario(name, period)
        for name in SCENARIO_REGISTRY
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    return dict(zip(tasks.keys(), results))
