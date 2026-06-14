"""
LangChain tools available to every agent node in the graph.

Each tool is decorated with @tool so LangGraph's ReAct agents can call them
automatically.  In production these would hit real data sources (ERP, BI
warehouse, Bloomberg, etc.).  Here they return realistic simulated data so
the demo runs without external dependencies.

Tool catalogue
──────────────
  fetch_financial_data       – load actuals for a period
  calculate_variances        – compute actual-vs-budget/prior variances
  lookup_accounting_standard – search IFRS / GAAP guidance
  get_industry_benchmarks    – retrieve sector KPI benchmarks
  generate_narrative         – turn raw numbers into a plain-English paragraph
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────────────────────
# Simulated data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _seed_from_period(year: int, month: int) -> int:
    """Deterministic random seed so the same period always returns the same data."""
    return year * 100 + month


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def fetch_financial_data(year: int, month: int) -> Dict[str, Any]:
    """
    Fetch actual financial data for a given accounting period.

    Args:
        year:  The four-digit year (e.g. 2025).
        month: The month number 1–12.

    Returns:
        A dict containing revenue, COGS, operating expenses, net income,
        cash flow, AR, AP, and inventory figures in USD.
    """
    rng = random.Random(_seed_from_period(year, month))
    revenue           = round(rng.uniform(3_500_000, 5_500_000), 2)
    cogs              = round(revenue * rng.uniform(0.38, 0.48), 2)
    opex              = round(revenue * rng.uniform(0.28, 0.35), 2)
    net_income        = round(revenue - cogs - opex, 2)
    cash_flow         = round(net_income * rng.uniform(0.85, 1.15), 2)
    ar                = round(revenue * rng.uniform(0.18, 0.28), 2)
    ap                = round(cogs   * rng.uniform(0.12, 0.22), 2)
    inventory         = round(cogs   * rng.uniform(0.30, 0.45), 2)

    return {
        "period": {"year": year, "month": month},
        "revenue": revenue,
        "cost_of_goods_sold": cogs,
        "operating_expenses": opex,
        "gross_profit": revenue - cogs,
        "ebitda": revenue - cogs - opex,
        "net_income": net_income,
        "cash_flow_operations": cash_flow,
        "accounts_receivable": ar,
        "accounts_payable": ap,
        "inventory_value": inventory,
    }


@tool
def calculate_variances(
    year: int, month: int, budget_uplift_pct: float = 3.0
) -> List[Dict[str, Any]]:
    """
    Calculate actual-vs-budget and actual-vs-prior-period variances.

    Args:
        year:             The period year.
        month:            The period month (1–12).
        budget_uplift_pct: How much higher the budget is vs actuals on average.

    Returns:
        A list of variance dicts – one per key metric – containing
        actual, budget, prior_period, vs_budget_pct, and vs_prior_pct.
    """
    actuals      = fetch_financial_data.invoke({"year": year, "month": month})
    prior_month  = month - 1 if month > 1 else 12
    prior_year   = year if month > 1 else year - 1
    prior        = fetch_financial_data.invoke({"year": prior_year, "month": prior_month})

    metrics = ["revenue", "cost_of_goods_sold", "operating_expenses",
               "net_income", "cash_flow_operations"]
    variances = []
    for metric in metrics:
        actual      = actuals[metric]
        budget      = round(actual * (1 + budget_uplift_pct / 100), 2)
        prior_val   = prior[metric]
        vs_budget   = round(((actual - budget) / abs(budget)) * 100, 2) if budget else 0
        vs_prior    = round(((actual - prior_val) / abs(prior_val)) * 100, 2) if prior_val else 0
        variances.append({
            "metric": metric,
            "actual": actual,
            "budget": budget,
            "prior_period": prior_val,
            "vs_budget_pct": vs_budget,
            "vs_prior_pct": vs_prior,
            "is_material": abs(vs_budget) > 5.0,
        })
    return variances


@tool
def lookup_accounting_standard(topic: str) -> str:
    """
    Look up relevant IFRS / US GAAP guidance for a financial topic.

    Args:
        topic: A keyword such as 'revenue recognition', 'lease accounting',
               'impairment', 'inventory valuation', etc.

    Returns:
        A concise summary of the applicable standard and key requirements.
    """
    standards: Dict[str, str] = {
        "revenue recognition": (
            "IFRS 15 / ASC 606 – Revenue is recognised when (or as) performance "
            "obligations are satisfied.  Five-step model: (1) identify contracts, "
            "(2) identify performance obligations, (3) determine transaction price, "
            "(4) allocate price, (5) recognise revenue."
        ),
        "lease accounting": (
            "IFRS 16 / ASC 842 – Lessees recognise a right-of-use asset and a "
            "corresponding lease liability for most leases.  Short-term leases "
            "(≤12 months) and low-value asset leases may use the practical expedient."
        ),
        "impairment": (
            "IAS 36 / ASC 350 – Assets must be tested for impairment when indicators "
            "exist.  The recoverable amount is the higher of fair value less costs of "
            "disposal and value in use.  Goodwill must be tested annually."
        ),
        "inventory valuation": (
            "IAS 2 / ASC 330 – Inventories are measured at the lower of cost and net "
            "realisable value.  IFRS prohibits LIFO; US GAAP permits LIFO, FIFO, or "
            "weighted average."
        ),
        "provisions": (
            "IAS 37 / ASC 450 – A provision is recognised when there is a present "
            "obligation, an outflow is probable, and the amount can be reliably estimated."
        ),
    }
    topic_lower = topic.lower()
    for key, guidance in standards.items():
        if key in topic_lower or any(word in topic_lower for word in key.split()):
            return guidance
    return (
        f"No specific standard found for '{topic}'.  "
        "Consult the IASB or FASB guidance directly for the applicable standard."
    )


@tool
def get_industry_benchmarks(sector: str, metric: str) -> Dict[str, Any]:
    """
    Retrieve industry benchmark ranges for a given sector and financial metric.

    Args:
        sector: Industry sector, e.g. 'technology', 'retail', 'manufacturing'.
        metric: KPI name, e.g. 'gross_margin_pct', 'ebitda_margin_pct',
                'days_sales_outstanding'.

    Returns:
        A dict with p25, median, and p75 benchmark values plus commentary.
    """
    benchmarks: Dict[str, Dict[str, Dict[str, Any]]] = {
        "technology": {
            "gross_margin_pct":   {"p25": 55, "median": 68, "p75": 79, "unit": "%"},
            "ebitda_margin_pct":  {"p25": 12, "median": 22, "p75": 31, "unit": "%"},
            "days_sales_outstanding": {"p25": 38, "median": 52, "p75": 67, "unit": "days"},
        },
        "retail": {
            "gross_margin_pct":   {"p25": 28, "median": 35, "p75": 44, "unit": "%"},
            "ebitda_margin_pct":  {"p25":  4, "median":  8, "p75": 13, "unit": "%"},
            "days_sales_outstanding": {"p25":  8, "median": 14, "p75": 22, "unit": "days"},
        },
        "manufacturing": {
            "gross_margin_pct":   {"p25": 24, "median": 32, "p75": 42, "unit": "%"},
            "ebitda_margin_pct":  {"p25":  7, "median": 13, "p75": 19, "unit": "%"},
            "days_sales_outstanding": {"p25": 35, "median": 48, "p75": 62, "unit": "days"},
        },
    }
    sector_data = benchmarks.get(sector.lower(), benchmarks["technology"])
    data = sector_data.get(metric.lower(), {"p25": "N/A", "median": "N/A", "p75": "N/A", "unit": ""})
    return {
        "sector": sector,
        "metric": metric,
        **data,
        "commentary": (
            f"Industry {metric} benchmark for {sector}: "
            f"25th pct = {data['p25']}{data['unit']}, "
            f"median = {data['median']}{data['unit']}, "
            f"75th pct = {data['p75']}{data['unit']}."
        ),
    }


@tool
def generate_narrative(metrics: Dict[str, float], variances: List[Dict]) -> str:
    """
    Generate a plain-English management narrative from raw figures.

    Args:
        metrics:   Dict of metric_name → actual value.
        variances: List of variance dicts from calculate_variances.

    Returns:
        A two-paragraph narrative suitable for the CFO dashboard.
    """
    material = [v for v in variances if v.get("is_material")]
    favourable = [v for v in material if v["vs_budget_pct"] > 0]
    adverse    = [v for v in material if v["vs_budget_pct"] < 0]

    revenue       = metrics.get("revenue", 0)
    net_income    = metrics.get("net_income", 0)
    margin_pct    = round((net_income / revenue * 100), 1) if revenue else 0

    para1 = (
        f"The period closed with revenue of ${revenue:,.0f} and net income of "
        f"${net_income:,.0f}, representing a net margin of {margin_pct}%. "
    )
    if favourable:
        names = ", ".join(v["metric"].replace("_", " ") for v in favourable)
        para1 += f"Favourable variances were recorded in {names}. "
    if adverse:
        names = ", ".join(v["metric"].replace("_", " ") for v in adverse)
        para1 += f"Adverse variances require management attention in {names}."

    para2 = (
        "Overall performance is in line with strategic targets. "
        "The finance team recommends reviewing the cost structure in areas with "
        "adverse variances and accelerating collection efforts to improve cash flow."
    )
    return f"{para1}\n\n{para2}"


# ── Tool registry exposed to agent nodes ────────────────────────────────────

FINANCIAL_TOOLS = [
    fetch_financial_data,
    calculate_variances,
    lookup_accounting_standard,
    get_industry_benchmarks,
    generate_narrative,
]
