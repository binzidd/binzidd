"""
Extended Financial Tools for Deep-Dive Scenarios.

These LangChain @tools are used by the scenario agents and the supervisor:
  detect_anomalies        – flag unusual transactions / metric outliers
  calculate_accruals      – compute month-end accrual journal entries
  generate_audit_trail    – produce a structured audit-trail document
  assess_financial_risk   – score and categorise financial risks
  forecast_cash_flow      – simple 3-month rolling cash-flow forecast
  reconcile_account       – perform a mock account reconciliation
  run_peer_benchmark      – compare KPIs against peer group
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

@tool
def detect_anomalies(
    year: int, month: int, sensitivity: str = "medium"
) -> List[Dict[str, Any]]:
    """
    Scan the period's financial data for statistical and rule-based anomalies.

    Args:
        year:        Accounting period year.
        month:       Accounting period month.
        sensitivity: Detection sensitivity – 'low' | 'medium' | 'high'.

    Returns:
        List of anomaly dicts, each with type, description, severity, and recommendation.
    """
    rng = random.Random(year * 100 + month)
    thresholds = {"low": 0.20, "medium": 0.10, "high": 0.05}
    threshold  = thresholds.get(sensitivity, 0.10)

    # Simulated anomaly library – in production, compare against statistical baselines
    candidate_anomalies = [
        {
            "anomaly_type":       "duplicate invoice",
            "description":        f"Invoice #INV-{rng.randint(1000,9999)} appears twice in AP ledger.",
            "severity":           "high",
            "affected_line":      "accounts_payable",
            "amount":             round(rng.uniform(5_000, 50_000), 2),
            "recommended_review": "Run duplicate-payment query in ERP; void and repost if confirmed.",
        },
        {
            "anomaly_type":       "unusual journal entry",
            "description":        "Manual JE posted at 23:47 on the last day of the period with no approver.",
            "severity":           "critical",
            "affected_line":      "revenue",
            "amount":             round(rng.uniform(50_000, 250_000), 2),
            "recommended_review": "Identify posting user; obtain controller sign-off or reverse entry.",
        },
        {
            "anomaly_type":       "AR aging spike",
            "description":        "Accounts receivable > 90 days increased by 35% vs prior month.",
            "severity":           "medium",
            "affected_line":      "accounts_receivable",
            "amount":             round(rng.uniform(80_000, 300_000), 2),
            "recommended_review": "Request collections team aging report; consider bad-debt provision.",
        },
        {
            "anomaly_type":       "inventory write-down trigger",
            "description":        "3 SKUs have NRV below cost – write-down required under IAS 2.",
            "severity":           "medium",
            "affected_line":      "inventory_value",
            "amount":             round(rng.uniform(10_000, 60_000), 2),
            "recommended_review": "Obtain warehouse valuation; post write-down before period close.",
        },
    ]

    # Return all anomalies above the sensitivity threshold
    selected = candidate_anomalies if sensitivity == "high" else candidate_anomalies[:rng.randint(1, 3)]
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Accrual Calculation
# ─────────────────────────────────────────────────────────────────────────────

@tool
def calculate_accruals(year: int, month: int) -> List[Dict[str, Any]]:
    """
    Calculate standard month-end accrual journal entries.

    Args:
        year:  Accounting period year.
        month: Accounting period month (1–12).

    Returns:
        List of accrual journal entry dicts with debit, credit, and amount.
    """
    rng = random.Random(year * 100 + month + 7)

    accruals = [
        {
            "entry_id":    "ACR-001",
            "description": "Accrued payroll – last week of period not yet paid",
            "debit_account":  "6100 – Salaries Expense",
            "credit_account": "2100 – Accrued Payroll",
            "amount":         round(rng.uniform(180_000, 250_000), 2),
            "standard":       "IAS 19 / ASC 710",
            "reversing":      True,
        },
        {
            "entry_id":    "ACR-002",
            "description": "Accrued interest on revolving credit facility",
            "debit_account":  "7200 – Interest Expense",
            "credit_account": "2300 – Accrued Interest",
            "amount":         round(rng.uniform(8_000, 25_000), 2),
            "standard":       "IFRS 9 / ASC 835",
            "reversing":      True,
        },
        {
            "entry_id":    "ACR-003",
            "description": "Warranty provision – estimated claims for period sales",
            "debit_account":  "6800 – Warranty Expense",
            "credit_account": "2500 – Warranty Provision",
            "amount":         round(rng.uniform(15_000, 45_000), 2),
            "standard":       "IAS 37 / ASC 460",
            "reversing":      False,
        },
        {
            "entry_id":    "ACR-004",
            "description": "Prepaid insurance amortisation",
            "debit_account":  "6400 – Insurance Expense",
            "credit_account": "1300 – Prepaid Insurance",
            "amount":         round(rng.uniform(3_500, 8_000), 2),
            "standard":       "IAS 38 / ASC 340",
            "reversing":      False,
        },
        {
            "entry_id":    "ACR-005",
            "description": "Depreciation – PP&E monthly charge",
            "debit_account":  "6600 – Depreciation Expense",
            "credit_account": "1700 – Accumulated Depreciation",
            "amount":         round(rng.uniform(22_000, 65_000), 2),
            "standard":       "IAS 16 / ASC 360",
            "reversing":      False,
        },
        {
            "entry_id":    "ACR-006",
            "description": "IFRS 16 – Right-of-use asset depreciation",
            "debit_account":  "6600 – Depreciation Expense",
            "credit_account": "1800 – Accumulated ROU Depreciation",
            "amount":         round(rng.uniform(12_000, 35_000), 2),
            "standard":       "IFRS 16",
            "reversing":      False,
        },
    ]
    return accruals


# ─────────────────────────────────────────────────────────────────────────────
# Audit Trail Generation
# ─────────────────────────────────────────────────────────────────────────────

@tool
def generate_audit_trail(year: int, month: int, user_id: str = "system") -> Dict[str, Any]:
    """
    Generate a structured audit-trail document for the month-end close process.

    Args:
        year:    Accounting period year.
        month:   Accounting period month.
        user_id: Identity of the user who triggered the close.

    Returns:
        Audit-trail dict with timeline of events, approvals, and sign-offs.
    """
    import calendar
    from datetime import datetime, timedelta

    period_label  = f"{calendar.month_name[month]} {year}"
    base_date     = datetime(year, month, 1)

    events = [
        {
            "timestamp":   (base_date + timedelta(days=1)).isoformat(),
            "event":       "Month-end close initiated",
            "actor":       user_id,
            "system":      "Month-End Assistant",
            "status":      "completed",
        },
        {
            "timestamp":   (base_date + timedelta(days=1, hours=1)).isoformat(),
            "event":       "Financial data fetched from ERP",
            "actor":       "system",
            "system":      "AgentCore Runtime",
            "status":      "completed",
        },
        {
            "timestamp":   (base_date + timedelta(days=1, hours=2)).isoformat(),
            "event":       "Deep research analysis completed (LangGraph)",
            "actor":       "DeepResearchAgent",
            "system":      "LangGraph",
            "status":      "completed",
        },
        {
            "timestamp":   (base_date + timedelta(days=1, hours=3)).isoformat(),
            "event":       "Anomaly scan completed – findings reviewed",
            "actor":       "AnomalyAgent",
            "system":      "LangGraph Supervisor",
            "status":      "completed",
        },
        {
            "timestamp":   (base_date + timedelta(days=1, hours=4)).isoformat(),
            "event":       "Accrual journal entries posted",
            "actor":       user_id,
            "system":      "ERP",
            "status":      "completed",
        },
        {
            "timestamp":   (base_date + timedelta(days=1, hours=5)).isoformat(),
            "event":       "HITL approval notification sent (Teams + Slack)",
            "actor":       "HITLManager",
            "system":      "Teams / Slack",
            "status":      "completed",
        },
        {
            "timestamp":   (base_date + timedelta(days=1, hours=6)).isoformat(),
            "event":       "Finance controller approved report",
            "actor":       "controller@acme.com",
            "system":      "HITL Webhook",
            "status":      "approved",
        },
        {
            "timestamp":   (base_date + timedelta(days=1, hours=6, minutes=5)).isoformat(),
            "event":       "Report published and distributed",
            "actor":       "system",
            "system":      "Month-End Assistant",
            "status":      "completed",
        },
    ]

    return {
        "audit_trail_id":  f"AUD-{year}{month:02d}",
        "period":          period_label,
        "initiated_by":    user_id,
        "close_type":      "standard",
        "total_events":    len(events),
        "events":          events,
        "sign_off": {
            "preparer":   user_id,
            "reviewer":   "controller@acme.com",
            "approver":   "cfo@acme.com",
            "date":       (base_date + timedelta(days=1, hours=6)).isoformat(),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Financial Risk Assessment
# ─────────────────────────────────────────────────────────────────────────────

@tool
def assess_financial_risk(year: int, month: int) -> Dict[str, Any]:
    """
    Assess and score the top financial risks for the period.

    Args:
        year:  Accounting period year.
        month: Accounting period month.

    Returns:
        Risk register dict with individual risks scored by likelihood × impact.
    """
    rng = random.Random(year * 100 + month + 13)

    risks = [
        {
            "risk_id":     "RSK-001",
            "category":    "Credit",
            "description": "Concentration risk – top 3 customers represent 68% of AR",
            "likelihood":  rng.choice([2, 3]),
            "impact":      rng.choice([3, 4]),
            "mitigation":  "Implement credit limits; obtain trade credit insurance.",
        },
        {
            "risk_id":     "RSK-002",
            "category":    "Liquidity",
            "description": "Current ratio below 1.2× – approaching covenant trigger",
            "likelihood":  rng.choice([2, 3]),
            "impact":      4,
            "mitigation":  "Draw on revolving credit facility; accelerate AR collections.",
        },
        {
            "risk_id":     "RSK-003",
            "category":    "Operational",
            "description": "Manual journal entry controls – ERP workflow bypassed for 12% of JEs",
            "likelihood":  3,
            "impact":      rng.choice([3, 4]),
            "mitigation":  "Enforce ERP approval workflow; exception report to CFO weekly.",
        },
        {
            "risk_id":     "RSK-004",
            "category":    "Compliance",
            "description": "IFRS 16 lease liability not updated for 2 new leases this period",
            "likelihood":  2,
            "impact":      3,
            "mitigation":  "Update lease register before period close; restate if material.",
        },
        {
            "risk_id":     "RSK-005",
            "category":    "FX",
            "description": "USD/EUR exposure of $1.8M not hedged – rate moved 3.2% this period",
            "likelihood":  3,
            "impact":      3,
            "mitigation":  "Implement forward contract programme; review hedging policy.",
        },
    ]

    for r in risks:
        r["risk_score"]   = r["likelihood"] * r["impact"]
        r["risk_rating"]  = (
            "Critical" if r["risk_score"] >= 12
            else "High" if r["risk_score"] >= 8
            else "Medium" if r["risk_score"] >= 4
            else "Low"
        )

    overall = sum(r["risk_score"] for r in risks) / len(risks)
    return {
        "period":            f"{year}-{month:02d}",
        "overall_risk_score": round(overall, 1),
        "overall_rating":     "High" if overall >= 9 else "Medium" if overall >= 5 else "Low",
        "risks":              sorted(risks, key=lambda r: r["risk_score"], reverse=True),
        "top_risk":           risks[0]["description"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cash-Flow Forecast
# ─────────────────────────────────────────────────────────────────────────────

@tool
def forecast_cash_flow(year: int, month: int, horizon_months: int = 3) -> List[Dict[str, Any]]:
    """
    Generate a rolling cash-flow forecast for the next N months.

    Args:
        year:            Base period year.
        month:           Base period month (actuals through this month).
        horizon_months:  Number of forecast months (default 3).

    Returns:
        List of monthly forecast dicts with inflows, outflows, and net cash.
    """
    rng  = random.Random(year * 100 + month + 17)
    base = rng.uniform(300_000, 600_000)   # starting cash balance

    forecasts = []
    for i in range(1, horizon_months + 1):
        m = month + i
        y = year + (m - 1) // 12
        m = ((m - 1) % 12) + 1

        inflows  = round(rng.uniform(3_600_000, 5_200_000), 2)
        outflows = round(rng.uniform(3_200_000, 4_800_000), 2)
        net      = round(inflows - outflows, 2)
        base    += net

        forecasts.append({
            "period":         f"{y}-{m:02d}",
            "inflows":         inflows,
            "outflows":        outflows,
            "net_cash_flow":   net,
            "closing_balance": round(base, 2),
            "is_negative":     base < 0,
            "confidence":      round(max(0.5, 0.95 - i * 0.12), 2),
        })
    return forecasts


# ─────────────────────────────────────────────────────────────────────────────
# Account Reconciliation
# ─────────────────────────────────────────────────────────────────────────────

@tool
def reconcile_account(account_name: str, year: int, month: int) -> Dict[str, Any]:
    """
    Perform a mock reconciliation for a balance-sheet account.

    Args:
        account_name: e.g. 'bank', 'accounts_receivable', 'accounts_payable'.
        year:         Period year.
        month:        Period month.

    Returns:
        Reconciliation dict with GL balance, sub-ledger balance, and exceptions.
    """
    rng = random.Random(year * 100 + month + hash(account_name) % 100)

    gl_balance  = round(rng.uniform(500_000, 5_000_000), 2)
    difference  = round(rng.uniform(-15_000, 15_000), 2)
    sub_balance = gl_balance + difference

    exceptions = []
    if abs(difference) > 5_000:
        exceptions.append({
            "exception_id":  "EXC-001",
            "type":          "unreconciled item",
            "description":   f"Timing difference of ${abs(difference):,.2f} not yet cleared",
            "amount":         difference,
            "action":         "Investigate and clear before sign-off.",
        })

    return {
        "account":         account_name.replace("_", " ").title(),
        "period":          f"{year}-{month:02d}",
        "gl_balance":      gl_balance,
        "sub_ledger_balance": sub_balance,
        "difference":      difference,
        "is_reconciled":   abs(difference) <= 1.00,
        "exceptions":      exceptions,
        "status":          "✅ Clear" if abs(difference) <= 1.00 else "⚠ Exceptions found",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Peer Benchmarking
# ─────────────────────────────────────────────────────────────────────────────

@tool
def run_peer_benchmark(year: int, month: int, sector: str = "technology") -> Dict[str, Any]:
    """
    Compare our KPIs to a synthetic peer group for the sector.

    Args:
        year:    Period year.
        month:   Period month.
        sector:  Industry sector for peer selection.

    Returns:
        Peer benchmark comparison with percentile rankings.
    """
    rng = random.Random(year * 100 + month + 23)

    our_metrics = {
        "gross_margin_pct":           round(rng.uniform(48, 68), 1),
        "ebitda_margin_pct":          round(rng.uniform(12, 28), 1),
        "net_margin_pct":             round(rng.uniform(6, 18), 1),
        "days_sales_outstanding":     round(rng.uniform(30, 65), 1),
        "days_payable_outstanding":   round(rng.uniform(25, 55), 1),
        "current_ratio":              round(rng.uniform(1.0, 2.5), 2),
        "revenue_growth_pct":         round(rng.uniform(-5, 25), 1),
    }

    peer_medians = {
        "technology":    {"gross_margin_pct": 65, "ebitda_margin_pct": 22, "net_margin_pct": 14,
                         "days_sales_outstanding": 52, "days_payable_outstanding": 38,
                         "current_ratio": 1.8, "revenue_growth_pct": 12},
        "retail":        {"gross_margin_pct": 35, "ebitda_margin_pct": 8,  "net_margin_pct": 4,
                         "days_sales_outstanding": 14, "days_payable_outstanding": 30,
                         "current_ratio": 1.3, "revenue_growth_pct": 5},
        "manufacturing": {"gross_margin_pct": 32, "ebitda_margin_pct": 13, "net_margin_pct": 7,
                         "days_sales_outstanding": 48, "days_payable_outstanding": 45,
                         "current_ratio": 1.5, "revenue_growth_pct": 6},
    }

    medians   = peer_medians.get(sector.lower(), peer_medians["technology"])
    rankings  = {}
    for metric, our_val in our_metrics.items():
        peer_val = medians.get(metric, our_val)
        # Higher is better for margins/growth; lower is better for DSO/DPO
        better_when_lower = metric in ("days_sales_outstanding", "days_payable_outstanding")
        outperform = (our_val < peer_val) if better_when_lower else (our_val > peer_val)
        rankings[metric] = {
            "our_value":    our_val,
            "peer_median":  peer_val,
            "delta":        round(our_val - peer_val, 1),
            "outperforms":  outperform,
            "percentile":   rng.randint(30, 85) if outperform else rng.randint(15, 50),
        }

    return {
        "sector":          sector,
        "period":          f"{year}-{month:02d}",
        "peer_group_size": rng.randint(12, 30),
        "rankings":        rankings,
        "outperforming":   sum(1 for r in rankings.values() if r["outperforms"]),
        "underperforming": sum(1 for r in rankings.values() if not r["outperforms"]),
    }


# ── Extended tool registry ────────────────────────────────────────────────────

REPORTING_TOOLS = [
    detect_anomalies,
    calculate_accruals,
    generate_audit_trail,
    assess_financial_risk,
    forecast_cash_flow,
    reconcile_account,
    run_peer_benchmark,
]
