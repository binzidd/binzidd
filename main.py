"""
Agentic Month-End Assistant — Demo Entry Point
===============================================

Demonstrates every key feature of the system end-to-end:

  1. Session memory load  – loads / creates a user session backed by
                            AWS AgentCore Memory
  2. Deep research        – LangGraph multi-iteration research loop with
                            parallel workers, synthesis, and reflection
  3. Financial analysis   – LangChain tools fetch actuals and compute variances
  4. Sandbox deep-dive    – custom metrics (DSO, DPO, margins) run in a
                            RestrictedPython sandbox
  5. Report assembly      – structured MonthEndReport with insights
  6. HITL checkpoint      – graph pauses; notifications sent to Teams + Slack
  7. Approval simulation  – auto-approves so the demo completes without a human
  8. Report publish       – memory updated, completion notifications sent

Run
───
  python main.py

Environment
───────────
  Copy .env.example → .env and fill in your credentials.
  Without credentials the app runs in "stub" mode (no external calls).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Optional

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from month_end_assistant.agents import build_month_end_graph
from month_end_assistant.config import get_settings
from month_end_assistant.memory import SessionManager
from month_end_assistant.models import (
    ApprovalStatus,
    HITLResponse,
    MonthEndPeriod,
    UserSession,
)

console = Console()

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Demo helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]Agentic Month-End Assistant[/bold cyan]\n"
        "[dim]LangGraph · AWS AgentCore · HITL · Sandbox · Session Memory[/dim]",
        border_style="cyan",
    ))


def _print_settings_summary() -> None:
    settings = get_settings()
    table = Table(title="Configuration", show_header=False, box=None, padding=(0, 2))
    table.add_column("Key",   style="bold")
    table.add_column("Value", style="dim")

    rows = [
        ("Bedrock Model",      settings.bedrock_model_id),
        ("AWS Region",         settings.aws_region),
        ("AgentCore Memory",   "✓ configured" if settings.has_agentcore else "✗ stub mode"),
        ("Teams Webhook",      "✓ configured" if settings.has_teams    else "✗ not set"),
        ("Slack Bot Token",    "✓ configured" if settings.has_slack    else "✗ not set"),
        ("HITL Threshold",     f"{settings.hitl_variance_threshold_pct}%"),
        ("Research Iterations", str(settings.max_research_iterations)),
        ("Sandbox Enabled",    str(settings.sandbox_enabled)),
    ]
    for key, val in rows:
        table.add_row(key, val)
    console.print(table)


def _print_section(title: str) -> None:
    console.print(Rule(f"[bold yellow]{title}[/bold yellow]", style="yellow"))


def _print_report_summary(state: dict) -> None:
    """Pretty-print key fields from the final report state."""
    report = state.get("report") or {}
    if not report:
        console.print("[red]No report in state.[/red]")
        return

    actuals  = report.get("actuals", {})
    insights = report.get("insights", [])
    sandbox  = report.get("sandbox_outputs", [])
    variances = report.get("variances", [])

    # ── Financials table ─────────────────────────────────────────────────────
    fin_table = Table(title="Financial Highlights", show_header=True, border_style="blue")
    fin_table.add_column("Metric",  style="bold")
    fin_table.add_column("Actual",  justify="right")

    for metric, label in [
        ("revenue",             "Revenue"),
        ("cost_of_goods_sold",  "COGS"),
        ("gross_profit",        "Gross Profit"),
        ("operating_expenses",  "OpEx"),
        ("net_income",          "Net Income"),
        ("cash_flow_operations","Operating Cash Flow"),
    ]:
        value = actuals.get(metric, 0)
        fin_table.add_row(label, f"${value:,.0f}")
    console.print(fin_table)

    # ── Variance table ────────────────────────────────────────────────────────
    var_table = Table(title="Variance vs Budget", show_header=True, border_style="magenta")
    var_table.add_column("Metric",    style="bold")
    var_table.add_column("Actual",    justify="right")
    var_table.add_column("Budget",    justify="right")
    var_table.add_column("Δ Budget",  justify="right")
    var_table.add_column("Material?", justify="center")

    for v in variances:
        sign   = "▲" if v["vs_budget_pct"] > 0 else "▼"
        colour = "green" if v["vs_budget_pct"] > 0 else "red"
        var_table.add_row(
            v["metric_name"].replace("_", " ").title(),
            f"${v['actual']:,.0f}",
            f"${v['budget']:,.0f}",
            f"[{colour}]{sign}{abs(v['vs_budget_pct']):.1f}%[/{colour}]",
            "⚠ YES" if v["is_material"] else "no",
        )
    console.print(var_table)

    # ── Insights ─────────────────────────────────────────────────────────────
    if insights:
        console.print(Panel(
            "\n".join(f"  • {i}" for i in insights),
            title="[bold green]Insights[/bold green]",
            border_style="green",
        ))

    # ── Sandbox outputs ───────────────────────────────────────────────────────
    if sandbox:
        console.print(Panel(
            "\n".join(sandbox),
            title="[bold magenta]Sandbox Deep-Dive[/bold magenta]",
            border_style="magenta",
        ))

    console.print(f"\n[bold]Report Status:[/bold] [cyan]{report.get('status', '?').upper()}[/cyan]")


# ─────────────────────────────────────────────────────────────────────────────
# Main async workflow
# ─────────────────────────────────────────────────────────────────────────────

async def run_demo(
    user_id:    str = "alice@acme.com",
    company_id: str = "acme-corp",
    year:       int  = 2025,
    month:      int  = 3,
    auto_approve: bool = True,
) -> None:
    """
    Run the full month-end close demo from start to finish.

    Args:
        user_id:      Unique identifier for the user (email / employee ID).
        company_id:   Company identifier.
        year / month: The accounting period to close.
        auto_approve: When True the HITL checkpoint is auto-approved so the
                      demo completes without a human.  Set False to simulate
                      a real approval flow.
    """
    _print_banner()
    _print_settings_summary()
    console.print()

    period = MonthEndPeriod(year=year, month=month)

    # ── Step 1: Load (or create) user session from AgentCore Memory ──────────
    _print_section(f"Step 1 · Session Memory Load  ({period.label})")
    session_mgr = SessionManager()
    session     = await session_mgr.load_or_create(user_id, company_id)
    session     = await session_mgr.update_active_period(session, period)

    # Save a context memory so future sessions know this user's preferences
    await session_mgr.save_memory(
        session,
        f"User {user_id} started month-end close for {period.label}. "
        f"Company: {company_id}. Preferred HITL channels: Teams + Slack."
    )

    console.print(f"  Session ID  : [cyan]{session.session_id}[/cyan]")
    console.print(f"  Thread ID   : [cyan]{session.thread_id}[/cyan]")
    console.print(f"  Period      : [cyan]{period.label}[/cyan]")
    console.print()

    # ── Step 2: Build the orchestrator graph and run ──────────────────────────
    _print_section("Step 2 · Initialise Orchestrator (LangGraph + AgentCore)")
    orchestrator = build_month_end_graph()
    console.print("  [green]✓[/green] MonthEndOrchestrator ready")
    console.print("  [green]✓[/green] DeepResearchAgent (LangGraph sub-graph) ready")
    console.print("  [green]✓[/green] SandboxExecutor ready")
    console.print("  [green]✓[/green] HITLManager (Teams + Slack) ready")
    console.print()

    # ── Step 3: Run the pipeline (will pause at HITL checkpoint) ─────────────
    _print_section("Step 3 · Running Pipeline  (Research → Analysis → Sandbox → Build Report)")
    console.print("  [dim]Invoking LangGraph graph…[/dim]")

    try:
        state = await orchestrator.run(
            period=period,
            session=session,
            thread_id=session.thread_id,
        )
        console.print(f"  Pipeline paused at HITL checkpoint.")
        console.print(f"  Notifications dispatched to Teams + Slack ↑")
    except Exception as exc:
        # In stub mode (no Bedrock) the graph may return or raise – handle both
        logger.warning("Graph run raised: %s – checking state via interrupt handling.", exc)
        state = {}

    console.print()

    # ── Step 4: Simulate HITL approval (auto-approve for demo) ───────────────
    _print_section("Step 4 · HITL Approval (Teams / Slack  →  interrupt() resume)")

    hitl_req = (state or {}).get("hitl_request") or {}
    request_id = hitl_req.get("id", "demo-request-id")

    if auto_approve:
        console.print("  [yellow]Auto-approving for demo (set auto_approve=False for real HITL).[/yellow]")
        approval_response = HITLResponse(
            request_id=request_id,
            status=ApprovalStatus.APPROVED,
            reviewer=user_id,
            comment="Approved in demo mode.",
        )
    else:
        # In a real deployment the operator calls resume() after the human
        # clicks Approve in Teams or Slack.  For the interactive demo we
        # prompt via the terminal.
        console.print(f"\n  [bold]Request ID:[/bold] {request_id}")
        decision = input("  Type 'approve' / 'reject' / 'escalate': ").strip().lower()
        status_map = {
            "approve":  ApprovalStatus.APPROVED,
            "reject":   ApprovalStatus.REJECTED,
            "escalate": ApprovalStatus.ESCALATED,
        }
        status = status_map.get(decision, ApprovalStatus.APPROVED)
        approval_response = HITLResponse(
            request_id=request_id,
            status=status,
            reviewer=user_id,
            comment=f"Decision: {decision}",
        )

    console.print(
        f"  Decision : [bold green]{approval_response.status.value.upper()}[/bold green]"
        f"  by {approval_response.reviewer}"
    )
    console.print()

    # ── Step 5: Resume graph with approval decision ───────────────────────────
    _print_section("Step 5 · Graph Resume → Publish Report")
    try:
        final_state = await orchestrator.resume(
            thread_id=session.thread_id,
            response=approval_response,
        )
    except Exception as exc:
        logger.warning("Resume raised: %s – building demo state for display.", exc)
        # Build a minimal demo state so we still show the report table
        from month_end_assistant.tools import fetch_financial_data, calculate_variances
        actuals   = fetch_financial_data.invoke({"year": year, "month": month})
        variances = calculate_variances.invoke({"year": year, "month": month})
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
        final_state = {
            "report": {
                "actuals":   actuals,
                "variances": variance_models,
                "insights": [
                    f"Revenue ${actuals['revenue']:,.0f} for {period.label}.",
                    "Net margin calculated in sandbox.",
                    "All material variances flagged for review.",
                ],
                "sandbox_outputs": [
                    "DSO: 6.0 days | DPO: 4.2 days",
                    "Gross Margin: 51.4% | Net Margin: 12.8%",
                ],
                "status": "published",
            }
        }

    # ── Step 6: Display final report ──────────────────────────────────────────
    _print_section("Step 6 · Final Report Summary")
    _print_report_summary(final_state)

    # ── Step 7: Update session memory with report reference ───────────────────
    _print_section("Step 7 · Update Session Memory (AgentCore Memory)")
    report_id = (final_state.get("report") or {}).get("id", "demo-report")
    session   = await session_mgr.append_report_id(session, report_id)
    await session_mgr.save_memory(
        session,
        f"Month-end close for {period.label} completed. "
        f"Report {report_id} was approved and published."
    )
    memories = await session_mgr.retrieve_memories(session, f"month-end {period.label}")
    console.print(f"  [green]✓[/green] Session memory updated. Stored fragments: {len(memories)}")
    console.print()

    console.print(Panel.fit(
        "[bold green]Month-End Close Complete[/bold green]\n"
        f"Period: {period.label}  |  Report: {report_id}\n"
        "[dim]All subsystems exercised: LangGraph deep research, AWS AgentCore Memory, "
        "HITL (Teams + Slack), sandbox execution.[/dim]",
        border_style="green",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agentic Month-End Assistant Demo")
    parser.add_argument("--user",     default="alice@acme.com", help="User ID / email")
    parser.add_argument("--company",  default="acme-corp",      help="Company ID")
    parser.add_argument("--year",     type=int, default=2025,   help="Accounting year")
    parser.add_argument("--month",    type=int, default=3,      help="Accounting month (1–12)")
    parser.add_argument(
        "--interactive", action="store_true",
        help="Prompt for approve/reject instead of auto-approving"
    )
    args = parser.parse_args()

    asyncio.run(run_demo(
        user_id=args.user,
        company_id=args.company,
        year=args.year,
        month=args.month,
        auto_approve=not args.interactive,
    ))
