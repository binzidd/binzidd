"""
Agentic Month-End Assistant — Demo Entry Point
===============================================

Showcases ALL features across LangChain, LangGraph, and AWS AgentCore:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LangChain Features                                                     │
  │  ─────────────────                                                      │
  │  • LCEL pipe |, RunnableParallel, RunnableBranch, RunnablePassthrough  │
  │  • with_structured_output  (Pydantic schema enforcement)               │
  │  • with_retry + with_fallbacks  (resilience)                           │
  │  • PydanticOutputParser, JsonOutputParser, StrOutputParser             │
  │  • FewShotChatMessagePromptTemplate  (in-context learning)             │
  │  • ChatPromptTemplate with MessagesPlaceholder  (chat history)         │
  │  • InMemoryVectorStore + RAG chain  (accounting-standards Q&A)         │
  │  • create_retrieval_chain + create_stuff_documents_chain               │
  │  • Custom Callbacks: RichConsoleCallback, TokenStreamingCallback,      │
  │    TracingCallback                                                      │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  LangGraph Features                                                     │
  │  ──────────────────                                                     │
  │  • StateGraph with typed AgentState TypedDict                          │
  │  • Send API for parallel fan-out (map-reduce)                          │
  │  • operator.add reducer for fan-in accumulation                        │
  │  • interrupt() for HITL checkpoint + Command(resume=…) to continue     │
  │  • create_react_agent for each supervisor worker                       │
  │  • Command(goto=…) for supervisor routing                              │
  │  • MemorySaver checkpointer (thread-level state persistence)           │
  │  • astream_events for real-time event streaming                        │
  │  • Multi-level nested graphs (orchestrator → research sub-graph)       │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  AWS AgentCore Features                                                 │
  │  ───────────────────────                                                │
  │  • AgentCore Memory: IngestConversations, RetrieveMemories             │
  │  • AgentCore Runtime: InvokeAgent with EventStream                     │
  │  • Session Memory load / save per user                                 │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  Scenarios (8 deep-dive agents)                                        │
  │  ──────────────────────────────                                        │
  │  1  Revenue Recognition   – structured output + RAG                   │
  │  2  Anomaly Detection      – few-shot + JSON parser                    │
  │  3  Account Reconciliation – LangGraph map-reduce                      │
  │  4  Accrual Calculation    – ReAct agent + sandbox                     │
  │  5  Peer Benchmarking      – RAG + parallel LCEL                       │
  │  6  Cash-Flow Forecast     – LCEL branch + sandbox DCF                 │
  │  7  Audit Trail            – sequential chain + callbacks               │
  │  8  Risk Assessment        – LCEL branching + sandbox                  │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  HITL (Human-in-the-Loop)                                              │
  │  ─────────────────────────                                             │
  │  • Teams Adaptive Cards with Approve/Reject/Escalate buttons           │
  │  • Slack Block Kit with interactive action buttons                     │
  │  • HITLManager: fan-out, timeout expiry, response routing              │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  Sandbox                                                               │
  │  ───────                                                               │
  │  • RestrictedPython (AST transform + safe builtins)                   │
  │  • DSO / DPO / margins / DCF sensitivity analysis                     │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  LangChain Deep Agents (full checklist)                                │
  │  ──────────────────────────────────────                                │
  │  ✅ Complex multi-step planning + decomposition                         │
  │  ✅ Context mgmt via ConversationSummaryBufferMemory + summarise_file   │
  │  ✅ Swappable backends: InMemory / LocalDisk / Sandbox / Durable(S3)   │
  │  ✅ execute tool → shell commands inside SandboxBackend tempdir         │
  │  ✅ Delegate to isolated subagents (create_react_agent per sub-task)    │
  │  ✅ Persist memory across threads (MemorySaver + AgentCore Memory)      │
  │  ✅ Declarative FilePermissions (allowed/denied paths, ext, size)       │
  └─────────────────────────────────────────────────────────────────────────┘

Run
───
  python main.py                         # full demo (all features)
  python main.py --deep-agent            # Deep Agents checklist demo
  python main.py --scenario anomaly      # single scenario
  python main.py --supervisor            # multi-agent supervisor demo
  python main.py --rag                   # RAG chain demo
  python main.py --lcel                  # LCEL chains showcase
  python main.py --server                # start FastAPI server (OpenWebUI)
  python main.py --interactive           # real HITL (prompts for approval)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from month_end_assistant.agents import build_month_end_graph, DeepAgent
from month_end_assistant.agents.scenarios import SCENARIO_REGISTRY, run_scenario
from month_end_assistant.filesystem import (
    FilePermissions, InMemoryBackend, SandboxBackend, LocalDiskBackend,
    DurableBackend, create_backend, build_filesystem_tools,
)
from month_end_assistant.agents.supervisor import MonthEndSupervisor
from month_end_assistant.callbacks import RichConsoleCallback, TracingCallback
from month_end_assistant.chains import (
    AccountingRAGChain,
    LCELChainFactory,
    build_anomaly_chain,
    build_parallel_analysis_chain,
    build_variance_chain,
)
from month_end_assistant.config import get_settings
from month_end_assistant.memory import SessionManager
from month_end_assistant.models import (
    ApprovalStatus,
    HITLResponse,
    MonthEndPeriod,
)
from month_end_assistant.tools import fetch_financial_data, calculate_variances

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]Agentic Month-End Assistant[/bold cyan]\n"
        "[dim]LangGraph · LangChain LCEL · AWS AgentCore · HITL · RAG · Sandbox · OpenWebUI[/dim]",
        border_style="cyan",
    ))


def _section(title: str) -> None:
    console.print(Rule(f"[bold yellow]{title}[/bold yellow]", style="yellow"))


def _settings_table() -> None:
    s = get_settings()
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("K", style="bold")
    t.add_column("V", style="dim")
    for key, val in [
        ("Bedrock Model",     s.bedrock_model_id),
        ("AgentCore Memory",  "✓" if s.has_agentcore else "✗ stub"),
        ("Teams",             "✓" if s.has_teams    else "✗ not set"),
        ("Slack",             "✓" if s.has_slack    else "✗ not set"),
        ("Sandbox",           str(s.sandbox_enabled)),
        ("Research Iters",    str(s.max_research_iterations)),
    ]:
        t.add_row(key, val)
    console.print(t)


def _print_json(data: dict, title: str = "") -> None:
    if title:
        console.print(f"[bold]{title}[/bold]")
    console.print_json(json.dumps(data, default=str, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1 – LCEL chains showcase
# ─────────────────────────────────────────────────────────────────────────────

async def demo_lcel_chains(period: MonthEndPeriod) -> None:
    """Showcase LCEL: parallel, branching, structured output, few-shot."""
    _section("LCEL Chains Showcase")

    from month_end_assistant.agents.base import BaseAgent
    llm     = BaseAgent()._build_llm()
    factory = LCELChainFactory(llm)

    actuals   = fetch_financial_data.invoke({"year": period.year, "month": period.month})
    variances = calculate_variances.invoke({"year": period.year, "month": period.month})
    rev_var   = next((v for v in variances if v["metric"] == "revenue"), variances[0])

    # 1. Structured output chain
    console.print("  [yellow]1.[/yellow] Variance chain → with_structured_output (Pydantic)")
    try:
        result = factory.variance_chain.invoke({
            "metric":        rev_var["metric"],
            "actual":        rev_var["actual"],
            "budget":        rev_var["budget"],
            "vs_budget_pct": rev_var["vs_budget_pct"],
            "prior_period":  rev_var["prior_period"],
        })
        console.print(f"     → {result}")
    except Exception as e:
        console.print(f"     → [dim](stub mode: {e})[/dim]")

    # 2. Branching chain
    console.print("  [yellow]2.[/yellow] Branching chain → RunnableBranch (routes by severity)")
    for test_pct in [-18.0, -7.0, 2.0]:
        try:
            commentary = factory.branching_chain.invoke({
                "metric": "revenue", "vs_budget_pct": test_pct
            })
            label = "Critical" if abs(test_pct) >= 15 else "Material" if abs(test_pct) >= 5 else "Immaterial"
            console.print(f"     {label:12s} ({test_pct:+.0f}%): {str(commentary)[:80]}…")
        except Exception as e:
            console.print(f"     ({test_pct:+.0f}%): [dim]stub: {e}[/dim]")

    # 3. Parallel chain
    console.print("  [yellow]3.[/yellow] Parallel chain → RunnableParallel (2 chains simultaneously)")
    fin_summary = f"Revenue ${actuals['revenue']:,.0f} | Net Income ${actuals['net_income']:,.0f}"
    try:
        result = factory.parallel_analysis_chain.invoke({
            "financial_summary": fin_summary, "sector": "technology",
        })
        console.print(f"     → Keys: {list(result.keys())}")
    except Exception as e:
        console.print(f"     → [dim]stub: {e}[/dim]")

    # 4. Few-shot anomaly chain
    console.print("  [yellow]4.[/yellow] Anomaly chain → FewShotChatMessagePromptTemplate + JsonOutputParser")
    try:
        anomalies = factory.anomaly_chain.invoke({
            "financial_data": f"Revenue ${actuals['revenue']:,.0f} | AR ${actuals['accounts_receivable']:,.0f}"
        })
        console.print(f"     → {len(anomalies)} anomaly/ies detected")
    except Exception as e:
        console.print(f"     → [dim]stub: {e}[/dim]")

    console.print("  [green]✓ LCEL chains showcase complete[/green]\n")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2 – RAG chain
# ─────────────────────────────────────────────────────────────────────────────

async def demo_rag_chain() -> None:
    """Showcase RAG: InMemoryVectorStore + create_retrieval_chain."""
    _section("RAG Chain — Accounting Standards Q&A")

    from month_end_assistant.agents.base import BaseAgent
    llm = BaseAgent()._build_llm()
    rag = AccountingRAGChain(llm)

    questions = [
        "What are the five steps of IFRS 15 revenue recognition?",
        "What journal entries are needed for IFRS 16 month-end close?",
        "When must a provision be recognised under IAS 37?",
    ]

    for i, q in enumerate(questions, 1):
        console.print(f"  [yellow]Q{i}:[/yellow] {q}")
        try:
            result = await rag.aask(q)
            answer = result["answer"]
            sources = [d.metadata.get("standard", "?") for d in result["source_documents"]]
            console.print(f"  [green]A:[/green] {answer[:200]}…")
            console.print(f"  [dim]Sources: {', '.join(set(sources))}[/dim]\n")
        except Exception as e:
            console.print(f"  [dim]stub: {e}[/dim]\n")

    console.print("  [green]✓ RAG chain demo complete[/green]\n")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3 – All 8 scenarios
# ─────────────────────────────────────────────────────────────────────────────

async def demo_scenarios(period: MonthEndPeriod, scenario_name: Optional[str] = None) -> None:
    """Run one or all 8 deep-dive scenarios."""
    _section("Deep-Dive Scenarios")

    if scenario_name:
        names = [scenario_name]
    else:
        names = list(SCENARIO_REGISTRY.keys())

    for name in names:
        console.print(f"  [yellow]▶[/yellow] {name.replace('_', ' ').title()}")
        try:
            result = await run_scenario(name, period)
            # Print a short summary of the result
            for k, v in result.items():
                if k in ("scenario", "period"):
                    continue
                preview = json.dumps(v, default=str)[:120] if not isinstance(v, str) else v[:120]
                console.print(f"    [dim]{k}:[/dim] {preview}…")
            console.print(f"  [green]✓ {name} done[/green]\n")
        except Exception as e:
            console.print(f"  [red]✗ {name} failed: {e}[/red]\n")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4 – LangGraph supervisor
# ─────────────────────────────────────────────────────────────────────────────

async def demo_supervisor(period: MonthEndPeriod) -> None:
    """Run the multi-agent supervisor with 5 specialist workers."""
    _section("LangGraph Multi-Agent Supervisor")

    supervisor = MonthEndSupervisor()

    console.print("  Sending task: 'Find anomalies, reconcile bank, assess credit risk'")
    try:
        result = await supervisor.run(
            query=(
                f"For {period.label}: "
                "1) Detect any financial anomalies. "
                "2) Reconcile the bank account. "
                "3) Assess financial risks and score them."
            ),
            period_label=period.label,
        )
        messages = result.get("messages", [])
        console.print(f"  [green]✓ Supervisor completed in {len(messages)} message turns[/green]")
        for msg in messages[-3:]:
            if hasattr(msg, "content") and msg.content:
                console.print(f"  [dim]{msg.content[:200]}…[/dim]")
    except Exception as e:
        console.print(f"  [dim]stub mode: {e}[/dim]")

    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5 – Full orchestrator pipeline with HITL
# ─────────────────────────────────────────────────────────────────────────────

async def demo_deep_agent(period: MonthEndPeriod) -> None:
    """
    Demonstrate every LangChain Deep Agents checklist item.

    Each backend is shown in turn so the 'swap filesystem backends' feature
    is visible in the output.
    """
    _section("LangChain Deep Agents — Full Checklist Demo")

    # ── 1. Declarative permission rules ───────────────────────────────────────
    console.print("  [yellow]✅ 1.[/yellow] Declarative FilePermissions")
    perms = FilePermissions(
        allowed_paths=["/reports", "/data", "/subagent_results"],
        denied_paths=["/secrets", "/etc", "/usr"],
        allowed_extensions=[".py", ".json", ".csv", ".txt", ".md"],
        max_file_size_mb=5,
        read_only=False,
    )
    console.print(f"     allowed_extensions: {perms.allowed_extensions}")
    console.print(f"     denied_paths: {perms.denied_paths}")
    console.print(f"     max_file_size: 5 MB  |  read_only: {perms.read_only}")

    # ── 2. Swappable filesystem backends ─────────────────────────────────────
    console.print("\n  [yellow]✅ 2.[/yellow] Swappable Filesystem Backends")
    for kind, kwargs in [
        ("memory",  {}),
        ("local",   {"root_dir": "/tmp/deep-agent-local"}),
        ("sandbox", {}),
    ]:
        b = create_backend(kind, permissions=perms, **kwargs)
        b.write("reports/test.json", json.dumps({"period": period.label, "backend": kind}))
        files = b.list()
        console.print(f"     [{kind:8s}] wrote reports/test.json → files: {files}")
        if hasattr(b, "close"):
            b.close()

    # ── 3. execute tool in sandbox ────────────────────────────────────────────
    console.print("\n  [yellow]✅ 3.[/yellow] execute tool (shell commands in SandboxBackend)")
    from month_end_assistant.filesystem.tools import execute, set_active_backend
    with SandboxBackend(permissions=perms) as sandbox:
        set_active_backend(sandbox)
        result = execute.invoke({"command": "python -c \"print('DSO =', 840000/4200000*30)\" "})
        console.print(f"     $ python -c '...' → {result.strip()}")
        result2 = execute.invoke({"command": "ls -la"})
        console.print(f"     $ ls -la → {result2.splitlines()[0] if result2 else '(empty)'}")
        # Test blocked command
        blocked = execute.invoke({"command": "curl http://example.com"})
        console.print(f"     $ curl (blocked) → {blocked}")

    # ── 4. Context summarisation ──────────────────────────────────────────────
    console.print("\n  [yellow]✅ 4.[/yellow] ConversationSummaryBufferMemory (large context mgmt)")
    from langchain.memory import ConversationSummaryBufferMemory
    from month_end_assistant.agents.base import BaseAgent
    llm = BaseAgent()._build_llm()
    mem = ConversationSummaryBufferMemory(llm=llm, max_token_limit=200, return_messages=True)
    # Simulate a long conversation that exceeds the token limit
    for i in range(5):
        mem.save_context(
            {"input":  f"What was the revenue variance in month {i+1}?"},
            {"output": f"Revenue variance in month {i+1} was {(i+1)*2.3:.1f}% above budget due to new contracts."},
        )
    try:
        vars_loaded = mem.load_memory_variables({})
        history = vars_loaded.get("chat_history", [])
        console.print(f"     History compressed: {len(history)} messages (auto-summarised beyond 200 tokens)")
    except Exception as e:
        console.print(f"     [dim]Memory (stub): {e}[/dim]")

    # ── 5. Full DeepAgent run (all features together) ─────────────────────────
    console.print("\n  [yellow]✅ 5.[/yellow] DeepAgent full run (sandbox backend + subagent delegation)")
    agent = DeepAgent(backend_kind="memory", max_iterations=1)
    try:
        state = await agent.run(
            task=(
                f"For {period.label}: fetch financial data, detect anomalies, "
                "write a risk summary to reports/risk_summary.txt, "
                "and return the top 3 risks."
            ),
            period=period,
        )
        console.print(f"     Subagent results: {len(state.get('subagent_results', []))}")
        console.print(f"     Files written:    {state.get('file_manifest', [])}")
        answer_preview = str(state.get('final_answer', ''))[:200]
        console.print(f"     Final answer:     {answer_preview}…")
    except Exception as exc:
        console.print(f"     [dim]stub mode: {exc}[/dim]")

    console.print("\n  [bold green]✅ All Deep Agents checklist items demonstrated![/bold green]\n")


async def demo_full_pipeline(
    period: MonthEndPeriod,
    user_id: str,
    company_id: str,
    auto_approve: bool = True,
) -> None:
    """Full LangGraph orchestrator pipeline including HITL."""
    _section("Full Orchestrator Pipeline (LangGraph + HITL)")

    session_mgr = SessionManager()
    session     = await session_mgr.load_or_create(user_id, company_id)
    session     = await session_mgr.update_active_period(session, period)

    console.print(f"  Session: [cyan]{session.session_id}[/cyan]  Thread: [cyan]{session.thread_id}[/cyan]")

    orchestrator = build_month_end_graph()
    console.print("  [dim]Invoking orchestrator…[/dim]")

    try:
        state = await orchestrator.run(period=period, session=session, thread_id=session.thread_id)
    except Exception as exc:
        logger.warning("Orchestrator raised %s – continuing with demo state.", exc)
        state = {}

    # Simulate HITL approval
    hitl_req  = (state or {}).get("hitl_request") or {}
    request_id = hitl_req.get("id", "demo-request-id")

    if auto_approve:
        console.print(f"  [yellow]Auto-approving HITL (request_id={request_id})[/yellow]")
    else:
        input(f"  HITL paused. Press Enter to approve (request_id={request_id}): ")

    approval = HITLResponse(
        request_id=request_id,
        status=ApprovalStatus.APPROVED,
        reviewer=user_id,
        comment="Approved in demo.",
    )
    try:
        final = await orchestrator.resume(thread_id=session.thread_id, response=approval)
    except Exception as exc:
        logger.warning("Resume raised %s – using demo state.", exc)
        final = state

    # Display a brief report summary
    report = (final or {}).get("report") or {}
    if report:
        actuals = report.get("actuals", {})
        console.print(
            f"  Revenue: [cyan]${actuals.get('revenue', 0):,.0f}[/cyan]  |  "
            f"Net Income: [cyan]${actuals.get('net_income', 0):,.0f}[/cyan]  |  "
            f"Status: [green]{report.get('status', '?').upper()}[/green]"
        )
    else:
        # Fallback: show financial data directly
        actuals = fetch_financial_data.invoke({"year": period.year, "month": period.month})
        console.print(
            f"  Revenue: [cyan]${actuals['revenue']:,.0f}[/cyan]  |  "
            f"Net Income: [cyan]${actuals['net_income']:,.0f}[/cyan]"
        )

    await session_mgr.save_memory(session, f"Month-end for {period.label} approved and published.")
    console.print("  [green]✓ Pipeline + HITL demo complete[/green]\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def run_demo(
    user_id:        str  = "alice@acme.com",
    company_id:     str  = "acme-corp",
    year:           int  = 2025,
    month:          int  = 3,
    scenario:       Optional[str] = None,
    run_lcel:       bool = False,
    run_rag:        bool = False,
    run_super:      bool = False,
    run_deep_agent: bool = False,
    run_server:     bool = False,
    interactive:    bool = False,
) -> None:
    _banner()
    _settings_table()
    console.print()

    period = MonthEndPeriod(year=year, month=month)

    if run_server:
        _section("FastAPI Server (OpenWebUI backend)")
        console.print("  Starting server at [cyan]http://localhost:8000[/cyan]")
        console.print("  Point OpenWebUI to:  [cyan]http://localhost:8000/v1[/cyan]\n")
        console.print("  Or use Docker Compose:  [cyan]docker compose -f frontend/docker-compose.yml up[/cyan]\n")
        import uvicorn
        uvicorn.run("frontend.api.server:app", host="0.0.0.0", port=8000, reload=True)
        return

    if run_lcel:
        await demo_lcel_chains(period)

    if run_rag:
        await demo_rag_chain()

    if run_super:
        await demo_supervisor(period)

    if run_deep_agent:
        await demo_deep_agent(period)

    if scenario:
        await demo_scenarios(period, scenario)
    elif not run_lcel and not run_rag and not run_super and not run_deep_agent:
        # Run everything
        await demo_deep_agent(period)      # Deep Agents first (checklist showcase)
        await demo_lcel_chains(period)
        await demo_rag_chain()
        await demo_scenarios(period)
        await demo_supervisor(period)
        await demo_full_pipeline(period, user_id, company_id, auto_approve=not interactive)
    else:
        await demo_full_pipeline(period, user_id, company_id, auto_approve=not interactive)

    console.print(Panel.fit(
        "[bold green]All demos complete![/bold green]\n"
        "[dim]FastAPI + OpenWebUI: run  python main.py --server\n"
        "Docker:  docker compose -f frontend/docker-compose.yml up --build[/dim]",
        border_style="green",
    ))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Agentic Month-End Assistant Demo")
    p.add_argument("--user",        default="alice@acme.com")
    p.add_argument("--company",     default="acme-corp")
    p.add_argument("--year",        type=int, default=2025)
    p.add_argument("--month",       type=int, default=3)
    p.add_argument("--scenario",    default=None, choices=list(SCENARIO_REGISTRY) + [None],
                   help="Run a single named scenario")
    p.add_argument("--lcel",        action="store_true", help="Run LCEL chains showcase only")
    p.add_argument("--rag",         action="store_true", help="Run RAG demo only")
    p.add_argument("--supervisor",  action="store_true", help="Run supervisor demo only")
    p.add_argument("--server",      action="store_true", help="Start FastAPI + OpenWebUI backend")
    p.add_argument("--interactive", action="store_true", help="Real HITL (prompt for approval)")
    args = p.parse_args()

    asyncio.run(run_demo(
        user_id=args.user,
        company_id=args.company,
        year=args.year,
        month=args.month,
        scenario=args.scenario,
        run_lcel=args.lcel,
        run_rag=args.rag,
        run_super=args.supervisor,
        run_server=args.server,
        interactive=args.interactive,
    ))
