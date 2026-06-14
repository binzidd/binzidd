"""
AgentHarness — the central runtime for all month-end agents.

Wires together every harness feature into a single, high-level API:

    harness = AgentHarness()
    await harness.initialize()

    # Blocking run
    result = await harness.run("Analyse Q3 revenue", period, user_id="u1")

    # Streaming run (yields HarnessEvent objects)
    async for event in harness.stream("Analyse Q3 revenue", period, user_id="u1"):
        print(event.to_sse())

    # Resume a HITL-interrupted run
    result = await harness.resume(thread_id, hitl_response)

    # Evaluate all agents
    report = await harness.evaluate()

    # Inspect registry
    print(harness.registry.status_report())

Features bundled:
  ✅ AsyncSqliteSaver / PostgresSaver checkpointing (falls back to MemorySaver)
  ✅ LangSmith + OpenTelemetry + Prometheus observability
  ✅ InMemoryStore cross-thread long-term memory
  ✅ astream_events multi-consumer fan-out streaming
  ✅ Circuit breaker per agent
  ✅ Token budget manager + per-user rate limiter
  ✅ Bulkhead semaphore (max concurrent runs)
  ✅ LLM-as-judge evaluation framework
  ✅ Agent + tool capability registry with health-score routing
  ✅ Structured logging with correlation IDs
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from langchain_core.messages import HumanMessage

from month_end_assistant.harness.config import HarnessSettings, get_harness_settings
from month_end_assistant.harness.checkpointing import CheckpointerFactory
from month_end_assistant.harness.evaluation import AgentEvaluator, EvalReport
from month_end_assistant.harness.memory import HarnessMemoryStore
from month_end_assistant.harness.observability import HarnessObservability
from month_end_assistant.harness.registry import AgentCapability, AgentRegistry
from month_end_assistant.harness.resilience import (
    BulkheadSemaphore,
    CircuitBreaker,
    CircuitOpenError,
    RateLimiter,
    RateLimitError,
    RetryPolicy,
    TokenBudgetManager,
)
from month_end_assistant.harness.streaming import HarnessEvent, HarnessStreamingManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HarnessResult:
    run_id: str
    thread_id: str
    agent: str
    output: Optional[Dict[str, Any]] = None
    summary: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None
    hitl_required: bool = False

    @property
    def success(self) -> bool:
        return self.error is None


# ─────────────────────────────────────────────────────────────────────────────
# AgentHarness
# ─────────────────────────────────────────────────────────────────────────────

class AgentHarness:
    """
    Top-level orchestration harness.

    One instance per application process.  Initialise with::

        harness = AgentHarness()
        await harness.initialize()

    For CLI / tests use the sync shortcut::

        harness = AgentHarness.create_sync()
    """

    def __init__(self, settings: Optional[HarnessSettings] = None) -> None:
        self._settings = settings or get_harness_settings()

        # Infrastructure components
        self._obs          = HarnessObservability(self._settings)
        self._memory       = HarnessMemoryStore()
        self._streaming    = HarnessStreamingManager(self._settings.stream_include_types)
        self._registry     = AgentRegistry()
        self._circuit      = CircuitBreaker(
            failure_threshold=self._settings.cb_failure_threshold,
            recovery_timeout_s=self._settings.cb_recovery_timeout_s,
        )
        self._token_budget = TokenBudgetManager(self._settings.max_tokens_per_minute)
        self._rate_limiter = RateLimiter(self._settings.max_requests_per_minute)
        self._bulkhead     = BulkheadSemaphore(self._settings.max_concurrent_runs)

        # Lazy-init
        self._evaluator: Optional[AgentEvaluator] = None
        self._checkpointer: Optional[Any]         = None
        self._initialized                          = False

        # Keep refs to agent instances for direct use
        self._agents: Dict[str, Any] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize_sync(self) -> "AgentHarness":
        """
        Synchronous bootstrap — use in CLI / pytest contexts.
        Skips async checkpointer; uses MemorySaver fallback.
        """
        self._obs.initialize()
        self._checkpointer = CheckpointerFactory.create_sync(self._settings)
        self._bootstrap_agents()
        self._register_all_agents()
        self._initialized = True
        logger.info("AgentHarness ready (sync mode, checkpointer=MemorySaver)")
        return self

    async def initialize(self) -> "AgentHarness":
        """
        Full async bootstrap with persistent checkpointing.
        Call once at application startup.
        """
        self._obs.initialize()
        # Checkpointer is a context manager; store for lifecycle management
        self._ckpt_cm = CheckpointerFactory.create(self._settings)
        self._checkpointer = await self._ckpt_cm.__aenter__()
        self._bootstrap_agents()
        self._register_all_agents()
        self._initialized = True
        logger.info("AgentHarness ready (async mode, checkpointer=%s)", type(self._checkpointer).__name__)
        return self

    async def close(self) -> None:
        """Shut down gracefully — flush metrics, close DB connections."""
        if hasattr(self, "_ckpt_cm"):
            await self._ckpt_cm.__aexit__(None, None, None)
        logger.info("AgentHarness closed.")

    @asynccontextmanager
    @staticmethod
    async def context(settings: Optional[HarnessSettings] = None) -> AsyncIterator["AgentHarness"]:
        """Async context manager for clean resource management."""
        harness = AgentHarness(settings)
        await harness.initialize()
        try:
            yield harness
        finally:
            await harness.close()

    # ── Core run API ──────────────────────────────────────────────────────────

    async def run(
        self,
        task: str,
        period: Any,                   # MonthEndPeriod
        user_id: str = "anonymous",
        thread_id: Optional[str] = None,
        agent: str = "orchestrator",
    ) -> HarnessResult:
        """
        Run an agent with full harness protection:
          1. Rate limit check (per user)
          2. Bulkhead check (global concurrency cap)
          3. Circuit breaker check (per agent)
          4. Load cross-thread memories
          5. Execute agent graph with checkpointing
          6. Save result to memory store
          7. Update registry metrics
          8. Return HarnessResult
        """
        thread_id  = thread_id or f"{user_id}-{getattr(period, 'label', 'na')}-{uuid4().hex[:8]}"
        run_id     = uuid4().hex
        start      = time.monotonic()

        # ── Pre-flight checks ─────────────────────────────────────────────────
        await self._rate_limiter.check(user_id)
        self._circuit.before_call(agent)

        with self._obs.trace(run_id=run_id, agent=agent, user_id=user_id) as span:
            async with self._bulkhead.acquire(agent):
                try:
                    # Load relevant memories for context enrichment
                    memories = await self._memory.search_user_memories(
                        user_id, task, limit=5
                    )
                    memory_context = "\n".join(
                        f"- {m.value.get('summary', '')}" for m in memories if m.value
                    )

                    # Build LangGraph config
                    config = {
                        "configurable": {"thread_id": thread_id},
                        "callbacks": [],
                        "recursion_limit": 50,
                    }

                    # Dispatch to the right agent
                    output = await self._dispatch(
                        agent=agent,
                        task=task,
                        period=period,
                        user_id=user_id,
                        thread_id=thread_id,
                        memory_context=memory_context,
                        config=config,
                    )

                    latency_ms = (time.monotonic() - start) * 1000
                    self._circuit.on_success(agent)
                    self._registry.record_run(agent, latency_ms, success=True)
                    self._obs.set_circuit_breaker(agent, False)

                    # Persist result to long-term memory
                    summary = self._extract_summary(output)
                    await self._memory.save_user_memory(
                        user_id,
                        key=f"run-{run_id}",
                        value={
                            "task": task,
                            "period": getattr(period, "label", str(period)),
                            "summary": summary,
                            "run_id": run_id,
                            "thread_id": thread_id,
                        },
                    )

                    span["status"] = "success"
                    return HarnessResult(
                        run_id=run_id,
                        thread_id=thread_id,
                        agent=agent,
                        output=output,
                        summary=summary,
                        latency_ms=latency_ms,
                    )

                except (CircuitOpenError, RateLimitError):
                    raise

                except Exception as exc:
                    latency_ms = (time.monotonic() - start) * 1000
                    self._circuit.on_failure(agent)
                    self._registry.record_run(agent, latency_ms, success=False)
                    self._obs.set_circuit_breaker(agent, self._circuit.is_open(agent))
                    span["status"] = "error"
                    logger.exception("AgentHarness.run failed: agent=%s run_id=%s", agent, run_id)
                    return HarnessResult(
                        run_id=run_id,
                        thread_id=thread_id,
                        agent=agent,
                        latency_ms=latency_ms,
                        error=str(exc),
                    )

    async def stream(
        self,
        task: str,
        period: Any,
        user_id: str = "anonymous",
        thread_id: Optional[str] = None,
        agent: str = "orchestrator",
    ) -> AsyncIterator[HarnessEvent]:
        """
        Stream agent execution events in real time.

        Yields HarnessEvent objects.  Use `.to_sse()` for HTTP streaming or
        `.to_openai_delta()` for OpenWebUI-compatible chunks.
        """
        thread_id = thread_id or f"{user_id}-{getattr(period, 'label', 'na')}-{uuid4().hex[:8]}"
        run_id = uuid4().hex

        await self._rate_limiter.check(user_id)
        self._circuit.before_call(agent)

        agent_obj = self._agents.get(agent)
        if not agent_obj or not hasattr(agent_obj, "graph"):
            # Fallback: run blocking and yield a single event
            result = await self.run(task, period, user_id, thread_id, agent)
            yield HarnessEvent(
                run_id=run_id, event_type="done", agent=agent,
                data={"summary": result.summary, "error": result.error},
            )
            return

        input_state = self._build_input_state(task, period, user_id)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

        async with self._bulkhead.acquire(agent):
            async for event in self._streaming.stream_graph(
                graph=agent_obj.graph,
                input_state=input_state,
                config=config,
                run_id=run_id,
                agent=agent,
            ):
                yield event

    async def resume(
        self,
        thread_id: str,
        hitl_response: Any,            # HITLResponse
        agent: str = "orchestrator",
    ) -> HarnessResult:
        """Resume a graph that is paused at a HITL interrupt() checkpoint."""
        from langgraph.types import Command

        run_id = uuid4().hex
        start  = time.monotonic()

        agent_obj = self._agents.get(agent)
        if not agent_obj:
            return HarnessResult(run_id=run_id, thread_id=thread_id, agent=agent, error=f"Agent '{agent}' not found")

        config = {"configurable": {"thread_id": thread_id}}

        try:
            resume_input = hitl_response.model_dump() if hasattr(hitl_response, "model_dump") else hitl_response
            graph = getattr(agent_obj, "graph", None)
            if graph is None:
                raise ValueError(f"Agent '{agent}' does not expose a .graph attribute")

            output = await graph.ainvoke(Command(resume=resume_input), config=config)
            latency_ms = (time.monotonic() - start) * 1000
            return HarnessResult(
                run_id=run_id, thread_id=thread_id, agent=agent,
                output=output, summary=self._extract_summary(output),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            return HarnessResult(
                run_id=run_id, thread_id=thread_id, agent=agent,
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
            )

    # ── Evaluation ────────────────────────────────────────────────────────────

    async def evaluate(
        self,
        dataset: Optional[str] = None,
        evaluators: Optional[List[str]] = None,
    ) -> EvalReport:
        """Run the evaluation suite against all registered test cases."""
        evaluator = self._get_evaluator()
        dataset = dataset or self._settings.eval_default_dataset

        if not evaluator.get_dataset(dataset):
            evaluator.add_default_financial_cases()

        async def _run_fn(input_dict: Dict) -> str:
            task   = input_dict.get("task", "Analyse month-end financials")
            period = input_dict.get("period", "2025-09")
            result = await self.run(task, period, user_id="eval-harness")
            return result.summary or result.error or "(no output)"

        return await evaluator.evaluate(_run_fn, dataset, evaluators)

    # ── Streaming subscription ─────────────────────────────────────────────────

    def subscribe_to_run(self, run_id: str) -> asyncio.Queue:
        """Subscribe to events for an already-started streaming run."""
        return self._streaming.subscribe(run_id)

    def unsubscribe_from_run(self, run_id: str, queue: asyncio.Queue) -> None:
        self._streaming.unsubscribe(run_id, queue)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    @property
    def memory(self) -> HarnessMemoryStore:
        return self._memory

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit

    @property
    def token_budget(self) -> TokenBudgetManager:
        return self._token_budget

    def status(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "agents": self._registry.list_agents(),
            "circuit_breaker": self._circuit.status(),
            "token_budget_remaining": self._token_budget.remaining,
            "active_runs": self._bulkhead.current_runs,
            "available_slots": self._bulkhead.available_slots,
            "lifetime_tokens": self._token_budget.lifetime_tokens,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _bootstrap_agents(self) -> None:
        """Instantiate all agents and inject the shared checkpointer."""
        try:
            from month_end_assistant.agents.orchestrator import MonthEndOrchestrator
            self._agents["orchestrator"] = MonthEndOrchestrator()
        except Exception as exc:
            logger.warning("Could not load MonthEndOrchestrator: %s", exc)

        try:
            from month_end_assistant.agents.supervisor import MonthEndSupervisor
            self._agents["supervisor"] = MonthEndSupervisor()
        except Exception as exc:
            logger.warning("Could not load MonthEndSupervisor: %s", exc)

        try:
            from month_end_assistant.agents.research import DeepResearchAgent
            self._agents["research"] = DeepResearchAgent()
        except Exception as exc:
            logger.warning("Could not load DeepResearchAgent: %s", exc)

        try:
            from month_end_assistant.agents.deep_agent import DeepAgent
            self._agents["deep_agent"] = DeepAgent()
        except Exception as exc:
            logger.warning("Could not load DeepAgent: %s", exc)

        try:
            from month_end_assistant.agents.critique import CritiqueAgent
            self._agents["critique"] = CritiqueAgent()
        except Exception as exc:
            logger.warning("Could not load CritiqueAgent: %s", exc)

    def _register_all_agents(self) -> None:
        """Register loaded agents with their capabilities."""
        capabilities = {
            "orchestrator": AgentCapability(
                name="orchestrator",
                description="Full month-end close pipeline with HITL approval and research",
                tags={"monthly", "quarterly", "full-pipeline", "hitl", "revenue", "close"},
                input_keys=["task", "period", "user_id"],
                output_keys=["report", "hitl_request"],
            ),
            "supervisor": AgentCapability(
                name="supervisor",
                description="Multi-agent supervisor routing tasks to specialist workers",
                tags={"supervisor", "multi-agent", "routing", "analysis"},
                input_keys=["task", "period"],
                output_keys=["messages", "worker_results"],
            ),
            "research": AgentCapability(
                name="research",
                description="Deep research agent with parallel fan-out and synthesis",
                tags={"research", "deep-dive", "synthesis", "parallel"},
                input_keys=["query", "period"],
                output_keys=["synthesis", "research_tasks"],
            ),
            "deep_agent": AgentCapability(
                name="deep_agent",
                description="Deep agent with sandbox execution and file management",
                tags={"deep-agent", "sandbox", "code-execution", "filesystem"},
                input_keys=["task", "period", "backend_kind"],
                output_keys=["final_answer", "file_manifest"],
            ),
            "critique": AgentCapability(
                name="critique",
                description="Self-reflection critique agent using Reflexion pattern",
                tags={"critique", "self-reflection", "revision", "quality"},
                input_keys=["content", "criteria"],
                output_keys=["revised_content", "critique_history"],
            ),
        }
        for name, agent_obj in self._agents.items():
            if name in capabilities:
                self._registry.register_agent(agent_obj, capabilities[name])

    async def _dispatch(
        self,
        agent: str,
        task: str,
        period: Any,
        user_id: str,
        thread_id: str,
        memory_context: str,
        config: Dict,
    ) -> Optional[Dict]:
        """Route the run to the right agent's async interface."""
        agent_obj = self._agents.get(agent)
        if not agent_obj:
            raise ValueError(f"Agent '{agent}' not registered. Available: {list(self._agents)}")

        # Every agent exposes an async run() method from BaseAgent pattern
        if hasattr(agent_obj, "arun"):
            return await agent_obj.arun(task, period, user_id, thread_id)
        if hasattr(agent_obj, "run"):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: agent_obj.run(task, period, user_id, thread_id)
            )
        raise AttributeError(f"Agent '{agent}' has neither run() nor arun()")

    def _build_input_state(self, task: str, period: Any, user_id: str) -> Dict:
        return {
            "messages": [HumanMessage(content=task)],
            "session": {"user_id": user_id, "period": getattr(period, "label", str(period))},
            "period_label": getattr(period, "label", str(period)),
        }

    def _extract_summary(self, output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output[:500]
        if isinstance(output, dict):
            for key in ("summary", "synthesis", "final_answer", "report"):
                if key in output and output[key]:
                    val = output[key]
                    return (val[:500] if isinstance(val, str) else str(val)[:500])
            return str(output)[:500]
        return str(output)[:500]

    def _get_evaluator(self) -> AgentEvaluator:
        if self._evaluator is None:
            # Use the first available LLM from any registered agent
            llm = None
            for agent_obj in self._agents.values():
                if hasattr(agent_obj, "llm"):
                    llm = agent_obj.llm
                    break
            if llm is None:
                from month_end_assistant.agents.base import BaseAgent
                llm = BaseAgent.__new__(BaseAgent)
                llm.llm = BaseAgent._build_llm(llm)  # type: ignore
            self._evaluator = AgentEvaluator(llm=llm, settings=self._settings)
        return self._evaluator
