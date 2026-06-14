"""
Full observability stack for the agent harness.

Layers:
  1. Structured logging  – structlog with correlation IDs on every log record
  2. LangSmith tracing   – automatic via LANGCHAIN_TRACING_V2 env var
  3. OpenTelemetry        – distributed traces exported via OTLP
  4. Prometheus metrics   – counters, histograms, gauges for every graph run

All instrumentation is optional and degrades gracefully when SDKs are absent.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
from uuid import uuid4

from month_end_assistant.harness.config import HarnessSettings

logger = logging.getLogger(__name__)

# ── Optional: structlog ───────────────────────────────────────────────────────
try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False

# ── Optional: OpenTelemetry ───────────────────────────────────────────────────
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

# ── Optional: Prometheus ──────────────────────────────────────────────────────
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metric objects (created once at module level)
# ─────────────────────────────────────────────────────────────────────────────

if _HAS_PROMETHEUS:
    AGENT_RUNS_TOTAL = Counter(
        "harness_agent_runs_total",
        "Total agent graph invocations",
        ["agent_name", "status"],
    )
    AGENT_RUN_DURATION = Histogram(
        "harness_agent_run_duration_seconds",
        "Agent graph run duration",
        ["agent_name"],
        buckets=[1, 5, 10, 30, 60, 120, 300],
    )
    AGENT_TOKENS_USED = Counter(
        "harness_tokens_used_total",
        "Cumulative tokens consumed",
        ["agent_name", "token_type"],
    )
    HITL_REQUESTS_TOTAL = Counter(
        "harness_hitl_requests_total",
        "HITL approval requests sent",
        ["channel", "outcome"],
    )
    CIRCUIT_BREAKER_STATE = Gauge(
        "harness_circuit_breaker_open",
        "1 when circuit breaker is open, 0 when closed",
        ["agent_name"],
    )
    ACTIVE_RUNS = Gauge(
        "harness_active_runs",
        "Number of currently running agent graphs",
        ["agent_name"],
    )
else:
    # Stub objects so callers don't need to guard every metric call
    class _NullMetric:
        def labels(self, **_): return self
        def inc(self, *a, **k): pass
        def dec(self, *a, **k): pass
        def set(self, *a, **k): pass
        def observe(self, *a, **k): pass
        def time(self): return _NullCtx()

    class _NullCtx:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    _null = _NullMetric()
    AGENT_RUNS_TOTAL = _null
    AGENT_RUN_DURATION = _null
    AGENT_TOKENS_USED = _null
    HITL_REQUESTS_TOTAL = _null
    CIRCUIT_BREAKER_STATE = _null
    ACTIVE_RUNS = _null


# ─────────────────────────────────────────────────────────────────────────────
# Main observability class
# ─────────────────────────────────────────────────────────────────────────────

class HarnessObservability:
    """
    One-stop shop for all observability features.

    Instantiate once at harness startup::

        obs = HarnessObservability(settings)
        await obs.initialize()

        with obs.trace("my-run-id", agent="orchestrator", user_id="u1") as span:
            ...
    """

    def __init__(self, settings: HarnessSettings) -> None:
        self._settings = settings
        self._tracer: Optional[Any] = None
        self._prometheus_started = False

    def initialize(self) -> None:
        """Synchronous setup — call at startup before serving requests."""
        self._setup_logging()
        self._setup_langsmith()
        self._setup_otel()
        self._setup_prometheus()

    # ── Structured logging ────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        if not _HAS_STRUCTLOG:
            logging.basicConfig(
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                level=logging.INFO,
            )
            return

        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
        ]
        if self._settings.structlog_json:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )

    # ── LangSmith ─────────────────────────────────────────────────────────────

    def _setup_langsmith(self) -> None:
        if self._settings.langsmith_api_key and self._settings.langsmith_tracing:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_API_KEY", self._settings.langsmith_api_key)
            os.environ.setdefault("LANGCHAIN_PROJECT", self._settings.langsmith_project)
            logger.info("LangSmith tracing enabled → project=%s", self._settings.langsmith_project)

    # ── OpenTelemetry ─────────────────────────────────────────────────────────

    def _setup_otel(self) -> None:
        if not _HAS_OTEL or not self._settings.otel_endpoint:
            return

        resource = Resource(attributes={"service.name": self._settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=self._settings.otel_endpoint,
            insecure=self._settings.otel_insecure,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(__name__)
        logger.info("OTel tracing enabled → %s", self._settings.otel_endpoint)

    # ── Prometheus ────────────────────────────────────────────────────────────

    def _setup_prometheus(self) -> None:
        if not _HAS_PROMETHEUS or not self._settings.prometheus_enabled:
            return
        if self._prometheus_started:
            return
        try:
            start_http_server(self._settings.prometheus_port)
            self._prometheus_started = True
            logger.info("Prometheus metrics → http://localhost:%d/metrics", self._settings.prometheus_port)
        except OSError:
            logger.warning("Prometheus port %d already in use — metrics server skipped.", self._settings.prometheus_port)

    # ── Trace context manager ─────────────────────────────────────────────────

    @contextmanager
    def trace(
        self,
        run_id: str,
        agent: str = "unknown",
        user_id: str = "anonymous",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager that:
          - Binds run_id + user_id to structlog context
          - Opens an OTel span (if configured)
          - Records Prometheus start/end metrics
          - Returns a span dict so callers can annotate it

        Usage::
            with obs.trace(run_id, agent="orchestrator", user_id="u42") as span:
                span["result"] = "success"
        """
        span_data: Dict[str, Any] = {"run_id": run_id, "agent": agent, "user_id": user_id}
        start_ts = time.monotonic()

        # Structlog context
        if _HAS_STRUCTLOG:
            structlog.contextvars.bind_contextvars(run_id=run_id, agent=agent, user_id=user_id)

        ACTIVE_RUNS.labels(agent_name=agent).inc()

        otel_span = None
        if self._tracer:
            otel_span = self._tracer.start_span(
                f"harness.{agent}.run",
                attributes={"run_id": run_id, "user_id": user_id},
            )

        try:
            yield span_data
            status = span_data.get("status", "success")
        except Exception as exc:
            status = "error"
            span_data["error"] = str(exc)
            if otel_span:
                otel_span.record_exception(exc)
            raise
        finally:
            elapsed = time.monotonic() - start_ts
            ACTIVE_RUNS.labels(agent_name=agent).dec()
            AGENT_RUNS_TOTAL.labels(agent_name=agent, status=status).inc()
            AGENT_RUN_DURATION.labels(agent_name=agent).observe(elapsed)

            if otel_span:
                otel_span.set_attribute("status", status)
                otel_span.set_attribute("duration_s", round(elapsed, 3))
                otel_span.end()

            if _HAS_STRUCTLOG:
                structlog.contextvars.unbind_contextvars("run_id", "agent", "user_id")

    def record_tokens(self, agent: str, prompt_tokens: int, completion_tokens: int) -> None:
        AGENT_TOKENS_USED.labels(agent_name=agent, token_type="prompt").inc(prompt_tokens)
        AGENT_TOKENS_USED.labels(agent_name=agent, token_type="completion").inc(completion_tokens)

    def record_hitl(self, channel: str, outcome: str) -> None:
        HITL_REQUESTS_TOTAL.labels(channel=channel, outcome=outcome).inc()

    def set_circuit_breaker(self, agent: str, is_open: bool) -> None:
        CIRCUIT_BREAKER_STATE.labels(agent_name=agent).set(1 if is_open else 0)
