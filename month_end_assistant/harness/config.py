"""
Harness configuration — separate from application settings.

Covers: checkpointing backend, observability endpoints, rate-limit budgets,
circuit-breaker thresholds, and evaluation parameters.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HarnessSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HARNESS_", env_file=".env", extra="ignore")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_backend: Literal["sqlite", "postgres", "memory"] = "sqlite"
    sqlite_db_path: str = "./harness_checkpoints.db"
    postgres_dsn: Optional[str] = Field(None, alias="HARNESS_POSTGRES_DSN")

    # ── Observability ─────────────────────────────────────────────────────────
    langsmith_api_key: Optional[str] = Field(None, alias="LANGCHAIN_API_KEY")
    langsmith_project: str = "month-end-assistant"
    langsmith_tracing: bool = False

    otel_endpoint: Optional[str] = None          # e.g. "http://localhost:4317"
    otel_service_name: str = "month-end-assistant"
    otel_insecure: bool = True                    # disable TLS for local collectors

    prometheus_enabled: bool = True
    prometheus_port: int = 9090

    structlog_json: bool = False                  # pretty console vs JSON lines

    # ── Rate limiting / token budget ──────────────────────────────────────────
    max_tokens_per_minute: int = 100_000
    max_requests_per_minute: int = 60
    max_concurrent_runs: int = 10

    # ── Circuit breaker ───────────────────────────────────────────────────────
    cb_failure_threshold: int = 5
    cb_recovery_timeout_s: int = 30

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_default_dataset: str = "month-end-default"
    eval_max_concurrency: int = 4
    eval_pass_threshold: float = 0.75             # min avg score to pass

    # ── Streaming ─────────────────────────────────────────────────────────────
    stream_event_version: Literal["v1", "v2"] = "v2"
    stream_include_types: list[str] = Field(
        default_factory=lambda: ["on_llm_stream", "on_chain_end", "on_tool_end", "on_custom_event"]
    )


_harness_settings: Optional[HarnessSettings] = None


def get_harness_settings() -> HarnessSettings:
    global _harness_settings
    if _harness_settings is None:
        _harness_settings = HarnessSettings()
    return _harness_settings
