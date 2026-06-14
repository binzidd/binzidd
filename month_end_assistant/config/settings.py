"""
Application settings loaded from environment variables.

Uses pydantic-settings so every value can be overridden via .env or the
real process environment without changing code.
"""

from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Month-End Assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── AWS ──────────────────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: str = Field(default="", description="AWS access key")
    aws_secret_access_key: str = Field(default="", description="AWS secret key")

    # Bedrock model used for all LLM calls
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        description="Bedrock model ID",
    )

    # AWS AgentCore identifiers
    agentcore_memory_id: str = Field(default="", description="AgentCore Memory ID")
    agentcore_agent_id: str = Field(default="", description="AgentCore Agent ID")
    agentcore_agent_alias_id: str = Field(
        default="TSTALIASID", description="AgentCore Agent Alias ID"
    )

    # ── MS Teams ─────────────────────────────────────────────────────────────
    teams_webhook_url: str = Field(default="", description="Teams incoming webhook URL")

    # ── Slack ─────────────────────────────────────────────────────────────────
    slack_bot_token: str = Field(default="", description="Slack bot OAuth token")
    slack_approval_channel: str = Field(
        default="#month-end-approvals", description="Slack approval channel"
    )

    # ── Behaviour tuning ─────────────────────────────────────────────────────
    # Variance (as % of budget) that triggers a HITL approval request
    hitl_variance_threshold_pct: float = Field(
        default=5.0, description="Variance threshold for HITL (%)"
    )
    # Maximum reflection iterations before the deep-research loop exits
    max_research_iterations: int = Field(
        default=3, description="Max deep-research reflection rounds"
    )
    # Run financial code inside the Python sandbox
    sandbox_enabled: bool = Field(default=True, description="Enable code sandbox")

    log_level: str = Field(default="INFO", description="Log level")

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def has_teams(self) -> bool:
        return bool(self.teams_webhook_url)

    @property
    def has_slack(self) -> bool:
        return bool(self.slack_bot_token)

    @property
    def has_agentcore(self) -> bool:
        return bool(self.agentcore_memory_id)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
