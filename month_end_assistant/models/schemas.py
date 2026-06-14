"""
Pydantic data models shared across the entire Month-End Assistant.

Design principles
─────────────────
• All models are immutable (model_config frozen=True) where practical.
• Enums are defined once and re-used so comparisons are always type-safe.
• AgentState is the LangGraph TypedDict that flows through every graph node.
"""

from __future__ import annotations

import operator
import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class ApprovalStatus(str, Enum):
    PENDING   = "pending"
    APPROVED  = "approved"
    REJECTED  = "rejected"
    ESCALATED = "escalated"
    TIMED_OUT = "timed_out"


class NotificationChannel(str, Enum):
    TEAMS = "teams"
    SLACK = "slack"


class ReportStatus(str, Enum):
    DRAFT     = "draft"
    IN_REVIEW = "in_review"
    APPROVED  = "approved"
    PUBLISHED = "published"


class ResearchStatus(str, Enum):
    PENDING    = "pending"
    IN_FLIGHT  = "in_flight"
    COMPLETE   = "complete"
    FAILED     = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Financial domain models
# ─────────────────────────────────────────────────────────────────────────────

class MonthEndPeriod(BaseModel):
    """Identifies a specific accounting period (year + month)."""

    year: int = Field(..., ge=2000, le=2100)
    month: int = Field(..., ge=1, le=12)

    @property
    def label(self) -> str:
        """Human-readable label, e.g. 'March 2025'."""
        import calendar
        return f"{calendar.month_name[self.month]} {self.year}"

    @property
    def iso(self) -> str:
        """ISO month string, e.g. '2025-03'."""
        return f"{self.year}-{self.month:02d}"


class FinancialMetrics(BaseModel):
    """Snapshot of key financial figures for one period."""

    period: MonthEndPeriod
    revenue: float              = Field(..., description="Total revenue (USD)")
    cost_of_goods_sold: float   = Field(..., description="COGS (USD)")
    gross_profit: float         = Field(0.0)
    operating_expenses: float   = Field(..., description="OpEx (USD)")
    ebitda: float               = Field(0.0)
    net_income: float           = Field(..., description="Net income (USD)")
    cash_flow_operations: float = Field(..., description="Operating cash flow (USD)")
    accounts_receivable: float  = Field(..., description="AR balance (USD)")
    accounts_payable: float     = Field(..., description="AP balance (USD)")
    inventory_value: float      = Field(..., description="Inventory at cost (USD)")

    @model_validator(mode="after")
    def _compute_derived(self) -> "FinancialMetrics":
        """Calculate gross profit and EBITDA if not provided."""
        self.gross_profit = self.revenue - self.cost_of_goods_sold
        self.ebitda = self.gross_profit - self.operating_expenses
        return self


class VarianceReport(BaseModel):
    """Variance of actuals vs budget/prior-period for each metric."""

    period: MonthEndPeriod
    metric_name: str
    actual: float
    budget: float
    prior_period: float

    @property
    def vs_budget_pct(self) -> float:
        """Variance as % of budget (negative = unfavourable)."""
        if self.budget == 0:
            return 0.0
        return ((self.actual - self.budget) / abs(self.budget)) * 100

    @property
    def vs_prior_pct(self) -> float:
        """Variance as % of prior period."""
        if self.prior_period == 0:
            return 0.0
        return ((self.actual - self.prior_period) / abs(self.prior_period)) * 100

    @property
    def is_material(self) -> bool:
        """True when the absolute budget variance exceeds 5 %."""
        return abs(self.vs_budget_pct) > 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Research models (used by the LangGraph deep-research agent)
# ─────────────────────────────────────────────────────────────────────────────

class ResearchTask(BaseModel):
    """A single research sub-task created by the planner node."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str
    category: str = Field(
        default="general",
        description="e.g. 'revenue', 'cost', 'benchmark', 'accounting_standard'",
    )
    status: ResearchStatus = ResearchStatus.PENDING
    findings: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ResearchPlan(BaseModel):
    """The decomposed plan produced by the planner node."""

    main_query: str
    tasks: List[ResearchTask]
    iteration: int = 1


class ResearchResult(BaseModel):
    """Aggregated output after all research tasks complete."""

    plan: ResearchPlan
    synthesis: str
    gaps: List[str] = Field(default_factory=list)
    is_sufficient: bool = False
    total_iterations: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# HITL (Human-in-the-Loop) models
# ─────────────────────────────────────────────────────────────────────────────

class HITLRequest(BaseModel):
    """Notification + approval request sent to Teams and/or Slack."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    summary: str
    detail: str
    context: Dict[str, Any] = Field(default_factory=dict)
    channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.TEAMS, NotificationChannel.SLACK]
    )
    requires_approval: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    deadline_minutes: int = Field(default=60, description="Approval window in minutes")


class HITLResponse(BaseModel):
    """Response captured after a human interacts with the approval request."""

    request_id: str
    status: ApprovalStatus
    reviewer: str = "unknown"
    comment: str = ""
    responded_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Report model
# ─────────────────────────────────────────────────────────────────────────────

class MonthEndReport(BaseModel):
    """Final month-end close report ready for distribution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    period: MonthEndPeriod
    metrics: FinancialMetrics
    variances: List[VarianceReport]
    research_synthesis: str = ""
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    sandbox_outputs: List[str] = Field(default_factory=list)
    status: ReportStatus = ReportStatus.DRAFT
    hitl_approval: Optional[HITLResponse] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def has_material_variances(self) -> bool:
        return any(v.is_material for v in self.variances)


# ─────────────────────────────────────────────────────────────────────────────
# Session / memory models
# ─────────────────────────────────────────────────────────────────────────────

class UserSession(BaseModel):
    """Persistent user session loaded from (or saved to) AgentCore Memory."""

    user_id: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company_id: str
    display_name: str = ""
    # LangGraph thread_id – allows resuming interrupted graphs
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # AgentCore Memory session token
    memory_session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    active_period: Optional[MonthEndPeriod] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    prior_reports: List[str] = Field(default_factory=list, description="Report IDs")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox result
# ─────────────────────────────────────────────────────────────────────────────

class SandboxResult(BaseModel):
    """Output returned by the sandboxed code executor."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    execution_time_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph AgentState TypedDict
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Shared state that flows through every node in the orchestrator graph.

    • messages      – full conversation history (auto-merged by add_messages)
    • research_tasks – list of ResearchTask dicts accumulated across parallel nodes
    • report         – the MonthEndReport being built (None until analysis phase)
    • hitl_request   – pending approval request (None when no approval needed)
    • hitl_response  – human response once received (None until responded)
    • session        – the current UserSession (dict form for serialisation)
    • iteration      – current deep-research reflection round
    • next_step      – routing signal set by conditional edge functions
    • error          – any error message that caused a node to fail
    """

    messages: Annotated[List[BaseMessage], add_messages]
    research_tasks: Annotated[List[dict], operator.add]
    report: Optional[dict]
    hitl_request: Optional[dict]
    hitl_response: Optional[dict]
    session: dict
    iteration: int
    next_step: str
    error: str
