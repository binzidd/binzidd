"""
HITL (Human-in-the-Loop) Manager.

The manager is the single point of contact for all human approval workflows.
It coordinates between the orchestrator graph and the notification channels
(Teams + Slack) and maintains an in-memory registry of pending requests.

Workflow
────────
  1. Agent node calls  manager.request_approval(request)
  2. Manager fans out notifications to all configured channels in parallel
  3. LangGraph graph is interrupted at the hitl_checkpoint node
  4. Human clicks Approve / Reject in Teams or Slack → POST to /hitl/respond
  5. Operator (or test harness) calls manager.record_response(response)
  6. Graph resumes from the checkpoint with the response in state

Integration with LangGraph interrupt()
────────────────────────────────────────
The orchestrator node calls `langgraph.types.interrupt()` AFTER the manager
has dispatched notifications.  When the graph is resumed with
`Command(resume=response_dict)` the interrupt value becomes the HITL response
that routes the conditional edge.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from month_end_assistant.config import get_settings
from month_end_assistant.models import (
    ApprovalStatus,
    HITLRequest,
    HITLResponse,
    NotificationChannel,
    VarianceReport,
)
from month_end_assistant.hitl.teams import TeamsNotifier
from month_end_assistant.hitl.slack import SlackNotifier

logger = logging.getLogger(__name__)


class HITLManager:
    """
    Orchestrate human-approval flows across Teams and Slack.

    The manager is designed to be a singleton created once per application
    startup and shared across requests.

    Attributes
    ──────────
        pending  – dict of request_id → HITLRequest for open approvals
        history  – dict of request_id → HITLResponse for closed approvals
    """

    def __init__(self) -> None:
        self._settings    = get_settings()
        self._teams       = TeamsNotifier()
        self._slack       = SlackNotifier()
        self.pending:  Dict[str, HITLRequest]  = {}
        self.history:  Dict[str, HITLResponse] = {}

    # ── Variance-triggered approval ───────────────────────────────────────────

    def should_request_approval(self, variance: VarianceReport) -> bool:
        """
        Return True when a variance is material enough to require human sign-off.

        The threshold is configured via HITL_VARIANCE_THRESHOLD_PCT (default 5%).
        """
        return abs(variance.vs_budget_pct) >= self._settings.hitl_variance_threshold_pct

    def build_variance_request(
        self,
        period_label: str,
        variances: list[VarianceReport],
    ) -> HITLRequest:
        """
        Build a HITLRequest from a list of material variances.

        Groups all material variances into a single approval card to avoid
        notification fatigue.
        """
        material = [v for v in variances if self.should_request_approval(v)]
        context  = {
            "Period":           period_label,
            "Material Variances": str(len(material)),
        }
        for v in material:
            sign  = "▲" if v.vs_budget_pct > 0 else "▼"
            context[v.metric_name.replace("_", " ").title()] = (
                f"${v.actual:,.0f} ({sign}{abs(v.vs_budget_pct):.1f}% vs budget)"
            )

        summary = (
            f"{len(material)} material variance(s) detected for {period_label}. "
            "Finance controller approval required before report publication."
        )
        detail = (
            "Please review the variances above and approve or reject this month-end report. "
            f"Rejection will return the report to the analysis team for investigation. "
            f"Deadline: {self._settings.hitl_variance_threshold_pct}% threshold applied."
        )
        return HITLRequest(
            title=f"Month-End Approval Required – {period_label}",
            summary=summary,
            detail=detail,
            context=context,
            channels=[NotificationChannel.TEAMS, NotificationChannel.SLACK],
            requires_approval=True,
        )

    # ── Dispatch notifications ────────────────────────────────────────────────

    async def request_approval(self, request: HITLRequest) -> HITLRequest:
        """
        Dispatch the approval request to all configured channels in parallel.

        Registers the request in the pending dict and returns the request
        (with the generated ID) so the caller can embed it in graph state.
        """
        self.pending[request.id] = request
        logger.info(
            "HITL approval requested  request_id=%s  channels=%s",
            request.id,
            [c.value for c in request.channels],
        )

        # Fan out to Teams and Slack concurrently
        tasks = []
        if NotificationChannel.TEAMS in request.channels:
            tasks.append(self._teams.send(request))
        if NotificationChannel.SLACK in request.channels:
            tasks.append(self._slack.send(request))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for channel, result in zip(request.channels, results):
                if isinstance(result, Exception):
                    logger.error("Channel %s failed: %s", channel, result)
                elif result:
                    logger.info("Notification delivered via %s", channel)

        return request

    async def send_info_notification(
        self,
        title: str,
        summary: str,
        context: Optional[Dict] = None,
    ) -> None:
        """
        Send a non-blocking informational notification (no approval needed).

        Used to keep stakeholders updated on progress milestones such as
        "Deep research complete" or "Report generated".
        """
        info_request = HITLRequest(
            title=title,
            summary=summary,
            detail="",
            context=context or {},
            channels=[NotificationChannel.TEAMS, NotificationChannel.SLACK],
            requires_approval=False,
        )
        await self.request_approval(info_request)

    # ── Response recording ────────────────────────────────────────────────────

    def record_response(self, response: HITLResponse) -> None:
        """
        Record a human response, move the request from pending → history.

        Called by the /hitl/respond webhook endpoint (or by the test harness).
        After recording, the calling code should resume the LangGraph graph
        using:
            graph.invoke(Command(resume=response.model_dump()), config=...)
        """
        request = self.pending.pop(response.request_id, None)
        if request is None:
            logger.warning(
                "Response for unknown or already-closed request: %s",
                response.request_id,
            )
            return
        self.history[response.request_id] = response
        logger.info(
            "HITL response recorded  request_id=%s  status=%s  reviewer=%s",
            response.request_id,
            response.status.value,
            response.reviewer,
        )

    def get_response(self, request_id: str) -> Optional[HITLResponse]:
        """Return the stored response for *request_id*, or None if still pending."""
        return self.history.get(request_id)

    def is_timed_out(self, request: HITLRequest) -> bool:
        """Return True when the approval window has elapsed."""
        deadline = request.created_at + timedelta(minutes=request.deadline_minutes)
        return datetime.utcnow() > deadline

    def expire_timed_out(self) -> list[str]:
        """
        Mark all overdue pending requests as TIMED_OUT and return their IDs.

        Should be called periodically (e.g. by a background scheduler).
        """
        expired_ids = []
        for req_id, request in list(self.pending.items()):
            if self.is_timed_out(request):
                self.pending.pop(req_id)
                self.history[req_id] = HITLResponse(
                    request_id=req_id,
                    status=ApprovalStatus.TIMED_OUT,
                    reviewer="system",
                    comment="Approval window expired.",
                )
                expired_ids.append(req_id)
                logger.warning("HITL request timed out: %s", req_id)
        return expired_ids
