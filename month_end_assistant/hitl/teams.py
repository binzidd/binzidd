"""
Microsoft Teams HITL Notifier.

Sends rich Adaptive Card notifications to a Teams channel via an Incoming
Webhook.  The card includes:

  • A colour-coded header (red = approval needed, amber = warning)
  • A summary fact table (metric → value)
  • Action buttons: Approve / Reject / Escalate

The approve/reject URLs would normally point back to the assistant's webhook
endpoint.  In this demo they call a placeholder URL that an operator would
replace with the real callback.

Adaptive Card schema: https://adaptivecards.io/
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx

from month_end_assistant.config import get_settings
from month_end_assistant.models import HITLRequest

logger = logging.getLogger(__name__)


class TeamsNotifier:
    """
    Post Adaptive Card messages to a Microsoft Teams channel.

    Usage
    ─────
        notifier = TeamsNotifier()
        success  = await notifier.send(hitl_request)
    """

    # Adaptive Card colour tokens
    _COLOR_APPROVAL = "attention"   # red-ish – human action required
    _COLOR_INFO     = "accent"      # blue – informational

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Public API ───────────────────────────────────────────────────────────

    async def send(self, request: HITLRequest) -> bool:
        """
        Send the HITL notification as an Adaptive Card to Teams.

        Returns True on success, False if the webhook is not configured or
        the HTTP call fails.
        """
        if not self._settings.has_teams:
            logger.warning("Teams webhook not configured – skipping Teams notification.")
            return False

        card_payload = self._build_adaptive_card(request)
        return await self._post_to_webhook(card_payload)

    # ── Card builder ─────────────────────────────────────────────────────────

    def _build_adaptive_card(self, request: HITLRequest) -> Dict[str, Any]:
        """
        Construct the Teams message payload containing an Adaptive Card.

        The card follows the MessageCard → AdaptiveCard pattern so it renders
        in both legacy and modern Teams clients.
        """
        colour     = self._COLOR_APPROVAL if request.requires_approval else self._COLOR_INFO
        deadline   = f"{request.deadline_minutes} minutes" if request.requires_approval else "N/A"

        # Build fact rows from the request context dict
        facts = [
            {"title": k.replace("_", " ").title(), "value": str(v)}
            for k, v in request.context.items()
        ]

        card_body = [
            # ── Header ──────────────────────────────────────────────────────
            {
                "type": "TextBlock",
                "text": f"🔔 {request.title}",
                "weight": "Bolder",
                "size": "Large",
                "color": colour,
                "wrap": True,
            },
            {
                "type": "TextBlock",
                "text": request.summary,
                "wrap": True,
                "spacing": "Medium",
            },
            # ── Fact table ──────────────────────────────────────────────────
            {
                "type": "FactSet",
                "facts": facts + [
                    {"title": "Approval Deadline", "value": deadline},
                    {"title": "Request ID",        "value": request.id},
                ],
                "spacing": "Medium",
            },
            # ── Detail text ─────────────────────────────────────────────────
            {
                "type": "TextBlock",
                "text": request.detail,
                "wrap": True,
                "spacing": "Medium",
                "isSubtle": True,
            },
        ]

        # ── Action buttons (only shown when approval is required) ─────────
        actions = []
        if request.requires_approval:
            base_url = "https://your-assistant.example.com/hitl"
            actions = [
                {
                    "type": "Action.OpenUrl",
                    "title": "✅  Approve",
                    "url": f"{base_url}/approve/{request.id}",
                    "style": "positive",
                },
                {
                    "type": "Action.OpenUrl",
                    "title": "❌  Reject",
                    "url": f"{base_url}/reject/{request.id}",
                    "style": "destructive",
                },
                {
                    "type": "Action.OpenUrl",
                    "title": "⬆️  Escalate",
                    "url": f"{base_url}/escalate/{request.id}",
                },
            ]

        adaptive_card = {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.4",
            "body": card_body,
            "actions": actions,
        }

        # Wrap in a Teams message attachment
        return {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "contentUrl": None,
                    "content": adaptive_card,
                }
            ],
        }

    # ── HTTP dispatch ─────────────────────────────────────────────────────────

    async def _post_to_webhook(self, payload: Dict[str, Any]) -> bool:
        """POST the Adaptive Card payload to the Teams Incoming Webhook URL."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    self._settings.teams_webhook_url,
                    content=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                logger.info("Teams notification sent (request_id=%s)", payload.get("id", "?"))
                return True
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Teams webhook returned HTTP %s: %s",
                exc.response.status_code,
                exc.response.text,
            )
        except httpx.RequestError as exc:
            logger.error("Teams webhook request failed: %s", exc)
        return False
