"""
Slack HITL Notifier.

Posts a richly-formatted Block Kit message to a Slack channel using the
Slack Web API (chat.postMessage).  The message includes:

  • A header section with colour-coded context
  • A two-column field grid showing key metrics
  • An overflow divider and detail text
  • Interactive buttons (Approve / Reject / Escalate) via Block Kit actions

Button clicks post back to the assistant's /slack/actions webhook endpoint
(replace the placeholder action_id values with your Slack app's handlers).

Slack Block Kit reference: https://api.slack.com/block-kit
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from month_end_assistant.config import get_settings
from month_end_assistant.models import HITLRequest

logger = logging.getLogger(__name__)

# Slack Web API endpoint
_SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"


class SlackNotifier:
    """
    Post Block Kit messages to a Slack channel via the Slack Web API.

    Usage
    ─────
        notifier = SlackNotifier()
        success  = await notifier.send(hitl_request)
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Public API ───────────────────────────────────────────────────────────

    async def send(self, request: HITLRequest) -> bool:
        """
        Send the HITL notification to Slack.

        Returns True on success, False if the token is not configured or
        the API call fails.
        """
        if not self._settings.has_slack:
            logger.warning("Slack bot token not configured – skipping Slack notification.")
            return False

        blocks  = self._build_blocks(request)
        payload = {
            "channel": self._settings.slack_approval_channel,
            "text":    request.title,   # fallback for notifications
            "blocks":  blocks,
        }
        return await self._call_api(payload)

    # ── Block Kit builder ────────────────────────────────────────────────────

    def _build_blocks(self, request: HITLRequest) -> List[Dict[str, Any]]:
        """
        Build the list of Slack Block Kit blocks for the notification.

        Structure:
          [Header] [Context line] [Divider] [Fields] [Divider] [Detail] [Actions]
        """
        # Emoji & colour hint embedded in the header text
        icon   = "🔴" if request.requires_approval else "🟡"
        blocks: List[Dict[str, Any]] = [
            # ── Header ──────────────────────────────────────────────────────
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{icon}  {request.title}",
                    "emoji": True,
                },
            },
            # ── Context metadata ────────────────────────────────────────────
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"*Request ID:* `{request.id}`  |  "
                            f"*Deadline:* {request.deadline_minutes} min  |  "
                            f"*Created:* {request.created_at.strftime('%Y-%m-%d %H:%M UTC')}"
                        ),
                    }
                ],
            },
            {"type": "divider"},
            # ── Summary paragraph ────────────────────────────────────────────
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Summary*\n{request.summary}"},
            },
        ]

        # ── Context key-value fields (rendered as a two-column grid) ─────────
        fields = [
            {"type": "mrkdwn", "text": f"*{k.replace('_', ' ').title()}*\n{v}"}
            for k, v in request.context.items()
        ]
        # Slack renders fields in pairs; split into chunks of 10 max per block
        for i in range(0, len(fields), 10):
            blocks.append({
                "type": "section",
                "fields": fields[i : i + 10],
            })

        # ── Detail text ───────────────────────────────────────────────────────
        blocks += [
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Details*\n{request.detail}"},
            },
        ]

        # ── Action buttons ────────────────────────────────────────────────────
        if request.requires_approval:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅  Approve", "emoji": True},
                        "style": "primary",
                        "action_id": f"hitl_approve_{request.id}",
                        "value": request.id,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌  Reject", "emoji": True},
                        "style": "danger",
                        "action_id": f"hitl_reject_{request.id}",
                        "value": request.id,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "⬆️  Escalate", "emoji": True},
                        "action_id": f"hitl_escalate_{request.id}",
                        "value": request.id,
                    },
                ],
            })

        return blocks

    # ── Slack API call ────────────────────────────────────────────────────────

    async def _call_api(self, payload: Dict[str, Any]) -> bool:
        """POST the Block Kit payload to Slack's chat.postMessage endpoint."""
        headers = {
            "Authorization": f"Bearer {self._settings.slack_bot_token}",
            "Content-Type":  "application/json; charset=utf-8",
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    _SLACK_POST_MESSAGE_URL,
                    content=json.dumps(payload),
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                if not data.get("ok"):
                    logger.error("Slack API error: %s", data.get("error", "unknown"))
                    return False
                logger.info(
                    "Slack notification sent to %s (ts=%s)",
                    payload.get("channel"),
                    data.get("ts"),
                )
                return True
        except httpx.HTTPStatusError as exc:
            logger.error("Slack API returned HTTP %s", exc.response.status_code)
        except httpx.RequestError as exc:
            logger.error("Slack API request failed: %s", exc)
        return False
