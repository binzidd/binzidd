"""
Session Memory Manager backed by AWS AgentCore Memory.

AWS AgentCore Memory (bedrock-agentcore namespace) provides a managed,
vector-enabled memory store for agents.  Each user gets a dedicated
"memory session" so the assistant remembers preferences, prior period
results, and custom rules across month-end cycles.

Key AgentCore Memory operations used here
──────────────────────────────────────────
  ingest_conversations   – save a new conversation turn to long-term memory
  retrieve_memories      – semantic search over the user's memory store
  list_sessions          – list sessions for a user
  create_session         – start a new memory session
  delete_session         – purge a session (GDPR / data-retention)

When AgentCore credentials are not configured the manager falls back to
an in-process dict store so the rest of the application keeps working.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from month_end_assistant.config import get_settings
from month_end_assistant.models import MonthEndPeriod, UserSession

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages per-user sessions with AWS AgentCore Memory as the backend.

    Usage
    ─────
        mgr = SessionManager()
        session = await mgr.load_or_create("alice", "acme-corp")
        await mgr.save_memory(session, "Revenue spiked in March due to new contract")
        memories = await mgr.retrieve_memories(session, "revenue trends")
    """

    # Fallback in-process store (used when AgentCore is not configured)
    _local_store: Dict[str, dict] = {}

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: Optional[Any] = self._build_client()

    # ── AWS client construction ───────────────────────────────────────────────

    def _build_client(self) -> Optional[Any]:
        """
        Build the boto3 AgentCore Memory client.

        Returns None (and logs a warning) when credentials or the memory ID
        are missing, activating the local fallback store.
        """
        if not self._settings.has_agentcore:
            logger.warning(
                "AGENTCORE_MEMORY_ID not set – falling back to in-process memory."
            )
            return None
        try:
            return boto3.client(
                "bedrock-agentcore",            # AgentCore service endpoint
                region_name=self._settings.aws_region,
                aws_access_key_id=self._settings.aws_access_key_id or None,
                aws_secret_access_key=self._settings.aws_secret_access_key or None,
            )
        except Exception as exc:
            logger.error("Failed to create AgentCore client: %s", exc)
            return None

    # ── Public API ───────────────────────────────────────────────────────────

    async def load_or_create(self, user_id: str, company_id: str) -> UserSession:
        """
        Load an existing session for *user_id* or create a fresh one.

        The session stores the LangGraph thread_id so interrupted graphs
        can be resumed across HTTP requests or CLI invocations.
        """
        stored = await self._fetch_session_record(user_id)
        if stored:
            session = UserSession(**stored)
            session.last_active = datetime.utcnow()
            logger.info("Loaded existing session for user=%s", user_id)
        else:
            session = UserSession(user_id=user_id, company_id=company_id)
            logger.info("Created new session for user=%s session_id=%s",
                        user_id, session.session_id)

        # Persist the (possibly refreshed) session
        await self._persist_session_record(session)
        return session

    async def save_memory(self, session: UserSession, content: str) -> None:
        """
        Ingest a free-text memory fragment into the user's AgentCore store.

        Example fragments:
          "Revenue for March 2025 was $4.2 M, 8% above budget."
          "User prefers variance threshold of 3% not 5%."
        """
        if self._client:
            await self._agentcore_ingest(session, content)
        else:
            key = f"mem:{session.user_id}"
            existing = self._local_store.get(key, [])
            existing.append({"content": content, "ts": datetime.utcnow().isoformat()})
            self._local_store[key] = existing
            logger.debug("Local memory saved for user=%s", session.user_id)

    async def retrieve_memories(
        self, session: UserSession, query: str, top_k: int = 5
    ) -> List[str]:
        """
        Semantically retrieve the *top_k* most relevant memories for *query*.

        Returns a list of plain-text memory fragments ordered by relevance.
        """
        if self._client:
            return await self._agentcore_retrieve(session, query, top_k)

        # Local fallback – return all stored fragments (no semantic ranking)
        fragments = self._local_store.get(f"mem:{session.user_id}", [])
        return [f["content"] for f in fragments[-top_k:]]

    async def update_active_period(
        self, session: UserSession, period: MonthEndPeriod
    ) -> UserSession:
        """Set the period the user is currently working on and persist it."""
        session.active_period = period
        session.last_active = datetime.utcnow()
        await self._persist_session_record(session)
        return session

    async def append_report_id(
        self, session: UserSession, report_id: str
    ) -> UserSession:
        """Record that a report was generated in this session (for history)."""
        if report_id not in session.prior_reports:
            session.prior_reports.append(report_id)
        await self._persist_session_record(session)
        return session

    async def delete_session(self, user_id: str) -> None:
        """Purge all session data for *user_id* (GDPR / data-retention)."""
        if self._client:
            await self._agentcore_delete_session(user_id)
        self._local_store.pop(f"session:{user_id}", None)
        self._local_store.pop(f"mem:{user_id}", None)
        logger.info("Deleted session data for user=%s", user_id)

    # ── Private helpers – AgentCore Memory API calls ─────────────────────────

    async def _agentcore_ingest(
        self, session: UserSession, content: str
    ) -> None:
        """
        Call AgentCore Memory → IngestConversations to persist a memory.

        The conversation payload wraps the content as a single assistant turn
        so it is indexed with metadata (user_id, session_id).
        """
        try:
            self._client.ingest_conversations(
                memoryId=self._settings.agentcore_memory_id,
                conversations=[
                    {
                        "role": "ASSISTANT",
                        "content": [{"text": content}],
                    }
                ],
                memoryStrategies=[
                    {
                        "semanticMemoryStrategy": {
                            "name": f"user-{session.user_id}",
                        }
                    }
                ],
            )
            logger.debug("AgentCore memory ingested for user=%s", session.user_id)
        except (BotoCoreError, ClientError) as exc:
            logger.error("AgentCore ingest failed: %s – storing locally", exc)
            key = f"mem:{session.user_id}"
            existing = self._local_store.get(key, [])
            existing.append({"content": content, "ts": datetime.utcnow().isoformat()})
            self._local_store[key] = existing

    async def _agentcore_retrieve(
        self, session: UserSession, query: str, top_k: int
    ) -> List[str]:
        """
        Call AgentCore Memory → RetrieveMemories for semantic search.
        """
        try:
            response = self._client.retrieve_memories(
                memoryId=self._settings.agentcore_memory_id,
                query=query,
                maxResults=top_k,
            )
            return [
                mem["content"]["text"]
                for mem in response.get("memories", [])
                if "content" in mem
            ]
        except (BotoCoreError, ClientError) as exc:
            logger.error("AgentCore retrieve failed: %s", exc)
            return []

    async def _agentcore_delete_session(self, user_id: str) -> None:
        """Delete the AgentCore Memory session for a user."""
        try:
            self._client.delete_memory(
                memoryId=self._settings.agentcore_memory_id,
                clientToken=user_id,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.error("AgentCore delete failed: %s", exc)

    # ── Private helpers – local session record persistence ───────────────────

    async def _fetch_session_record(self, user_id: str) -> Optional[dict]:
        """Retrieve the raw session dict from the local store."""
        return self._local_store.get(f"session:{user_id}")

    async def _persist_session_record(self, session: UserSession) -> None:
        """Serialise and save the session to the local store."""
        self._local_store[f"session:{session.user_id}"] = json.loads(
            session.model_dump_json()
        )
