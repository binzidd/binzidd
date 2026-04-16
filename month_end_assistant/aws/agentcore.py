"""
AWS AgentCore Client.

AWS AgentCore (announced AWS Summit 2025) is a managed runtime for deploying,
executing, and observing AI agents at scale.  This module wraps three of its
key capabilities:

  1. AgentCore Runtime   – invoke a deployed agent and stream its response
  2. AgentCore Memory    – managed long-term memory store (semantic search)
  3. AgentCore Tools     – managed browser / code-interpreter / file-system

boto3 service namespace: ``bedrock-agentcore``
boto3 runtime  namespace: ``bedrock-agent-runtime``   (for InvokeAgent)

When credentials are absent or the service returns errors, every method
falls back gracefully so the rest of the application keeps running.

Reference
─────────
  https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html
  https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeAgent.html
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from month_end_assistant.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentInvocationResult:
    """Aggregated result from an AgentCore Runtime invocation."""

    session_id:    str
    completion:    str                    = ""
    trace_events:  List[Dict[str, Any]]  = field(default_factory=list)
    citations:     List[Dict[str, Any]]  = field(default_factory=list)
    success:       bool                  = True
    error_message: str                   = ""


@dataclass
class MemoryFragment:
    """A single memory item retrieved from AgentCore Memory."""

    id:         str
    content:    str
    score:      float          = 0.0
    metadata:   Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Main client
# ─────────────────────────────────────────────────────────────────────────────

class AgentCoreClient:
    """
    Unified client for AWS AgentCore Runtime and Memory.

    Instantiate once at application startup and share across request handlers.

    Example
    ───────
        client  = AgentCoreClient()
        result  = await client.invoke_agent(
            input_text="Summarise March 2025 revenue variances",
            session_id="user-alice-thread-001",
        )
        print(result.completion)

        memories = await client.retrieve_memories(
            query="revenue trends", user_id="alice", top_k=5
        )
    """

    def __init__(self) -> None:
        self._settings        = get_settings()
        self._runtime_client  = self._build_runtime_client()
        self._memory_client   = self._build_memory_client()

    # ── Client factory helpers ────────────────────────────────────────────────

    def _build_runtime_client(self) -> Optional[Any]:
        """
        Build the bedrock-agent-runtime boto3 client used for InvokeAgent.
        """
        try:
            return boto3.client(
                "bedrock-agent-runtime",
                region_name=self._settings.aws_region,
                aws_access_key_id=self._settings.aws_access_key_id or None,
                aws_secret_access_key=self._settings.aws_secret_access_key or None,
            )
        except Exception as exc:
            logger.warning("Could not build AgentCore runtime client: %s", exc)
            return None

    def _build_memory_client(self) -> Optional[Any]:
        """
        Build the bedrock-agentcore boto3 client used for Memory operations.

        This targets the new bedrock-agentcore service endpoint introduced in
        the AWS SDK alongside the AgentCore launch (SDK v1.35+).
        """
        if not self._settings.has_agentcore:
            logger.info("AgentCore Memory not configured – memory calls are no-ops.")
            return None
        try:
            return boto3.client(
                "bedrock-agentcore",
                region_name=self._settings.aws_region,
                aws_access_key_id=self._settings.aws_access_key_id or None,
                aws_secret_access_key=self._settings.aws_secret_access_key or None,
            )
        except Exception as exc:
            logger.warning("Could not build AgentCore memory client: %s", exc)
            return None

    # ── AgentCore Runtime: InvokeAgent ────────────────────────────────────────

    async def invoke_agent(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        enable_trace: bool = True,
    ) -> AgentInvocationResult:
        """
        Invoke the deployed AgentCore agent and collect its full response.

        AgentCore streams the response as a series of EventStream chunks.
        This method accumulates them into a single AgentInvocationResult.

        Args:
            input_text:   The user query or task description.
            session_id:   Conversation thread ID (AgentCore maintains context).
            enable_trace: Include reasoning trace events in the result.

        Returns:
            AgentInvocationResult with the agent's completion text and traces.
        """
        if not self._runtime_client:
            logger.warning("Runtime client unavailable – returning stub response.")
            return AgentInvocationResult(
                session_id=session_id or str(uuid.uuid4()),
                completion=f"[AgentCore stub] Processed: {input_text}",
                success=True,
            )

        session_id = session_id or str(uuid.uuid4())

        try:
            response = self._runtime_client.invoke_agent(
                agentId=self._settings.agentcore_agent_id,
                agentAliasId=self._settings.agentcore_agent_alias_id,
                sessionId=session_id,
                inputText=input_text,
                enableTrace=enable_trace,
            )
            return self._consume_event_stream(response, session_id)
        except (BotoCoreError, ClientError) as exc:
            logger.error("AgentCore InvokeAgent failed: %s", exc)
            return AgentInvocationResult(
                session_id=session_id,
                success=False,
                error_message=str(exc),
            )

    def _consume_event_stream(
        self, response: Any, session_id: str
    ) -> AgentInvocationResult:
        """
        Consume the InvokeAgent EventStream and build an AgentInvocationResult.

        The stream emits three main event types:
          • chunk          – a piece of the final completion text
          • trace          – reasoning / tool-use trace events
          • returnControl  – agent is waiting for a function result
        """
        result = AgentInvocationResult(session_id=session_id)
        try:
            for event in response.get("completion", []):
                if "chunk" in event:
                    result.completion += event["chunk"].get("bytes", b"").decode("utf-8")
                elif "trace" in event:
                    result.trace_events.append(event["trace"])
                elif "returnControl" in event:
                    # Agent needs external tool result – handle in orchestrator
                    result.trace_events.append({"returnControl": event["returnControl"]})
        except Exception as exc:
            logger.error("Error consuming AgentCore event stream: %s", exc)
            result.success = False
            result.error_message = str(exc)
        return result

    # ── AgentCore Memory ──────────────────────────────────────────────────────

    async def store_memory(
        self,
        user_id: str,
        content: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Persist a memory fragment in the AgentCore Memory store.

        Uses the IngestConversations API to create a semantic memory that
        will be retrievable by future queries from the same user.

        Args:
            user_id:    Unique user identifier (used as memory namespace).
            content:    Plain-text content to remember.
            session_id: Optional session context for the memory.

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self._memory_client:
            return False
        try:
            self._memory_client.ingest_conversations(
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
                            "name": f"user-{user_id}",
                        }
                    }
                ],
                **({"sessionId": session_id} if session_id else {}),
            )
            logger.debug("AgentCore memory stored for user=%s", user_id)
            return True
        except (BotoCoreError, ClientError) as exc:
            logger.error("AgentCore store_memory failed: %s", exc)
            return False

    async def retrieve_memories(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> List[MemoryFragment]:
        """
        Semantically retrieve the most relevant memories for *query*.

        Uses AgentCore Memory's built-in vector search so the agent always
        has access to historical context without blowing the context window.

        Args:
            query:   Natural-language search query.
            user_id: Limit results to this user's memory namespace.
            top_k:   Maximum number of results to return.

        Returns:
            Ordered list of MemoryFragment objects (most relevant first).
        """
        if not self._memory_client:
            return []
        try:
            response = self._memory_client.retrieve_memories(
                memoryId=self._settings.agentcore_memory_id,
                query=query,
                maxResults=top_k,
                filter={"equals": {"key": "user_id", "value": user_id}},
            )
            return [
                MemoryFragment(
                    id=mem.get("memoryId", ""),
                    content=mem.get("content", {}).get("text", ""),
                    score=mem.get("score", 0.0),
                    metadata=mem.get("metadata", {}),
                )
                for mem in response.get("memories", [])
            ]
        except (BotoCoreError, ClientError) as exc:
            logger.error("AgentCore retrieve_memories failed: %s", exc)
            return []

    async def list_memory_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all memory sessions for a user.

        Useful for the session-load feature – shows the user which previous
        month-end close sessions are stored in AgentCore Memory.
        """
        if not self._memory_client:
            return []
        try:
            response = self._memory_client.list_sessions(
                memoryId=self._settings.agentcore_memory_id,
                filter={"equals": {"key": "user_id", "value": user_id}},
            )
            return response.get("sessions", [])
        except (BotoCoreError, ClientError) as exc:
            logger.error("AgentCore list_sessions failed: %s", exc)
            return []

    async def delete_memory_session(self, session_id: str) -> bool:
        """Delete a specific memory session (e.g. for GDPR compliance)."""
        if not self._memory_client:
            return False
        try:
            self._memory_client.delete_session(
                memoryId=self._settings.agentcore_memory_id,
                sessionId=session_id,
            )
            return True
        except (BotoCoreError, ClientError) as exc:
            logger.error("AgentCore delete_session failed: %s", exc)
            return False

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, bool]:
        """Return the availability status of each AgentCore sub-service."""
        return {
            "runtime_client":  self._runtime_client is not None,
            "memory_client":   self._memory_client is not None,
            "agent_configured": bool(self._settings.agentcore_agent_id),
            "memory_configured": bool(self._settings.agentcore_memory_id),
        }
