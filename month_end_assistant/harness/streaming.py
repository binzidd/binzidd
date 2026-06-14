"""
Full astream_events integration with multi-consumer fan-out.

Features:
  - Wraps LangGraph's `graph.astream_events(..., version="v2")`
  - Fans out every event to N async subscribers via asyncio.Queue
  - Formats events as SSE (text/event-stream) or structured dicts
  - Filters by event type to reduce noise
  - Tracks token deltas for real-time streaming to OpenWebUI
  - Provides a WebSocket broadcast helper

Event taxonomy (LangGraph v2 events):
  on_chain_start / on_chain_end / on_chain_stream
  on_llm_start   / on_llm_end   / on_llm_stream
  on_tool_start  / on_tool_end
  on_retriever_start / on_retriever_end
  on_custom_event  (user-emitted via astream_events dispatch)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

SENTINEL = object()  # marks end-of-stream for queue consumers


@dataclass
class HarnessEvent:
    """Normalised harness event emitted to all subscribers."""
    run_id: str
    event_type: str           # "token" | "tool_call" | "tool_result" | "node" | "done" | "error"
    agent: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.monotonic)

    def to_sse(self) -> str:
        """Format as SSE line for HTTP streaming."""
        return f"data: {json.dumps(self.__dict__, default=str)}\n\n"

    def to_openai_delta(self) -> Optional[str]:
        """Format as OpenAI-compatible streaming chunk."""
        if self.event_type != "token":
            return None
        chunk = {
            "id": f"chatcmpl-{self.run_id[:8]}",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": self.data.get("token", "")},
                "finish_reason": None,
            }],
        }
        return f"data: {json.dumps(chunk)}\n\n"


class HarnessStreamingManager:
    """
    Multi-consumer event fan-out for all agent graph runs.

    Each run has a `run_id`.  Callers subscribe to a run_id and receive
    HarnessEvent objects via an asyncio.Queue.  When the run ends the
    manager puts SENTINEL in every subscriber queue.

    Usage::
        manager = HarnessStreamingManager(settings)

        # Producer side (inside harness.run()):
        async for event in manager.stream_graph(graph, state, config, run_id, agent):
            ...  # already broadcast to all subscribers

        # Consumer side (FastAPI SSE endpoint):
        q = manager.subscribe(run_id)
        try:
            while True:
                event = await q.get()
                if event is SENTINEL:
                    break
                yield event.to_sse()
        finally:
            manager.unsubscribe(run_id, q)
    """

    def __init__(self, include_types: Optional[List[str]] = None) -> None:
        self._subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._include_types: Set[str] = set(include_types or [
            "on_llm_stream", "on_chain_end", "on_tool_end", "on_custom_event",
        ])

    # ── Producer: stream a graph and fan-out events ───────────────────────────

    async def stream_graph(
        self,
        graph: Any,                      # CompiledGraph
        input_state: Dict[str, Any],
        config: Dict[str, Any],
        run_id: str,
        agent: str = "unknown",
        extra_include_types: Optional[List[str]] = None,
    ) -> AsyncIterator[HarnessEvent]:
        """
        Invoke graph.astream_events and yield normalised HarnessEvent objects.

        Every event is also broadcast to all active subscribers for this run_id.
        """
        include = self._include_types | set(extra_include_types or [])

        try:
            async for raw in graph.astream_events(input_state, config=config, version="v2"):
                event_name: str = raw.get("event", "")
                if event_name not in include:
                    continue

                harness_event = self._normalise(raw, run_id, agent)
                if harness_event:
                    await self._broadcast(run_id, harness_event)
                    yield harness_event

            # Done sentinel
            done_event = HarnessEvent(
                run_id=run_id, event_type="done", agent=agent, data={"message": "Run complete"},
            )
            await self._broadcast(run_id, done_event)
            yield done_event

        except Exception as exc:
            error_event = HarnessEvent(
                run_id=run_id, event_type="error", agent=agent, data={"error": str(exc)},
            )
            await self._broadcast(run_id, error_event)
            yield error_event
            raise
        finally:
            self._close_subscribers(run_id)

    # ── Consumer: subscribe / unsubscribe ─────────────────────────────────────

    def subscribe(self, run_id: str, maxsize: int = 500) -> asyncio.Queue:
        """Register a consumer queue for a specific run."""
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers[run_id].append(q)
        logger.debug("New subscriber for run %s (total=%d)", run_id, len(self._subscribers[run_id]))
        return q

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        """Deregister a consumer queue."""
        try:
            self._subscribers[run_id].remove(queue)
        except ValueError:
            pass
        if not self._subscribers[run_id]:
            del self._subscribers[run_id]

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _broadcast(self, run_id: str, event: HarnessEvent) -> None:
        for q in list(self._subscribers.get(run_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full for run %s — dropping event", run_id)

    def _close_subscribers(self, run_id: str) -> None:
        for q in list(self._subscribers.get(run_id, [])):
            try:
                q.put_nowait(SENTINEL)
            except asyncio.QueueFull:
                pass

    def _normalise(self, raw: Dict[str, Any], run_id: str, agent: str) -> Optional[HarnessEvent]:
        """Convert a raw LangGraph astream_events event to a HarnessEvent."""
        event_name: str = raw.get("event", "")
        data = raw.get("data", {})
        name = raw.get("name", "")

        if event_name == "on_llm_stream":
            chunk = data.get("chunk", {})
            # Handle both AIMessageChunk and plain dict
            content = ""
            if hasattr(chunk, "content"):
                content = chunk.content
            elif isinstance(chunk, dict):
                content = chunk.get("content", "")
            if not content:
                return None
            return HarnessEvent(
                run_id=run_id, event_type="token", agent=agent,
                data={"token": content, "node": name},
            )

        if event_name == "on_tool_end":
            return HarnessEvent(
                run_id=run_id, event_type="tool_result", agent=agent,
                data={"tool": name, "output": str(data.get("output", ""))[:500]},
            )

        if event_name == "on_chain_end":
            return HarnessEvent(
                run_id=run_id, event_type="node", agent=agent,
                data={"node": name, "output_keys": list(data.get("output", {}).keys()) if isinstance(data.get("output"), dict) else []},
            )

        if event_name == "on_custom_event":
            return HarnessEvent(
                run_id=run_id, event_type="custom", agent=agent,
                data={"name": name, **data},
            )

        return None

    @staticmethod
    def format_sse_done() -> str:
        return "data: [DONE]\n\n"

    @staticmethod
    async def consume_to_list(
        queue: asyncio.Queue,
        timeout_s: float = 120.0,
    ) -> List[HarnessEvent]:
        """Drain a queue to a list (useful in tests)."""
        events: List[HarnessEvent] = []
        deadline = asyncio.get_event_loop().time() + timeout_s
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=remaining)
                if item is SENTINEL:
                    break
                events.append(item)
            except asyncio.TimeoutError:
                break
        return events
