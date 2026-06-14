"""
Cross-thread long-term memory store.

LangGraph graphs use per-thread state (checkpointer) for short-term memory.
This module provides cross-thread, long-term memory using LangGraph's Store API
(`InMemoryStore` for dev/testing, swappable to AsyncPostgresStore in prod).

Namespace design:
    ("user",   <user_id>,  "memories")   – per-user semantic memories
    ("user",   <user_id>,  "preferences")– user preferences / config overrides
    ("global", "standards", "accounting")– shared accounting standards knowledge
    ("agent",  <agent_name>, "state")    – agent-level shared state

Usage::
    store = HarnessMemoryStore()
    await store.save(("user", "u1", "memories"), "run-abc", {"summary": "..."})
    hits = await store.search(("user", "u1", "memories"), query="revenue recognition")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

Namespace = Tuple[str, ...]

# ── Optional: LangGraph InMemoryStore ─────────────────────────────────────────
try:
    from langgraph.store.memory import InMemoryStore as _LGInMemoryStore
    _HAS_LG_STORE = True
except ImportError:
    _HAS_LG_STORE = False


@dataclass
class MemoryItem:
    key: str
    value: Dict[str, Any]
    namespace: Namespace
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    score: float = 1.0  # relevance score when returned by search


class HarnessMemoryStore:
    """
    Cross-thread memory store with semantic search.

    When `langgraph.store.memory.InMemoryStore` is available it delegates to
    the native LangGraph store so memories flow directly into graph nodes via
    the `store` parameter of `create_react_agent`.

    Otherwise falls back to a simple dict with substring search — same API,
    lower fidelity retrieval.
    """

    def __init__(self) -> None:
        if _HAS_LG_STORE:
            self._backend = _LGInMemoryStore()
            self._use_lg = True
            logger.info("HarnessMemoryStore → LangGraph InMemoryStore")
        else:
            self._backend = {}          # namespace → {key: MemoryItem}
            self._use_lg = False
            logger.info("HarnessMemoryStore → dict fallback (no langgraph.store available)")

    # ── Core operations ───────────────────────────────────────────────────────

    async def save(self, namespace: Namespace, key: str, value: Dict[str, Any]) -> None:
        """Upsert a memory item."""
        if self._use_lg:
            self._backend.put(namespace, key, value)
        else:
            ns_key = namespace
            self._backend.setdefault(ns_key, {})[key] = MemoryItem(
                key=key, value=value, namespace=namespace
            )
        logger.debug("MemoryStore.save: ns=%s key=%s", namespace, key)

    async def get(self, namespace: Namespace, key: str) -> Optional[Dict[str, Any]]:
        """Fetch a single item by exact key."""
        if self._use_lg:
            item = self._backend.get(namespace, key)
            return item.value if item else None
        item = self._backend.get(namespace, {}).get(key)
        return item.value if item else None

    async def delete(self, namespace: Namespace, key: str) -> None:
        """Remove an item."""
        if self._use_lg:
            self._backend.delete(namespace, key)
        else:
            self._backend.get(namespace, {}).pop(key, None)

    async def search(
        self,
        namespace: Namespace,
        query: str,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """
        Semantic search within a namespace.

        LangGraph's InMemoryStore provides cosine-similarity search when
        embeddings are configured; the fallback does case-insensitive substring
        matching across serialised values.
        """
        if self._use_lg:
            try:
                results = self._backend.search(namespace, query=query, limit=limit)
                return [
                    MemoryItem(
                        key=r.key,
                        value=r.value,
                        namespace=namespace,
                        score=getattr(r, "score", 1.0),
                    )
                    for r in results
                ]
            except Exception as exc:
                logger.debug("LG store search failed (%s) – using fallback", exc)

        # Dict fallback: substring match on serialised JSON
        items = list(self._backend.get(namespace, {}).values())
        q = query.lower()
        scored = []
        for item in items:
            text = json.dumps(item.value).lower()
            if q in text:
                scored.append(MemoryItem(
                    key=item.key,
                    value=item.value,
                    namespace=namespace,
                    score=text.count(q) / (len(text) + 1),
                ))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    async def list_keys(self, namespace: Namespace) -> List[str]:
        """List all keys in a namespace."""
        if self._use_lg:
            items = self._backend.search(namespace, query="", limit=1000)
            return [i.key for i in items]
        return list(self._backend.get(namespace, {}).keys())

    # ── Convenience: per-user memory ─────────────────────────────────────────

    async def save_user_memory(self, user_id: str, key: str, value: Dict[str, Any]) -> None:
        await self.save(("user", user_id, "memories"), key, value)

    async def search_user_memories(self, user_id: str, query: str, limit: int = 5) -> List[MemoryItem]:
        return await self.search(("user", user_id, "memories"), query, limit)

    async def save_user_preference(self, user_id: str, preference: str, value: Any) -> None:
        await self.save(("user", user_id, "preferences"), preference, {"value": value})

    async def get_user_preference(self, user_id: str, preference: str, default: Any = None) -> Any:
        item = await self.get(("user", user_id, "preferences"), preference)
        return item["value"] if item else default

    # ── Convenience: shared knowledge ─────────────────────────────────────────

    async def save_global(self, category: str, key: str, value: Dict[str, Any]) -> None:
        await self.save(("global", category), key, value)

    async def search_global(self, category: str, query: str, limit: int = 5) -> List[MemoryItem]:
        return await self.search(("global", category), query, limit)

    def get_native_store(self) -> Optional[Any]:
        """Return the native LangGraph store for direct injection into agents."""
        return self._backend if self._use_lg else None
