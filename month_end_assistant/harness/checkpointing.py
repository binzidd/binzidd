"""
Production-grade checkpointing for LangGraph graphs.

Provides a factory that tries backends in preference order:
  1. AsyncSqliteSaver  (langgraph-checkpoint-sqlite – lightweight, zero-infra)
  2. AsyncPostgresSaver (langgraph-checkpoint-postgres – production, team sharing)
  3. MemorySaver        (fallback, no persistence across restarts)

All graph compilations should call `CheckpointerFactory.create()` rather than
hard-coding `MemorySaver()`, so the backend can be swapped via env vars.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from langgraph.checkpoint.memory import MemorySaver

from month_end_assistant.harness.config import HarnessSettings

logger = logging.getLogger(__name__)

# Optional imports — not all deployments will have these packages installed
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    _HAS_SQLITE = True
except ImportError:
    _HAS_SQLITE = False

try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    _HAS_POSTGRES = True
except ImportError:
    _HAS_POSTGRES = False


class CheckpointerFactory:
    """Creates the appropriate checkpointer for the configured backend."""

    @staticmethod
    @asynccontextmanager
    async def create(settings: HarnessSettings) -> AsyncIterator:
        """
        Async context manager that yields a ready-to-use checkpointer.

        Usage::
            async with CheckpointerFactory.create(settings) as checkpointer:
                graph = my_graph.compile(checkpointer=checkpointer)
        """
        backend = settings.checkpoint_backend

        if backend == "postgres" and _HAS_POSTGRES and settings.postgres_dsn:
            logger.info("Checkpointer → AsyncPostgresSaver (%s)", _mask_dsn(settings.postgres_dsn))
            async with AsyncPostgresSaver.from_conn_string(settings.postgres_dsn) as cp:
                await cp.setup()
                yield cp

        elif backend in ("sqlite", "postgres") and _HAS_SQLITE:
            logger.info("Checkpointer → AsyncSqliteSaver (%s)", settings.sqlite_db_path)
            async with AsyncSqliteSaver.from_conn_string(settings.sqlite_db_path) as cp:
                yield cp

        else:
            logger.warning(
                "Falling back to MemorySaver (backend=%s, sqlite=%s, postgres=%s). "
                "State will NOT persist across restarts.",
                backend, _HAS_SQLITE, _HAS_POSTGRES,
            )
            yield MemorySaver()

    @staticmethod
    def create_sync(settings: HarnessSettings):
        """Synchronous fallback used in non-async contexts (tests, CLI)."""
        return MemorySaver()


def _mask_dsn(dsn: str) -> str:
    """Hide credentials in log output."""
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(dsn)
        masked = p._replace(netloc=f"***:***@{p.hostname}:{p.port}")
        return urlunparse(masked)
    except Exception:
        return "<dsn>"
