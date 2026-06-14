"""
Resilience primitives for the agent harness.

  CircuitBreaker     – prevents cascade failures when the LLM service is down
  TokenBudgetManager – enforces per-minute token budgets (avoids throttling)
  RateLimiter        – sliding-window request-per-minute limit per user
  RetryPolicy        – tenacity-based retry with exponential backoff + jitter
  BulkheadSemaphore  – limits concurrent graph runs (bulkhead pattern)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted while the circuit breaker is open."""

class RateLimitError(RuntimeError):
    """Raised when per-minute budget or request limit is exceeded."""

class BulkheadFullError(RuntimeError):
    """Raised when maximum concurrent runs are already in progress."""


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────────────────────────────────────

class _CBState(Enum):
    CLOSED    = "closed"     # normal operation
    OPEN      = "open"       # blocking all calls
    HALF_OPEN = "half_open"  # testing recovery


@dataclass
class CircuitBreaker:
    """
    Three-state circuit breaker.

    Tracks consecutive failures per agent.  When `failure_threshold` is
    reached the circuit opens and all calls are rejected until
    `recovery_timeout_s` elapses.  One probe call is then allowed
    (HALF_OPEN); success closes the circuit, failure re-opens it.

    Usage::
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout_s=30)
        cb.before_call("orchestrator")   # raises CircuitOpenError if open
        try:
            result = await agent.run(...)
            cb.on_success("orchestrator")
        except Exception:
            cb.on_failure("orchestrator")
            raise
    """
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0

    _states: Dict[str, _CBState]       = field(default_factory=dict)
    _failures: Dict[str, int]          = field(default_factory=dict)
    _opened_at: Dict[str, float]       = field(default_factory=dict)
    _on_state_change: Optional[Callable] = field(default=None)

    def _state(self, name: str) -> _CBState:
        return self._states.get(name, _CBState.CLOSED)

    def before_call(self, name: str) -> None:
        state = self._state(name)
        if state == _CBState.CLOSED:
            return
        if state == _CBState.OPEN:
            elapsed = time.monotonic() - self._opened_at.get(name, 0)
            if elapsed >= self.recovery_timeout_s:
                logger.info("CircuitBreaker[%s]: OPEN → HALF_OPEN (probe)", name)
                self._states[name] = _CBState.HALF_OPEN
                return
            remaining = self.recovery_timeout_s - elapsed
            raise CircuitOpenError(
                f"Circuit breaker for '{name}' is OPEN. Retry in {remaining:.0f}s."
            )
        # HALF_OPEN: allow through

    def on_success(self, name: str) -> None:
        prev = self._state(name)
        self._failures[name] = 0
        self._states[name] = _CBState.CLOSED
        if prev != _CBState.CLOSED:
            logger.info("CircuitBreaker[%s]: %s → CLOSED", name, prev.value)
            if self._on_state_change:
                self._on_state_change(name, prev, _CBState.CLOSED)

    def on_failure(self, name: str) -> None:
        count = self._failures.get(name, 0) + 1
        self._failures[name] = count
        if count >= self.failure_threshold:
            self._states[name] = _CBState.OPEN
            self._opened_at[name] = time.monotonic()
            logger.warning("CircuitBreaker[%s]: → OPEN after %d failures", name, count)
            if self._on_state_change:
                self._on_state_change(name, _CBState.CLOSED, _CBState.OPEN)

    def is_open(self, name: str) -> bool:
        return self._state(name) == _CBState.OPEN

    def status(self) -> Dict[str, str]:
        return {name: s.value for name, s in self._states.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Token Budget Manager
# ─────────────────────────────────────────────────────────────────────────────

class TokenBudgetManager:
    """
    Sliding one-minute window token budget.

    Tracks tokens across all agents to avoid hitting Bedrock throttle limits.
    Async-safe via asyncio.Lock.

    Usage::
        budget = TokenBudgetManager(max_tokens_per_minute=100_000)
        await budget.consume(1500)   # raises RateLimitError if over budget
    """

    def __init__(self, max_tokens_per_minute: int = 100_000) -> None:
        self._limit = max_tokens_per_minute
        self._window_start = time.monotonic()
        self._used = 0
        self._lock = asyncio.Lock()
        self._lifetime_tokens = 0

    async def consume(self, tokens: int) -> None:
        async with self._lock:
            now = time.monotonic()
            if now - self._window_start >= 60.0:
                self._used = 0
                self._window_start = now

            if self._used + tokens > self._limit:
                wait_s = 60.0 - (now - self._window_start)
                raise RateLimitError(
                    f"Token budget exceeded ({self._used}/{self._limit} tpm). "
                    f"Retry in {wait_s:.0f}s."
                )
            self._used += tokens
            self._lifetime_tokens += tokens

    @property
    def remaining(self) -> int:
        elapsed = time.monotonic() - self._window_start
        if elapsed >= 60:
            return self._limit
        return max(0, self._limit - self._used)

    @property
    def lifetime_tokens(self) -> int:
        return self._lifetime_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Per-user Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Per-user sliding-window request rate limiter.

    Tracks request counts per user_id in 60-second windows.

    Usage::
        rl = RateLimiter(max_per_minute=10)
        await rl.check("user-42")   # raises RateLimitError if over limit
    """

    def __init__(self, max_per_minute: int = 60) -> None:
        self._limit = max_per_minute
        self._windows: Dict[str, tuple] = {}   # user_id → (window_start, count)
        self._lock = asyncio.Lock()

    async def check(self, user_id: str) -> None:
        async with self._lock:
            now = time.monotonic()
            start, count = self._windows.get(user_id, (now, 0))
            if now - start >= 60.0:
                start, count = now, 0
            if count >= self._limit:
                wait_s = 60.0 - (now - start)
                raise RateLimitError(
                    f"Rate limit for user '{user_id}': {count}/{self._limit} rpm. "
                    f"Retry in {wait_s:.0f}s."
                )
            self._windows[user_id] = (start, count + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Bulkhead Semaphore
# ─────────────────────────────────────────────────────────────────────────────

class BulkheadSemaphore:
    """
    Limits the number of simultaneous agent graph runs.

    Usage::
        bulkhead = BulkheadSemaphore(max_concurrent=10)
        async with bulkhead.acquire("orchestrator"):
            result = await agent.run(...)
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)
        self._max = max_concurrent
        self._current = 0
        self._lock = asyncio.Lock()

    @property
    def current_runs(self) -> int:
        return self._current

    @property
    def available_slots(self) -> int:
        return self._max - self._current

    def acquire(self, agent: str = "unknown"):
        return _BulkheadContext(self, agent)

    async def _acquire(self, agent: str) -> None:
        acquired = self._sem.locked() is False or True  # always try
        if not await asyncio.wait_for(self._sem.acquire(), timeout=0.001) if self._sem.locked() else (self._sem.acquire(), True):
            pass
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=0.01)
        except asyncio.TimeoutError:
            raise BulkheadFullError(
                f"Bulkhead full ({self._max} concurrent runs). Agent '{agent}' rejected."
            )
        async with self._lock:
            self._current += 1

    async def _release(self) -> None:
        self._sem.release()
        async with self._lock:
            self._current = max(0, self._current - 1)


class _BulkheadContext:
    def __init__(self, bulkhead: BulkheadSemaphore, agent: str) -> None:
        self._bh = bulkhead
        self._agent = agent
        self._acquired = False

    async def __aenter__(self):
        # Direct semaphore acquire (simpler than the above)
        if not self._bh._sem._value:  # type: ignore[attr-defined]
            raise BulkheadFullError(
                f"Bulkhead full — max {self._bh._max} concurrent runs active."
            )
        await self._bh._sem.acquire()
        async with self._bh._lock:
            self._bh._current += 1
        self._acquired = True
        return self

    async def __aexit__(self, *args):
        if self._acquired:
            self._bh._sem.release()
            async with self._bh._lock:
                self._bh._current = max(0, self._bh._current - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Retry Policy
# ─────────────────────────────────────────────────────────────────────────────

class RetryPolicy:
    """
    Pre-configured tenacity retry strategies for different failure modes.

    Usage::
        async for attempt in RetryPolicy.llm_call():
            with attempt:
                result = await llm.ainvoke(messages)
    """

    @staticmethod
    def llm_call(max_attempts: int = 3):
        """Retry on transient LLM/network errors with exponential backoff."""
        return AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    @staticmethod
    def tool_call(max_attempts: int = 2):
        """Retry tool calls once on any exception."""
        return AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    @staticmethod
    def aws_api(max_attempts: int = 4):
        """Retry AWS API calls with longer backoff (handles throttling)."""
        return AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=2, min=5, max=60),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
