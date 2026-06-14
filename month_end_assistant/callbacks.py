"""
Custom LangChain Callback Handlers.

LangChain callbacks fire at every step of a chain or agent run, making
them the right hook for:
  • Real-time streaming to the OpenWebUI frontend
  • Structured logging / LangSmith-style tracing
  • Cost / token accounting
  • Progress notifications to the HITL manager

Three handlers are provided:

  RichConsoleCallback    – pretty-prints every step to the terminal
  TokenStreamingCallback – accumulates streamed tokens and emits them
                          to an asyncio.Queue for SSE fan-out
  TracingCallback        – records a full execution trace (timings,
                          inputs, outputs) for observability / audit
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
from rich.console import Console
from rich.markup import escape

logger = logging.getLogger(__name__)
_console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rich Console Callback  – pretty terminal output during development
# ─────────────────────────────────────────────────────────────────────────────

class RichConsoleCallback(BaseCallbackHandler):
    """
    Prints each chain / agent step in colour using Rich.

    Attach to any runnable:
        chain.invoke(input, config={"callbacks": [RichConsoleCallback()]})
    """

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        model = serialized.get("name", "LLM")
        _console.print(f"\n[dim cyan]▶ LLM call:[/dim cyan] [bold]{escape(model)}[/bold]")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        _console.print(token, end="", highlight=False)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = response.llm_output.get("usage", {}) if response.llm_output else {}
        if usage:
            _console.print(
                f"\n[dim]tokens: {usage.get('input_tokens', '?')} in / "
                f"{usage.get('output_tokens', '?')} out[/dim]"
            )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "Chain")
        _console.print(f"[yellow]⚙ Chain start:[/yellow] [bold]{escape(name)}[/bold]")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        _console.print("[green]✓ Chain complete[/green]")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        tool = serialized.get("name", "tool")
        _console.print(f"[magenta]🔧 Tool:[/magenta] [bold]{escape(tool)}[/bold]  input={escape(input_str[:80])}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        _console.print(f"[dim]   → {escape(str(output)[:120])}[/dim]")

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        _console.print(f"[blue]🤖 Agent action:[/blue] {escape(str(action.tool))} — {escape(str(action.tool_input)[:80])}")

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        _console.print(f"[green]🏁 Agent finished:[/green] {escape(str(finish.return_values)[:120])}")

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        _console.print(f"[red]✗ Chain error:[/red] {escape(str(error))}")

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        _console.print(f"[red]✗ Tool error:[/red] {escape(str(error))}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Token Streaming Callback  – SSE fan-out for the FastAPI server
# ─────────────────────────────────────────────────────────────────────────────

class TokenStreamingCallback(BaseCallbackHandler):
    """
    Puts each generated token into an asyncio.Queue.

    The FastAPI streaming endpoint reads from this queue and forwards tokens
    as Server-Sent Events to the OpenWebUI frontend.

    Usage:
        queue    = asyncio.Queue()
        callback = TokenStreamingCallback(queue)
        chain.invoke(input, config={"callbacks": [callback]})
        # Then in the SSE generator:
        while True:
            token = await queue.get()
            if token is None:   # sentinel
                break
            yield f"data: {token}\\n\\n"
    """

    # Sentinel placed in the queue to signal stream completion
    DONE = None

    def __init__(self, queue: asyncio.Queue) -> None:
        super().__init__()
        self._queue = queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Non-blocking put; drops tokens if queue is full (back-pressure)."""
        try:
            self._queue.put_nowait(token)
        except asyncio.QueueFull:
            logger.warning("Token queue full – dropping token")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Signal the SSE generator that the stream is finished."""
        try:
            self._queue.put_nowait(self.DONE)
        except asyncio.QueueFull:
            pass

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Surface errors through the queue so the SSE handler can close cleanly."""
        try:
            self._queue.put_nowait(f"\n[ERROR] {error}")
            self._queue.put_nowait(self.DONE)
        except asyncio.QueueFull:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tracing Callback  – structured execution trace for audit / observability
# ─────────────────────────────────────────────────────────────────────────────

class TracingCallback(BaseCallbackHandler):
    """
    Captures a full execution trace with timing and I/O at each step.

    After a chain run, inspect `callback.trace` for a list of step dicts
    suitable for logging to CloudWatch, OpenTelemetry, or LangSmith.

    Usage:
        tracer = TracingCallback(run_id="month-end-march-2025")
        chain.invoke(input, config={"callbacks": [tracer]})
        print(tracer.summary())
    """

    def __init__(self, run_id: str = "") -> None:
        super().__init__()
        self.run_id = run_id
        self.trace:  List[Dict[str, Any]] = []
        self._stack: Dict[str, float]     = {}   # step_key → start time

    # ── Recording helpers ─────────────────────────────────────────────────────

    def _start(self, kind: str, name: str, run_id: UUID, data: Dict) -> None:
        key = str(run_id)
        self._stack[key] = time.perf_counter()
        self.trace.append({"kind": kind, "name": name, "run_id": key, "phase": "start", **data})

    def _end(self, kind: str, name: str, run_id: UUID, data: Dict) -> None:
        key = str(run_id)
        elapsed = round((time.perf_counter() - self._stack.pop(key, time.perf_counter())) * 1000, 1)
        self.trace.append({"kind": kind, "name": name, "run_id": key, "phase": "end",
                           "elapsed_ms": elapsed, **data})

    # ── LangChain event hooks ─────────────────────────────────────────────────

    def on_llm_start(self, serialized: Dict, prompts: List[str],
                     run_id: UUID = UUID(int=0), **kw: Any) -> None:
        self._start("llm", serialized.get("name", "llm"), run_id,
                    {"prompt_count": len(prompts)})

    def on_llm_end(self, response: LLMResult, run_id: UUID = UUID(int=0), **kw: Any) -> None:
        usage = (response.llm_output or {}).get("usage", {})
        self._end("llm", "llm", run_id,
                  {"tokens_in": usage.get("input_tokens"),
                   "tokens_out": usage.get("output_tokens")})

    def on_chain_start(self, serialized: Dict, inputs: Dict,
                       run_id: UUID = UUID(int=0), **kw: Any) -> None:
        self._start("chain", serialized.get("name", "chain"), run_id,
                    {"input_keys": list(inputs.keys())})

    def on_chain_end(self, outputs: Dict, run_id: UUID = UUID(int=0), **kw: Any) -> None:
        self._end("chain", "chain", run_id, {"output_keys": list(outputs.keys())})

    def on_tool_start(self, serialized: Dict, input_str: str,
                      run_id: UUID = UUID(int=0), **kw: Any) -> None:
        self._start("tool", serialized.get("name", "tool"), run_id,
                    {"input": input_str[:200]})

    def on_tool_end(self, output: str, run_id: UUID = UUID(int=0), **kw: Any) -> None:
        self._end("tool", "tool", run_id, {"output": str(output)[:200]})

    def on_chain_error(self, error: Exception, run_id: UUID = UUID(int=0), **kw: Any) -> None:
        self.trace.append({"kind": "error", "run_id": str(run_id),
                           "message": str(error), "phase": "error"})

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return aggregated timing and step count for the full run."""
        steps = [s for s in self.trace if s["phase"] == "end"]
        total_ms = sum(s.get("elapsed_ms", 0) for s in steps)
        return {
            "run_id":        self.run_id,
            "total_steps":   len(steps),
            "total_ms":      round(total_ms, 1),
            "llm_calls":     sum(1 for s in steps if s["kind"] == "llm"),
            "tool_calls":    sum(1 for s in steps if s["kind"] == "tool"),
            "chain_calls":   sum(1 for s in steps if s["kind"] == "chain"),
            "errors":        sum(1 for s in self.trace if s["phase"] == "error"),
        }
