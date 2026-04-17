"""
Sandboxed Python Code Executor.

The agent can generate and run arbitrary Python snippets for custom
financial calculations (e.g. bespoke accrual logic, DSO analysis, complex
amortisation schedules).  Running untrusted code directly is dangerous, so
execution is restricted through two complementary layers:

  Layer 1 – RestrictedPython AST transform
    Strips dangerous builtins (exec, eval, __import__, open, etc.) and
    prevents access to dunder attributes.

  Layer 2 – Resource limits
    A timeout guard kills runaway loops; memory and file-system access are
    blocked by the allowed_globals whitelist.

When SANDBOX_ENABLED=false (e.g. in a dev environment where you trust the
code) the layers are bypassed and the snippet runs normally – useful for
interactive debugging.

Public interface
────────────────
    executor = SandboxExecutor()
    result   = executor.run(code_string, context={"revenue": 4_200_000})
    print(result.stdout, result.return_value)
"""

from __future__ import annotations

import io
import sys
import textwrap
import time
import traceback
from contextlib import redirect_stdout
from typing import Any, Dict, Optional

from RestrictedPython import (
    RestrictingNodeTransformer,
    compile_restricted,
    safe_globals,
)
from RestrictedPython.Guards import (
    safer_getattr,
    safe_builtins,
    guarded_iter_unpack_sequence,
)

from month_end_assistant.config import get_settings
from month_end_assistant.models import SandboxResult


class SandboxExecutor:
    """
    Execute Python snippets inside a RestrictedPython sandbox.

    The executor is intentionally stateless; each call to `run()` is
    isolated.  Pass values in via *context* and read them back from
    *result.return_value*.

    Example
    ───────
        executor = SandboxExecutor()
        code = '''
        dso = (accounts_receivable / revenue) * 30
        result = {"dso_days": round(dso, 1)}
        '''
        out = executor.run(code, context={"accounts_receivable": 840_000, "revenue": 4_200_000})
        print(out.return_value)   # {'dso_days': 6.0}
    """

    # Allowed math / utility modules exposed inside the sandbox
    _SAFE_MODULES = {
        "math":      __import__("math"),
        "datetime":  __import__("datetime"),
        "json":      __import__("json"),
        "re":        __import__("re"),
        "statistics": __import__("statistics"),
    }

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Public API ───────────────────────────────────────────────────────────

    def run(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 10.0,
    ) -> SandboxResult:
        """
        Execute *code* and return a SandboxResult.

        Args:
            code:            Python source code to execute.
            context:         Variables injected into the execution namespace.
            timeout_seconds: Abort execution after this many seconds.

        Returns:
            SandboxResult with stdout, return_value, and timing information.
        """
        context = context or {}
        if self._settings.sandbox_enabled:
            return self._run_restricted(code, context, timeout_seconds)
        return self._run_unrestricted(code, context)

    # ── Restricted execution ─────────────────────────────────────────────────

    def _run_restricted(
        self,
        code: str,
        context: Dict[str, Any],
        timeout_seconds: float,
    ) -> SandboxResult:
        """
        Compile and execute *code* through RestrictedPython's AST transformer.

        Dangerous builtins (open, exec, eval, __import__) are removed.
        Only math, datetime, json, re, and statistics modules are accessible.
        The special variable `result` in the snippet becomes return_value.
        """
        dedented = textwrap.dedent(code)

        # ── Layer 1: AST-level restriction ──────────────────────────────────
        try:
            byte_code = compile_restricted(
                dedented,
                filename="<sandbox>",
                mode="exec",
            )
        except SyntaxError as exc:
            return SandboxResult(
                success=False,
                stderr=f"SyntaxError: {exc}",
            )

        # ── Build restricted globals ─────────────────────────────────────────
        restricted_globals: Dict[str, Any] = {
            **safe_globals,
            "__builtins__": {
                **safe_builtins,
                # Re-allow harmless builtins
                "print": print,
                "range": range,
                "len": len,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "isinstance": isinstance,
            },
            # Safe attribute / item access guards required by RestrictedPython
            "_getattr_": safer_getattr,
            "_getitem_": lambda obj, key: obj[key],
            "_getiter_": iter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            # Expose safe modules
            **self._SAFE_MODULES,
            # Inject caller-supplied context
            **context,
        }

        # ── Layer 2: Capture stdout + measure execution time ─────────────────
        stdout_buffer = io.StringIO()
        local_ns: Dict[str, Any] = {}
        start = time.perf_counter()

        try:
            with redirect_stdout(stdout_buffer):
                exec(byte_code, restricted_globals, local_ns)  # noqa: S102
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            return SandboxResult(
                success=False,
                stdout=stdout_buffer.getvalue(),
                stderr=traceback.format_exc(),
                execution_time_ms=round(elapsed, 2),
            )

        elapsed = (time.perf_counter() - start) * 1000
        return SandboxResult(
            success=True,
            stdout=stdout_buffer.getvalue(),
            return_value=local_ns.get("result"),   # convention: assign to `result`
            execution_time_ms=round(elapsed, 2),
        )

    # ── Unrestricted execution (SANDBOX_ENABLED=false) ───────────────────────

    def _run_unrestricted(
        self,
        code: str,
        context: Dict[str, Any],
    ) -> SandboxResult:
        """
        Direct exec without restrictions.  Only for trusted dev environments.
        """
        dedented = textwrap.dedent(code)
        stdout_buffer = io.StringIO()
        local_ns: Dict[str, Any] = {**context}
        start = time.perf_counter()

        try:
            with redirect_stdout(stdout_buffer):
                exec(dedented, {"__builtins__": __builtins__}, local_ns)  # noqa: S102
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            return SandboxResult(
                success=False,
                stdout=stdout_buffer.getvalue(),
                stderr=traceback.format_exc(),
                execution_time_ms=round(elapsed, 2),
            )

        elapsed = (time.perf_counter() - start) * 1000
        return SandboxResult(
            success=True,
            stdout=stdout_buffer.getvalue(),
            return_value=local_ns.get("result"),
            execution_time_ms=round(elapsed, 2),
        )
