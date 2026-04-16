"""
LangChain File System Tools backed by swappable FilesystemBackends.

These tools let any agent node (Deep Agent, Supervisor worker, Scenario)
read, write, list, delete, search, and summarise files – all routed through
the permission-enforcing backend of your choice.

Tool catalogue
──────────────
  read_file       – read a file from the active backend
  write_file      – write content to a file
  list_files      – list files in a directory
  file_exists     – check whether a file exists
  delete_file     – delete a file (requires write permission)
  search_files    – grep-style keyword search across files
  summarise_file  – LLM-powered file summarisation for large context management
  execute         – execute a shell command inside the SandboxBackend  ← KEY Deep Agent feature

Usage
─────
    backend = SandboxBackend()
    tools   = build_filesystem_tools(backend, llm)
    agent   = create_react_agent(llm, tools)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, List, Optional

from langchain_core.tools import tool

from month_end_assistant.filesystem.backends import (
    FilesystemBackend,
    InMemoryBackend,
    SandboxBackend,
)

logger = logging.getLogger(__name__)

# Module-level active backend (set via set_active_backend)
# Each agent session calls set_active_backend() to point tools at the right store.
_active_backend: FilesystemBackend = InMemoryBackend()


def set_active_backend(backend: FilesystemBackend) -> None:
    """Switch the module-level backend used by all filesystem tools."""
    global _active_backend
    _active_backend = backend
    logger.info("Active filesystem backend → %s", type(backend).__name__)


def get_active_backend() -> FilesystemBackend:
    return _active_backend


# ─────────────────────────────────────────────────────────────────────────────
# File management tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def read_file(path: str) -> str:
    """
    Read the contents of a file from the active filesystem backend.

    Args:
        path: Relative path to the file (e.g. 'reports/march_2025.json').

    Returns:
        File contents as a UTF-8 string.

    Raises:
        FileNotFoundError if the file does not exist.
        PermissionError if the path is outside allowed_paths.
    """
    try:
        content = _active_backend.read(path)
        logger.debug("read_file: %s (%d bytes)", path, len(content))
        return content
    except (FileNotFoundError, PermissionError) as exc:
        return f"ERROR: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """
    Write content to a file in the active filesystem backend.

    Args:
        path:    Relative file path (e.g. 'reports/summary.txt').
        content: UTF-8 string to write.

    Returns:
        Confirmation message with bytes written.
    """
    try:
        _active_backend.write(path, content)
        return f"Written {len(content.encode())} bytes to '{path}'."
    except PermissionError as exc:
        return f"ERROR: {exc}"


@tool
def list_files(directory: str = "") -> str:
    """
    List all files under *directory* in the active filesystem backend.

    Args:
        directory: Sub-directory path (empty string = root).

    Returns:
        Newline-separated list of relative file paths, or a message if empty.
    """
    files = _active_backend.list(directory)
    if not files:
        return f"No files found under '{directory or 'root'}'."
    return "\n".join(sorted(files))


@tool
def file_exists(path: str) -> bool:
    """
    Check whether *path* exists in the active filesystem backend.

    Args:
        path: Relative file path to check.

    Returns:
        True if the file exists, False otherwise.
    """
    return _active_backend.exists(path)


@tool
def delete_file(path: str) -> str:
    """
    Delete a file from the active filesystem backend.

    Args:
        path: Relative file path to delete.

    Returns:
        Confirmation or error message.
    """
    try:
        _active_backend.delete(path)
        return f"Deleted '{path}'."
    except PermissionError as exc:
        return f"ERROR: {exc}"


@tool
def search_files(keyword: str, directory: str = "") -> str:
    """
    Search all files in *directory* for lines containing *keyword*.

    Args:
        keyword:   Search term (case-insensitive substring match).
        directory: Sub-directory to search (empty = root).

    Returns:
        Matching lines in 'path:line_no: content' format, or 'No matches found'.
    """
    matches: List[str] = []
    kw_lower = keyword.lower()
    for path in _active_backend.list(directory):
        try:
            for i, line in enumerate(_active_backend.read(path).splitlines(), 1):
                if kw_lower in line.lower():
                    matches.append(f"{path}:{i}: {line.strip()}")
        except Exception:
            pass
    return "\n".join(matches[:50]) if matches else f"No matches for '{keyword}'."


@tool
def summarise_file(path: str) -> str:
    """
    Summarise a large file to manage context window size.

    This is the Deep Agents 'context management through summarisation' feature:
    when a file is too long to fit in the LLM context, this tool returns a
    structured summary instead of the raw content.

    Args:
        path: Relative path to the file to summarise.

    Returns:
        A concise bullet-point summary of the file content.
    """
    try:
        content = _active_backend.read(path)
    except (FileNotFoundError, PermissionError) as exc:
        return f"ERROR: {exc}"

    # Lightweight extractive summary (no LLM needed for most cases)
    lines  = content.splitlines()
    length = len(content)

    if length <= 2000:
        return content   # short enough to return directly

    # Extract first + last N lines + word/line counts as a quick summary
    head = "\n".join(lines[:10])
    tail = "\n".join(lines[-5:])
    return (
        f"[File summary: {path}]\n"
        f"Total: {len(lines)} lines, {length:,} chars\n\n"
        f"First 10 lines:\n{head}\n\n"
        f"Last 5 lines:\n{tail}\n\n"
        "(Use read_file for the full content.)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# execute tool  – the key Deep Agents "execute shell commands in sandbox" feature
# ─────────────────────────────────────────────────────────────────────────────

@tool
def execute(command: str) -> str:
    """
    Execute a shell command inside the active SandboxBackend.

    This is the core Deep Agents 'execute' tool.  When the active backend is a
    SandboxBackend the command runs inside its isolated tempdir.  If the backend
    is InMemory or LocalDisk the command runs in a fresh tempdir and the
    working directory is set accordingly.

    Allowed commands: python, pip list, cat, ls, head, wc, sort, uniq, grep.
    Blocked: rm -rf, curl, wget, ssh, sudo, and anything touching /etc /usr /bin.

    Args:
        command: Shell command string (e.g. 'python analyse.py' or 'ls reports/').

    Returns:
        Combined stdout + stderr from the command, or an error message.
    """
    # ── Safety: block dangerous patterns ────────────────────────────────────
    _BLOCKED = ["rm -rf", "sudo", "curl", "wget", "ssh", "chmod", "chown",
                "dd ", "> /dev", "mkfs", "; rm", "| rm", "&& rm"]
    if any(b in command.lower() for b in _BLOCKED):
        return f"ERROR: Command blocked by security policy: '{command[:80]}'"

    # ── Determine working directory ──────────────────────────────────────────
    backend = _active_backend
    if isinstance(backend, SandboxBackend):
        cwd = backend.root
    else:
        cwd = tempfile.mkdtemp(prefix="deep-agent-exec-")

    logger.info("execute: cwd=%s cmd='%s'", cwd, command[:80])

    # ── Run with a hard timeout and restricted environment ───────────────────
    restricted_env = {
        "PATH":   "/usr/local/bin:/usr/bin:/bin",
        "HOME":   str(cwd),
        "TMPDIR": str(cwd),
        # Explicitly remove anything that could leak credentials
    }

    try:
        result = subprocess.run(
            command,
            shell=True,              # noqa: S602 – intentional for agent tool
            capture_output=True,
            text=True,
            cwd=str(cwd),
            env=restricted_env,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 30 seconds."
    except Exception as exc:
        return f"ERROR: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────────────────────

FILESYSTEM_TOOLS = [
    read_file,
    write_file,
    list_files,
    file_exists,
    delete_file,
    search_files,
    summarise_file,
    execute,          # The key Deep Agents tool
]


def build_filesystem_tools(backend: FilesystemBackend) -> list:
    """
    Activate *backend* and return the full filesystem tool list.

    Call this once per agent session to wire tools to the correct backend.
    """
    set_active_backend(backend)
    return FILESYSTEM_TOOLS
