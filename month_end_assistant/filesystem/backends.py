"""
Swappable Filesystem Backends for LangChain Deep Agents.

Deep Agents need to read and write files as part of their work (reports,
intermediate data, cached research).  This module provides an abstraction
layer so the same agent code can run against:

  InMemoryBackend   – ephemeral dict; ideal for tests and short-lived agents
  LocalDiskBackend  – real files under a configurable root directory
  SandboxBackend    – isolated tempdir, auto-cleaned when the agent session ends
  DurableBackend    – S3-backed store (via boto3) for cross-session persistence

All backends enforce the FilePermissions rules declared at construction time,
so no agent node can bypass the access-control policy.

Usage
─────
    backend = SandboxBackend(permissions=FilePermissions(
        allowed_extensions=[".csv", ".json", ".txt", ".py"],
        max_file_size_mb=5,
    ))
    backend.write("reports/march_2025.json", json.dumps(report))
    content = backend.read("reports/march_2025.json")
    files   = backend.list("reports/")
    backend.close()   # cleans up sandbox tempdir
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Declarative Permission Rules
# ─────────────────────────────────────────────────────────────────────────────

class FilePermissions:
    """
    Declarative filesystem permission rules.

    Enforced by every backend before any read/write/delete operation.
    Instantiate once per agent or session and pass to the backend constructor.

    Example:
        perms = FilePermissions(
            allowed_paths=["/reports", "/data"],
            denied_paths=["/secrets"],
            allowed_extensions=[".csv", ".json", ".txt"],
            max_file_size_mb=10,
            read_only=False,
        )
    """

    def __init__(
        self,
        allowed_paths:      Optional[List[str]] = None,
        denied_paths:       Optional[List[str]] = None,
        allowed_extensions: Optional[List[str]] = None,
        max_file_size_mb:   float = 50.0,
        read_only:          bool  = False,
    ) -> None:
        # Normalise to forward slashes, strip trailing /
        self.allowed_paths      = [p.rstrip("/") for p in (allowed_paths or [])]
        self.denied_paths       = [p.rstrip("/") for p in (denied_paths  or [])]
        self.allowed_extensions = [e.lower() for e in (allowed_extensions or [])]
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.read_only           = read_only

    # ── Path checks ───────────────────────────────────────────────────────────

    def _is_allowed_path(self, path: str) -> bool:
        """Return True if *path* is within an allowed_paths entry (or no restriction set)."""
        if not self.allowed_paths:
            return True
        return any(path.startswith(p) for p in self.allowed_paths)

    def _is_denied_path(self, path: str) -> bool:
        """Return True if *path* falls under a denied_paths entry."""
        return any(path.startswith(p) for p in self.denied_paths)

    def _is_allowed_extension(self, path: str) -> bool:
        """Return True if the file extension is permitted (or no restriction set)."""
        if not self.allowed_extensions:
            return True
        return Path(path).suffix.lower() in self.allowed_extensions

    # ── Public enforcement API ────────────────────────────────────────────────

    def check_read(self, path: str) -> None:
        """Raise PermissionError if *path* cannot be read."""
        if self._is_denied_path(path):
            raise PermissionError(f"Read denied: '{path}' is in denied_paths.")
        if not self._is_allowed_path(path):
            raise PermissionError(f"Read denied: '{path}' is outside allowed_paths.")
        if not self._is_allowed_extension(path):
            raise PermissionError(f"Read denied: extension of '{path}' not allowed.")

    def check_write(self, path: str, content_bytes: int = 0) -> None:
        """Raise PermissionError if *path* cannot be written."""
        if self.read_only:
            raise PermissionError("Filesystem is read-only.")
        if self._is_denied_path(path):
            raise PermissionError(f"Write denied: '{path}' is in denied_paths.")
        if not self._is_allowed_path(path):
            raise PermissionError(f"Write denied: '{path}' is outside allowed_paths.")
        if not self._is_allowed_extension(path):
            raise PermissionError(f"Write denied: extension of '{path}' not allowed.")
        if content_bytes > self.max_file_size_bytes:
            mb = content_bytes / (1024 * 1024)
            raise PermissionError(
                f"Write denied: file size {mb:.1f} MB exceeds limit "
                f"{self.max_file_size_bytes / (1024*1024):.0f} MB."
            )

    def check_delete(self, path: str) -> None:
        """Raise PermissionError if *path* cannot be deleted."""
        if self.read_only:
            raise PermissionError("Filesystem is read-only.")
        if self._is_denied_path(path):
            raise PermissionError(f"Delete denied: '{path}' is in denied_paths.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Abstract Base Backend
# ─────────────────────────────────────────────────────────────────────────────

class FilesystemBackend(ABC):
    """
    Abstract filesystem backend.

    All backends share the same interface so agent tools are backend-agnostic.
    Swap the backend at construction time without changing any agent code.
    """

    def __init__(self, permissions: Optional[FilePermissions] = None) -> None:
        self.permissions = permissions or FilePermissions()

    @abstractmethod
    def read(self, path: str) -> str:
        """Read file content as a UTF-8 string."""

    @abstractmethod
    def write(self, path: str, content: str) -> None:
        """Write UTF-8 string content to *path*."""

    @abstractmethod
    def list(self, directory: str = "") -> List[str]:
        """List files under *directory*."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Return True if *path* exists in the backend."""

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete *path*."""

    def close(self) -> None:
        """Release any resources held by the backend (e.g. tempdir cleanup)."""

    # ── Convenience ───────────────────────────────────────────────────────────

    def read_json(self, path: str) -> Any:
        return json.loads(self.read(path))

    def write_json(self, path: str, data: Any) -> None:
        self.write(path, json.dumps(data, indent=2, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# 3. InMemoryBackend  – dict-backed, ephemeral
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryBackend(FilesystemBackend):
    """
    In-process dict filesystem.

    Ideal for:
      • Unit tests (no I/O)
      • Short-lived agent sessions where persistence is not needed
      • Passing data between nodes without touching disk

    Paths are normalised to forward-slash strings (no OS dependency).
    """

    def __init__(self, permissions: Optional[FilePermissions] = None) -> None:
        super().__init__(permissions)
        self._store: Dict[str, str] = {}

    def read(self, path: str) -> str:
        self.permissions.check_read(path)
        if path not in self._store:
            raise FileNotFoundError(f"'{path}' not found in InMemoryBackend.")
        return self._store[path]

    def write(self, path: str, content: str) -> None:
        self.permissions.check_write(path, content_bytes=len(content.encode()))
        self._store[path] = content
        logger.debug("InMemory write: %s (%d bytes)", path, len(content))

    def list(self, directory: str = "") -> List[str]:
        prefix = directory.rstrip("/") + "/" if directory else ""
        return [p for p in self._store if p.startswith(prefix)]

    def exists(self, path: str) -> bool:
        return path in self._store

    def delete(self, path: str) -> None:
        self.permissions.check_delete(path)
        self._store.pop(path, None)

    @property
    def size(self) -> int:
        """Total bytes stored (for monitoring)."""
        return sum(len(v.encode()) for v in self._store.values())


# ─────────────────────────────────────────────────────────────────────────────
# 4. LocalDiskBackend  – real files on local filesystem
# ─────────────────────────────────────────────────────────────────────────────

class LocalDiskBackend(FilesystemBackend):
    """
    Local-disk backend rooted at a configurable base directory.

    All paths are relative to *root_dir* and are jail-ed (path traversal
    attempts raise PermissionError before reaching the OS).

    Usage:
        backend = LocalDiskBackend(root_dir="/var/data/agent-workspace")
    """

    def __init__(
        self,
        root_dir: str,
        permissions: Optional[FilePermissions] = None,
    ) -> None:
        super().__init__(permissions)
        self.root = Path(root_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info("LocalDiskBackend rooted at %s", self.root)

    def _safe_path(self, path: str) -> Path:
        """Resolve *path* relative to root and ensure it stays within root (jail)."""
        full = (self.root / path.lstrip("/")).resolve()
        if not str(full).startswith(str(self.root)):
            raise PermissionError(
                f"Path traversal detected: '{path}' resolves outside root '{self.root}'."
            )
        return full

    def read(self, path: str) -> str:
        self.permissions.check_read(path)
        full = self._safe_path(path)
        if not full.exists():
            raise FileNotFoundError(f"'{path}' not found on disk.")
        return full.read_text(encoding="utf-8")

    def write(self, path: str, content: str) -> None:
        self.permissions.check_write(path, content_bytes=len(content.encode()))
        full = self._safe_path(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        logger.debug("LocalDisk write: %s (%d bytes)", path, len(content))

    def list(self, directory: str = "") -> List[str]:
        base  = self._safe_path(directory) if directory else self.root
        if not base.is_dir():
            return []
        return [
            str(p.relative_to(self.root))
            for p in base.rglob("*")
            if p.is_file()
        ]

    def exists(self, path: str) -> bool:
        return self._safe_path(path).exists()

    def delete(self, path: str) -> None:
        self.permissions.check_delete(path)
        full = self._safe_path(path)
        if full.exists():
            full.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# 5. SandboxBackend  – isolated tempdir, auto-cleaned on close()
# ─────────────────────────────────────────────────────────────────────────────

class SandboxBackend(LocalDiskBackend):
    """
    Sandboxed filesystem that lives in an OS tempdir.

    The tempdir is created automatically at construction and destroyed when
    close() is called (or used as a context manager).

    This is the backend used by the `execute` tool and DeepAgent when running
    code in isolation – matching the LangChain Deep Agents 'sandbox backend'.

    Usage:
        with SandboxBackend() as fs:
            fs.write("script.py", "print('hello')")
            output = execute_tool.invoke({"command": "python script.py"})
    """

    def __init__(self, permissions: Optional[FilePermissions] = None) -> None:
        tempdir = tempfile.mkdtemp(prefix="deep-agent-sandbox-")
        # Default sandbox permissions: no system paths, only safe extensions
        default_perms = permissions or FilePermissions(
            denied_paths=["/etc", "/usr", "/bin", "/sbin", "/home"],
            allowed_extensions=[".py", ".csv", ".json", ".txt", ".md", ".html"],
            max_file_size_mb=10,
        )
        super().__init__(root_dir=tempdir, permissions=default_perms)
        self._tempdir = tempdir
        logger.info("SandboxBackend created at %s", tempdir)

    def close(self) -> None:
        """Remove the entire sandbox tempdir."""
        if os.path.exists(self._tempdir):
            shutil.rmtree(self._tempdir, ignore_errors=True)
            logger.info("SandboxBackend cleaned up: %s", self._tempdir)

    def __enter__(self) -> "SandboxBackend":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. DurableBackend  – S3-backed for cross-session persistence
# ─────────────────────────────────────────────────────────────────────────────

class DurableBackend(FilesystemBackend):
    """
    S3-backed durable filesystem for cross-session, cross-instance persistence.

    Agent data (reports, research notes, cached data) survives container
    restarts and is accessible from any instance.

    Falls back to InMemoryBackend if boto3 / S3 is unavailable.

    Usage:
        backend = DurableBackend(bucket="my-agent-bucket", prefix="month-end/")
    """

    def __init__(
        self,
        bucket:      str,
        prefix:      str  = "agent-data/",
        region:      str  = "us-east-1",
        permissions: Optional[FilePermissions] = None,
    ) -> None:
        super().__init__(permissions)
        self._bucket  = bucket
        self._prefix  = prefix.rstrip("/") + "/"
        self._client  = self._build_s3_client(region)
        self._fallback = InMemoryBackend(permissions)

    def _build_s3_client(self, region: str) -> Optional[Any]:
        try:
            import boto3
            return boto3.client("s3", region_name=region)
        except Exception as exc:
            logger.warning("S3 client unavailable (%s) – falling back to InMemory.", exc)
            return None

    def _s3_key(self, path: str) -> str:
        return self._prefix + path.lstrip("/")

    def read(self, path: str) -> str:
        self.permissions.check_read(path)
        if not self._client:
            return self._fallback.read(path)
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=self._s3_key(path))
            return obj["Body"].read().decode("utf-8")
        except Exception as exc:
            raise FileNotFoundError(f"S3 read failed for '{path}': {exc}")

    def write(self, path: str, content: str) -> None:
        self.permissions.check_write(path, content_bytes=len(content.encode()))
        if not self._client:
            self._fallback.write(path, content)
            return
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=self._s3_key(path),
                Body=content.encode("utf-8"),
            )
            logger.debug("S3 write: s3://%s/%s", self._bucket, self._s3_key(path))
        except Exception as exc:
            logger.error("S3 write failed: %s – falling back to InMemory.", exc)
            self._fallback.write(path, content)

    def list(self, directory: str = "") -> List[str]:
        if not self._client:
            return self._fallback.list(directory)
        try:
            prefix = self._prefix + directory.lstrip("/")
            response = self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
            return [
                obj["Key"].removeprefix(self._prefix)
                for obj in response.get("Contents", [])
            ]
        except Exception:
            return self._fallback.list(directory)

    def exists(self, path: str) -> bool:
        if not self._client:
            return self._fallback.exists(path)
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._s3_key(path))
            return True
        except Exception:
            return False

    def delete(self, path: str) -> None:
        self.permissions.check_delete(path)
        if not self._client:
            self._fallback.delete(path)
            return
        try:
            self._client.delete_object(Bucket=self._bucket, Key=self._s3_key(path))
        except Exception as exc:
            logger.error("S3 delete failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Backend factory
# ─────────────────────────────────────────────────────────────────────────────

def create_backend(
    kind: str,
    permissions: Optional[FilePermissions] = None,
    **kwargs: Any,
) -> FilesystemBackend:
    """
    Factory function – create a backend by name.

    Args:
        kind:        'memory' | 'local' | 'sandbox' | 'durable'
        permissions: FilePermissions instance (uses safe defaults if None)
        **kwargs:    Passed to the backend constructor
                     (root_dir for 'local'; bucket for 'durable')

    Returns:
        A FilesystemBackend instance ready for use.
    """
    registry = {
        "memory":  InMemoryBackend,
        "local":   LocalDiskBackend,
        "sandbox": SandboxBackend,
        "durable": DurableBackend,
    }
    cls = registry.get(kind.lower())
    if cls is None:
        raise ValueError(f"Unknown backend '{kind}'. Choose from: {list(registry)}")

    if permissions:
        kwargs["permissions"] = permissions
    return cls(**kwargs)
