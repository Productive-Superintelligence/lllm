"""
lllm.logging — logging and session persistence for LLLM.

Public API:
    LogBackend          — abstract KV storage driver
    LocalFileBackend    — one file per key under a root directory
    SQLiteBackend       — single SQLite file
    NoOpBackend         — silently discards everything
    LogStore            — persists / queries TacticCallSessions
    SessionRecord       — full session + metadata (returned by load_session_record)
    SessionSummary      — lightweight descriptor (returned by list_sessions)
    ColoredFormatter    — ANSI-colored logging.Formatter
    setup_logging       — convenience function to configure the lllm logger

Convenience factories:
    local_store(path, partition)   — LocalFileBackend-backed LogStore
    sqlite_store(path, partition)  — SQLiteBackend-backed LogStore
    noop_store(partition)          — NoOpBackend-backed LogStore (dry run)
"""

from lllm.logging.backend import (
    LogBackend,
    LocalFileBackend,
    SQLiteBackend,
    NoOpBackend,
)
from lllm.logging.store import LogStore
from lllm.logging.models import SessionRecord, SessionSummary
from lllm.logging.formatter import ColoredFormatter, setup_logging


def local_store(
    path: str,
    partition: str = "default",
    runtime=None,
) -> LogStore:
    """Create a LogStore backed by the local filesystem.

    Args:
        path:      Directory where session files will be written.
                   Created automatically if it does not exist.
        partition: Key prefix for partitioning a shared backend among multiple
                   stores.  Two stores with different partitions in the same
                   directory are fully isolated.
        runtime:   Optional Runtime instance.  Enables alias resolution in
                   ``list_sessions(tactic_path=...)``.

    Example::

        store = local_store("~/.lllm/logs")
        store = local_store("/data/runs", partition="exp-42", runtime=rt)
    """
    import os
    return LogStore(
        LocalFileBackend(os.path.expanduser(path)),
        partition=partition,
        runtime=runtime,
    )


def sqlite_store(
    path: str,
    partition: str = "default",
    runtime=None,
) -> LogStore:
    """Create a LogStore backed by a single SQLite file.

    Preferred over ``local_store`` when you need atomic writes, concurrent
    access from multiple processes, or portability of the whole log as one file.

    Args:
        path:      Path to the ``.db`` file.  Parent directory is created
                   automatically.  Use ``:memory:`` for in-process testing.
        partition: Key prefix for partitioning a shared backend.
        runtime:   Optional Runtime instance for alias resolution.

    Example::

        store = sqlite_store("~/.lllm/logs.db")
        store = sqlite_store(":memory:", partition="test")
    """
    import os
    return LogStore(
        SQLiteBackend(os.path.expanduser(path)),
        partition=partition,
        runtime=runtime,
    )


def noop_store(partition: str = "default", runtime=None) -> LogStore:
    """Create a LogStore that silently discards everything.

    Useful for disabling persistence in tests or dry-run scripts without
    changing tactic code.

    Example::

        store = noop_store()
        tactic = build_tactic(config, ckpt_dir, log_store=store)
    """
    return LogStore(NoOpBackend(), partition=partition, runtime=runtime)


__all__ = [
    # Core abstractions
    "LogBackend",
    "LocalFileBackend",
    "SQLiteBackend",
    "NoOpBackend",
    "LogStore",
    # Data models
    "SessionRecord",
    "SessionSummary",
    # Terminal logging
    "ColoredFormatter",
    "setup_logging",
    # Convenience factories
    "local_store",
    "sqlite_store",
    "noop_store",
]
