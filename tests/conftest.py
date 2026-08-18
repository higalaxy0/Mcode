"""Shared fixtures.

Key design: ``mcodecore`` modules bind path constants such as ``WORKDIR`` /
``TASKS_DIR`` / ``MAILBOX_DIR`` at *import time* (``from .config import X``),
so tests must redirect these constants to ``tmp_path`` in **each consuming
module's namespace**, otherwise the real workspace's ``.tasks`` /
``.mailboxes`` would be used and cause cross-test interference.

The ``isolate_paths`` autouse fixture does this uniformly, covering all known
consumers.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


# List of "module attribute names" whose path constants need redirecting
# (module name -> list of constant names bound within that module)
_PATH_BINDINGS: dict[str, list[str]] = {
    "mcodecore.tasks": ["TASKS_DIR"],
    "mcodecore.bus": ["MAILBOX_DIR", "WORKDIR", "SESSION_ID"],
    "mcodecore.memory": ["MEMORY_DIR", "MEMORY_INDEX", "TRANSCRIPT_DIR"],
    "mcodecore.fsops": ["WORKDIR", "_BG_OUTPUT_DIR"],
    "mcodecore.tools": ["WORKDIR"],
    "mcodecore.teammates": ["WORKDIR", "TEAM_HISTORY_DIR"],
}


@pytest.fixture(autouse=True)
def isolate_paths(tmp_path, monkeypatch):
    """Redirect path constants of all consuming modules to subdirs of tmp_path.

    This way test reads/writes to ``.tasks`` / ``.mailboxes`` / ``.memory``
    all land in a temporary directory, isolated from each other and not
    polluting the real workspace.

    Session-scoped dirs (``.mailboxes/<sid>``, ``.tasks/<sid>``,
    ``.team_history/<sid>``) use a *fixed test session id* so tests remain
    deterministic; the legacy flat layout (pre-session-scoping) is kept as
    a plain ``.mailboxes`` dir NOT containing session subdirs, exactly
    like the real workspace after the multi-window fix.
    """
    sid = "s_testsession"
    subdirs = {
        "WORKDIR": tmp_path,
        "SESSION_ID": sid,
        "TASKS_DIR": tmp_path / ".tasks" / sid,
        "MAILBOX_DIR": tmp_path / ".mailboxes" / sid,
        "TEAM_HISTORY_DIR": tmp_path / ".team_history" / sid,
        "MEMORY_DIR": tmp_path / ".memory",
        "MEMORY_INDEX": tmp_path / ".memory" / "MEMORY.md",
        "_BG_OUTPUT_DIR": tmp_path / ".task_outputs" / "bg-logs",
    }
    for d in ("TASKS_DIR", "MAILBOX_DIR", "TEAM_HISTORY_DIR",
              "MEMORY_DIR", "_BG_OUTPUT_DIR"):
        subdirs[d].mkdir(parents=True, exist_ok=True)

    for modname, attrs in _PATH_BINDINGS.items():
        mod = importlib.import_module(modname)
        for attr in attrs:
            if attr in subdirs:
                monkeypatch.setattr(mod, attr, subdirs[attr], raising=True)

    # The memory module caches list_memory_files() results at module level;
    # invalidate it so a fresh tmp_path does not return the previous test's data.
    from mcodecore import memory as _mem
    _mem._invalidate_memory_cache()

    # ctx is a global singleton; reset mutable runtime state to avoid leaking across tests
    from mcodecore.context import ctx
    ctx.current_todos = []
    ctx.pending_requests = {}
    ctx.active_teammates = {}
    ctx.teammate_registry = {}
    ctx.skill_registry = {}
    ctx.hooks = {k: [] for k in ctx.hooks}
    # Reset calibrator samples/factor too, so token-estimation tests don't interfere
    ctx.calibrator.samples.clear()
    ctx.calibrator.calibration_factor = 1.0
    yield


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Convenience alias returning the isolated working directory."""
    return tmp_path
