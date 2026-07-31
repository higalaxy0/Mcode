"""AppContext - singleton container for all *mutable runtime state*.

Path/API configuration remains in :mod:`mcodecore.config`; only mutable runtime
state (todos, pending requests, active teammates, skill registry, bus, memory
lock, calibrator) is collected here.

A global singleton ``ctx`` is provided for other modules to use via
``from .context import ctx``.
"""

from __future__ import annotations

import threading
from typing import Any

from .calibrator import TokenCalibrator


class AppContext:
    """Global mutable runtime state container.

    All former module-level globals are collected as instance attributes.
    Path/API configuration is still referenced from :mod:`mcodecore.config`.
    """

    def __init__(self) -> None:
        # todos ----------------------------------------------------------------
        self.current_todos: list[dict] = []
        # session id (always None; kept for compatibility)
        self.current_session_id: str | None = None

        # message bus & protocol (lazily created, see _bus property) -----------
        self._bus = None
        self._mcp_manager = None
        self.pending_requests: dict[str, Any] = {}
        # teammates ------------------------------------------------------------
        self.active_teammates: dict[str, threading.Event] = {}
        self.teammate_registry: dict[str, dict] = {}

        # skills ---------------------------------------------------------------
        self.skill_registry: dict[str, dict] = {}

        # hooks ----------------------------------------------------------------
        self.hooks: dict[str, list] = {
            "UserPromptSubmit": [],
            "PreToolUse": [],
            "PostToolUse": [],
            "Stop": [],
        }

        # memory lock + token calibrator ---------------------------------------
        self.memory_lock: threading.Lock = threading.Lock()
        self.memory_lock_timeout: int = 30
        self.calibrator: TokenCalibrator = TokenCalibrator()

    # -- message bus (lazily created to avoid circular imports) ----------------
    @property
    def bus(self):
        if self._bus is None:
            from .bus import MessageBus
            self._bus = MessageBus()
        return self._bus

    # -- MCP manager (lazily created to avoid circular imports) ---------------
    @property
    def mcp(self):
        if self._mcp_manager is None:
            from .mcp import MCPManager
            self._mcp_manager = MCPManager()
        return self._mcp_manager

    # -- hooks convenience methods --------------------------------------------
    def register_hook(self, event: str, callback) -> None:
        """Register a hook callback."""
        self.hooks[event].append(callback)

    def trigger_hooks(self, event: str, *args):
        """Trigger all hooks for *event*; the first non-None return wins."""
        for callback in self.hooks[event]:
            result = callback(*args)
            if result is not None:
                return result
        return None


# Global singleton ----------------------------------------------------------------
ctx: AppContext = AppContext()
