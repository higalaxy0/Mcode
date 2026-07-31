"""Hook system: registration / dispatch / built-in hooks."""

from __future__ import annotations

from .config import WORKDIR, _DENY_LIST
from .context import ctx
from .utils import parse_tool_args


def register_hook(event: str, callback) -> None:
    """Register a hook callback for *event*."""
    ctx.register_hook(event, callback)


def trigger_hooks(event: str, *args):
    """Trigger all hooks for *event*; the first non-None return wins."""
    return ctx.trigger_hooks(event, *args)


# -- Built-in hooks ----------------------------------------------------------- #

def permission_hook(function) -> str | None:
    """PreToolUse: deny-list command check."""
    if function.name == "bash":
        cmd = parse_tool_args(function.arguments).get("command", "")
        for p in _DENY_LIST:
            if p in cmd:
                print(f"\n\033[31m⛔ Blocked: '{p}'\033[0m")
                return "Permission denied"
    return None


def log_hook(function) -> None:
    """PreToolUse: print tool call log."""
    print(f"\033[90m[HOOK] {function.name}\033[0m")
    return None


def context_inject_hook(query: str) -> None:
    """UserPromptSubmit: print working directory."""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def summary_hook(messages: list) -> None:
    """Stop: print total tool call count for the session."""
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("role") == "tool")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


def install_default_hooks() -> None:
    """Register all built-in hooks (call once at startup)."""
    register_hook("UserPromptSubmit", context_inject_hook)
    register_hook("PreToolUse", permission_hook)
    register_hook("PreToolUse", log_hook)
    register_hook("Stop", summary_hook)
