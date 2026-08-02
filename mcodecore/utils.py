"""Generic helper functions."""

from __future__ import annotations

import re


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse Markdown frontmatter (a ``---``-delimited YAML header).

    Parameters
    ----------
    text:
        Raw file text.

    Returns
    -------
    (metadata, body)
        ``metadata`` is a ``key: value`` dict; ``body`` is the remaining text.
        Returns ``({}, text)`` when no frontmatter is present.
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, parts[2].strip()


def parse_bg_command(command: str) -> tuple[bool, str | None, str]:
    """Parse a ``bg: <cmd>`` prefix.

    Returns
    -------
    (is_bg, log_name, cmd_core)
    """
    m = re.match(r'^bg:\s*(.*)', command.strip(), re.DOTALL)
    if not m:
        return False, None, command
    cmd_core = m.group(1).strip()
    log_name = None
    lm = re.search(r'\s+log=(\S+)\s*$', cmd_core)
    if lm:
        log_name = lm.group(1)
        cmd_core = cmd_core[:lm.start()].rstrip()
    return True, log_name, cmd_core


def parse_explicit_timeout(command: str) -> tuple[int, str]:
    """Parse a trailing ``# timeout=N`` explicit timeout.

    Returns
    -------
    (timeout, stripped_command)
    """
    from .config import BASH_TIMEOUT
    m = re.search(r'#\s*timeout=(\d+)\s*$', command)
    if m:
        return int(m.group(1)), command[:m.start()].rstrip()
    return BASH_TIMEOUT, command


def truncate(text, limit: int = 200) -> str:
    """Truncate *text* to *limit* characters, appending ``...`` if exceeded."""
    if not isinstance(text, str):
        text = str(text)
    return text if len(text) <= limit else text[:limit] + "..."


def new_request_id() -> str:
    """Generate a 6-digit random request id."""
    import random
    return f"req_{random.randint(0, 999999):06d}"


def parse_tool_args(arguments: str | None) -> dict:
    """Safely parse a tool-call ``arguments`` JSON string.

    Returns an empty dict on any decode error or empty input, instead of
    raising ``JSONDecodeError`` (which would crash the agent loop / teammate
    thread / subagent).
    """
    import json as _json
    if not arguments:
        return {}
    try:
        parsed = _json.loads(arguments)
    except (ValueError, TypeError):
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _validate_tool_args(arguments: str | None) -> str:
    """Return a valid JSON-object string for tool-call *arguments*.

    The backend rejects assistant messages whose ``tool_calls[].function.arguments``
    is not a valid JSON object (e.g. truncated JSON from an interrupted stream,
    a bare number, or an empty string when the schema requires an object).
    This function repairs such values in-place:

    * Empty / ``None`` -> ``"{}"``
    * Valid JSON object -> unchanged
    * Valid JSON but not an object (array / number / string / null) -> ``"{}"``
    * Invalid JSON -> ``"{}"``

    The original (possibly malformed) text is preserved inside a ``_raw``
    key so no information is lost for debugging, while keeping the payload
    parseable for the API.
    """
    import json as _json
    if not arguments:
        return "{}"
    try:
        parsed = _json.loads(arguments)
    except (ValueError, TypeError):
        return "{}"
    if isinstance(parsed, dict):
        return arguments
    return "{}"


def sanitize_message(msg: dict) -> dict:
    """Ensure a message dict always carries a ``content`` key and valid tool_calls.

    Two repairs are performed:

    1. Every message gets a non-null ``content`` (empty string when absent).
       The backend rejects messages missing ``content``; assistant messages
       from ``model_dump(exclude_none=True)`` drop it when it is ``None``
       (pure tool-call turns).

    2. Assistant messages with ``tool_calls`` have each tool call's
       ``arguments`` validated.  Malformed / truncated JSON arguments
       (left behind by an interrupted stream or a persisted transcript)
       cause a 400 BadRequest on the next API call, so they are repaired
       to ``"{}"`` here as a safety net.
    """
    role = msg.get("role")
    if role in ("assistant", "user", "system", "tool"):
        if "content" not in msg or msg["content"] is None:
            msg["content"] = ""
    if role == "assistant" and msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc.get("function")
            if fn and "arguments" in fn:
                fn["arguments"] = _validate_tool_args(fn["arguments"])
    return msg


def sanitize_messages(messages: list) -> list:
    """Return a copy of *messages* where every message has a ``content`` key."""
    return [sanitize_message(dict(m)) for m in messages]
