"""Streaming response wrappers.

Module-level dataclasses that mimic the OpenAI SDK return objects,
so the rest of the codebase can call ``.model_dump(exclude_none=True)``,
``.choices[0].message``, etc. without depending on the SDK's pydantic models.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import openai

from .config import client

# Maximum number of mcode-level retries for transient stream errors
# (429 / 5xx / connection errors) on top of the SDK's own max_retries.
MAX_STREAM_RETRIES: int = 3


# --------------------------------------------------------------------------- #
# Transient-error classification & backoff helpers
# --------------------------------------------------------------------------- #

def classify_transient(exc: BaseException) -> bool:
    """Return ``True`` if *exc* represents a transient error worth retrying.

    Covers:
    - ``openai.RateLimitError`` (HTTP 429)
    - ``openai.InternalServerError`` (HTTP 500/502/503/529)
    - ``openai.APIConnectionError`` (network jitter, DNS, TCP reset;
      ``APITimeoutError`` is a subclass)
    - Bare ``TimeoutError`` / ``ConnectionError`` and look-alike messages
      via string fallback.
    """
    if isinstance(exc, (openai.RateLimitError,
                        openai.InternalServerError,
                        openai.APIConnectionError)):
        return True
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return any(kw in name or kw in msg for kw in (
        "timeout", "timed out", "connection",
        "overloaded", "rate limit", "429", "529"))


def retry_after_seconds(exc: BaseException) -> float | None:
    """Parse the ``Retry-After`` header from an SDK exception.

    Returns the delay in seconds, or ``None`` when the header is absent
    (caller should fall back to exponential backoff).
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    # Non-standard retry-after-ms (milliseconds, more precise).
    ms = headers.get("retry-after-ms")
    if ms is not None:
        try:
            return float(ms) / 1000
        except (TypeError, ValueError):
            pass
    # Standard retry-after (seconds, may be a float).
    ra = headers.get("retry-after")
    if ra is not None:
        try:
            return float(ra)
        except (TypeError, ValueError):
            pass
    return None


def backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter: ``min(cap, base * 2**attempt) + jitter``.

    Parameters
    ----------
    attempt:
        Zero-based retry attempt number.
    """
    base = 1.0
    cap = 30.0
    delay = min(cap, base * (2 ** attempt))
    return delay + random.uniform(0, 0.5)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class ToolCallFunction:
    """The ``function`` part of a tool call."""
    name: str = ""
    arguments: str = ""


@dataclass
class ToolCall:
    """A single tool call."""
    id: str = ""
    type: str = "function"
    function: ToolCallFunction = field(default_factory=ToolCallFunction)
    index: int = 0


@dataclass
class StreamMessage:
    """An assistant message aggregated from a streaming response.

    ``model_dump`` mirrors the OpenAI SDK ``ChatCompletionMessage`` interface
    and ensures assistant messages always carry a ``content`` key (the backend
    requires it).
    """
    _d: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.content = self._d.get("content")
        tcs = self._d.get("tool_calls")
        self.tool_calls = [ToolCall(id=tc.get("id", ""),
                                    type=tc.get("type", "function"),
                                    function=ToolCallFunction(
                                        name=tc.get("function", {}).get("name", ""),
                                        arguments=tc.get("function", {}).get("arguments", "")),
                                    index=tc.get("index", 0)) for tc in tcs] if tcs else None

    def model_dump(self, exclude_none: bool = True) -> dict:
        """Return the message dict; assistant messages are guaranteed a ``content`` key."""
        out = {k: v for k, v in self._d.items() if v is not None} if exclude_none else dict(self._d)
        if out.get("role") == "assistant" and "content" not in out:
            out["content"] = ""
        return out


@dataclass
class StreamChoice:
    """A single choice (message + finish_reason)."""
    message: StreamMessage
    finish_reason: str | None


@dataclass
class StreamResponse:
    """A complete streaming response."""
    choices: list[StreamChoice] = field(default_factory=list)
    usage: Any = None


# --------------------------------------------------------------------------- #
# stream_response
# --------------------------------------------------------------------------- #

def stream_response(**kwargs) -> StreamResponse:
    """Consume an OpenAI streaming response and aggregate it into a :class:`StreamResponse`.

    Behavior:
    - Print content to stdout in real time.
    - Accumulate tool_call fragments.
    - When the stream is cut off (finish_reason None but partial tool_calls),
      mark as interrupted.
    - Retry on transient errors (429 / 5xx / connection) up to
      :data:`MAX_STREAM_RETRIES` times with exponential backoff, respecting
      the ``Retry-After`` header when present.
    - Assistant message content is always ``""`` rather than ``None``.
    """
    kwargs.setdefault("stream_options", {"include_usage": True})

    for attempt in range(MAX_STREAM_RETRIES + 1):
        # Reset accumulation state for each attempt.
        content_parts: list[str] = []
        tool_calls_parts: dict[int, dict] = {}
        finish_reason: str | None = None
        interrupted = False
        _cli_prefix_printed = False
        stream_usage = None
        try:
            stream = client.chat.completions.create(stream=True, **kwargs)
            for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    stream_usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if not _cli_prefix_printed:
                        sys.stdout.write("\nMcode: ")
                        sys.stdout.flush()
                        _cli_prefix_printed = True
                    sys.stdout.write(delta.content)
                    sys.stdout.flush()
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_parts:
                            tool_calls_parts[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_parts[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls_parts[idx]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls_parts[idx]["arguments"] += tc.function.arguments
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            if _cli_prefix_printed:
                sys.stdout.write("\n")
                sys.stdout.flush()

            # Detect interrupted / truncated stream: we accumulated partial
            # tool_call fragments but the stream did not finish with
            # finish_reason == "tool_calls".  This covers two cases:
            #   1. Connection cut off (finish_reason is None).
            #   2. max_tokens hit mid-arguments (finish_reason == "length").
            # In both cases the arguments string is almost certainly truncated
            # JSON; storing it would cause a 400 BadRequest on the next turn
            # (orphaned tool_calls with no matching tool results).  Mark as
            # interrupted so the partial tool_calls are dropped (see the
            # ``not interrupted`` guard below).
            if finish_reason != "tool_calls" and tool_calls_parts:
                interrupted = True

            full_content = "".join(content_parts)
            msg_dict: dict = {"role": "assistant"}
            if full_content:
                msg_dict["content"] = full_content
            if interrupted:
                # Partial tool_calls were dropped; leave a note so the LLM
                # knows the previous tool call was truncated and should be
                # re-issued rather than continued.
                msg_dict["content"] = (msg_dict.get("content") or "") + \
                    "\n[stream was interrupted; partial tool call discarded]"
            if tool_calls_parts and not interrupted:
                msg_dict["tool_calls"] = [
                    {
                        "id": tool_calls_parts[idx]["id"] or f"call_{idx}_{random.randint(0, 999999):06d}",
                        "type": "function",
                        "function": {
                            "name": tool_calls_parts[idx]["name"],
                            "arguments": tool_calls_parts[idx]["arguments"],
                        },
                    }
                    for idx in sorted(tool_calls_parts.keys())
                ]

            msg = StreamMessage(msg_dict)
            if interrupted:
                fr = "interrupted"
            else:
                fr = finish_reason or ("tool_calls" if tool_calls_parts else "stop")
            return StreamResponse(choices=[StreamChoice(msg, fr)], usage=stream_usage)
        except Exception as e:
            if attempt < MAX_STREAM_RETRIES and classify_transient(e):
                # Flush any partial output before retrying.
                if _cli_prefix_printed:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                delay = retry_after_seconds(e)
                if delay is None:
                    delay = backoff_delay(attempt)
                print(f"\033[33m[stream retry {attempt + 1}/{MAX_STREAM_RETRIES} "
                      f"after {delay:.1f}s: {type(e).__name__}]\033[0m")
                time.sleep(delay)
                continue
            raise
    # Unreachable: the loop either returns or raises.
    raise RuntimeError("unreachable")  # pragma: no cover
