"""Streaming response wrappers.

Module-level dataclasses that mimic the OpenAI SDK return objects,
so the rest of the codebase can call ``.model_dump(exclude_none=True)``,
``.choices[0].message``, etc. without depending on the SDK's pydantic models.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import Any

from .config import client


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


def stream_response(**kwargs) -> StreamResponse:
    """Consume an OpenAI streaming response and aggregate it into a :class:`StreamResponse`.

    Behavior:
    - Print content to stdout in real time.
    - Accumulate tool_call fragments.
    - When the stream is cut off (finish_reason None but partial tool_calls),
      mark as interrupted.
    - Assistant message content is always ``""`` rather than ``None``.
    """
    content_parts: list[str] = []
    tool_calls_parts: dict[int, dict] = {}
    finish_reason: str | None = None
    interrupted = False
    _cli_prefix_printed = False
    stream_usage = None

    kwargs.setdefault("stream_options", {"include_usage": True})
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

    full_content = "".join(content_parts)
    msg_dict: dict = {"role": "assistant", "content": full_content or ""}
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
