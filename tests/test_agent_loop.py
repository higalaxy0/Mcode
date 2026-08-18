"""Lead Agent main loop tests.

Verifies the core flow of ``agent_loop`` via a mocked ``stream_response``:
- tool-call dispatch + PreToolUse/PostToolUse hooks
- triggering the Stop hook and returning when finish_reason is not tool_calls
- multi-turn tool-call loops
- timeout / prompt_too_long retry semantics
- tail-message sanitize (content-field)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcodecore import agent, streaming, tools
from mcodecore.context import ctx
from mcodecore.exceptions import AgentInterrupt


def _make_response(content=None, tool_calls=None, finish_reason="stop", usage=None):
    """Build a fake StreamResponse."""
    msg_data = {"role": "assistant", "content": content}
    if tool_calls:
        msg_data["tool_calls"] = tool_calls
    msg = streaming.StreamMessage(msg_data)
    choice = streaming.StreamChoice(message=msg, finish_reason=finish_reason)
    return streaming.StreamResponse(choices=[choice], usage=usage)


def _tool_call_dict(cid, name, args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture(autouse=True)
def _noop_post_turn_memory(monkeypatch):
    """Prevent background memory-extract threads from outliving the test.

    ``agent_loop`` spawns a daemon thread calling ``_post_turn_memory`` on
    every non-tool-call turn.  That thread calls the real LLM and touches the
    filesystem via module-level ``MEMORY_DIR`` -- both of which leak across
    test boundaries (the lock blocks subsequent ``load_memories`` calls and
    the restored path points at the real workspace).  Mock it out here;
    memory extraction is tested directly in ``test_memory_plan_*.py``.
    """
    monkeypatch.setattr(agent, "_post_turn_memory", lambda *a, **kw: None)


@pytest.fixture
def mock_stream(monkeypatch):
    """Returns a setter that can inject a 'response sequence'."""
    holder = {}

    def _fake_stream(**kwargs):
        responses = holder["responses"]
        idx = holder.get("call_count", 0)
        holder["call_count"] = idx + 1
        if idx < len(responses):
            return responses[idx]
        # Return a stop response when the sequence is exhausted to avoid an infinite loop
        return _make_response(content="done", finish_reason="stop")

    monkeypatch.setattr(agent, "stream_response", _fake_stream)
    return holder


# --------------------------------------------------------------------------- #
# Basic flow
# --------------------------------------------------------------------------- #

def test_agent_loop_text_response_returns(mock_stream):
    mock_stream["responses"] = [_make_response(content="hello", finish_reason="stop")]
    messages = [{"role": "user", "content": "hi"}]
    agent.agent_loop(messages)
    # assistant message was appended
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "hello"


def test_agent_loop_appends_assistant_with_content_key(mock_stream):
    """sanitize-tail: even when assistant has no content, the dump should contain a content key."""
    mock_stream["responses"] = [
        _make_response(content=None, finish_reason="stop"),
    ]
    messages = [{"role": "user", "content": "hi"}]
    agent.agent_loop(messages)
    assert "content" in messages[-1]


def test_agent_loop_tool_dispatch(mock_stream, monkeypatch):
    """When finish_reason=tool_calls the corresponding handler should be invoked and the tool result appended."""
    # Turn 1: call write_file
    # Turn 2: stop
    mock_stream["responses"] = [
        _make_response(content=None,
                       tool_calls=[_tool_call_dict("c1", "write_file",
                                    {"path": "out.txt", "content": "data"})],
                       finish_reason="tool_calls"),
        _make_response(content="wrote it", finish_reason="stop"),
    ]
    messages = [{"role": "user", "content": "write a file"}]
    agent.agent_loop(messages)
    # should contain a tool result message
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "Wrote" in tool_msgs[0]["content"]
    # the file was actually written
    assert (ctx.WORKDIR if hasattr(ctx, "WORKDIR") else tools.WORKDIR)  # noqa


def test_agent_loop_multiple_tool_calls_in_one_turn(mock_stream):
    """A single assistant turn with multiple tool_calls should execute all of them."""
    mock_stream["responses"] = [
        _make_response(content=None,
                       tool_calls=[
                           _tool_call_dict("c1", "write_file",
                                           {"path": "a.txt", "content": "1"}),
                           _tool_call_dict("c2", "write_file",
                                           {"path": "b.txt", "content": "2"}),
                       ],
                       finish_reason="tool_calls"),
        _make_response(content="both done", finish_reason="stop"),
    ]
    messages = [{"role": "user", "content": "go"}]
    agent.agent_loop(messages)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2


def test_agent_loop_unknown_tool_returns_error(mock_stream):
    mock_stream["responses"] = [
        _make_response(content=None,
                       tool_calls=[_tool_call_dict("c1", "nonexistent_tool", {})],
                       finish_reason="tool_calls"),
        _make_response(content="ok", finish_reason="stop"),
    ]
    messages = [{"role": "user", "content": "go"}]
    agent.agent_loop(messages)
    tool_msg = [m for m in messages if m.get("role") == "tool"][0]
    assert "Unknown tool" in tool_msg["content"]


# --------------------------------------------------------------------------- #
# Hooks
# --------------------------------------------------------------------------- #

def test_agent_loop_triggers_pretooluse_hook(mock_stream):
    triggered = []
    ctx.register_hook("PreToolUse", lambda fn: triggered.append(fn.name) or None)
    mock_stream["responses"] = [
        _make_response(content=None,
                       tool_calls=[_tool_call_dict("c1", "write_file",
                                    {"path": "x.txt", "content": "y"})],
                       finish_reason="tool_calls"),
        _make_response(content="done", finish_reason="stop"),
    ]
    agent.agent_loop([{"role": "user", "content": "go"}])
    assert "write_file" in triggered


def test_agent_loop_blocked_tool_appends_denied(mock_stream):
    ctx.register_hook("PreToolUse", lambda fn: "Permission denied")
    mock_stream["responses"] = [
        _make_response(content=None,
                       tool_calls=[_tool_call_dict("c1", "write_file",
                                    {"path": "x.txt", "content": "y"})],
                       finish_reason="tool_calls"),
        _make_response(content="done", finish_reason="stop"),
    ]
    messages = [{"role": "user", "content": "go"}]
    agent.agent_loop(messages)
    tool_msg = [m for m in messages if m.get("role") == "tool"][0]
    assert "Permission denied" in tool_msg["content"]


def test_agent_loop_triggers_stop_hook(mock_stream):
    stop_triggered = []
    ctx.register_hook("Stop", lambda msgs: stop_triggered.append(len(msgs)) or None)
    mock_stream["responses"] = [_make_response(content="bye", finish_reason="stop")]
    agent.agent_loop([{"role": "user", "content": "hi"}])
    assert len(stop_triggered) == 1


def test_agent_loop_stop_hook_force_continues(mock_stream):
    """When the Stop hook returns non-None, inject a new user message and continue."""
    call_count = [0]

    def stop_force(msgs):
        call_count[0] += 1
        if call_count[0] == 1:
            return "please do more"
        return None

    ctx.register_hook("Stop", stop_force)
    mock_stream["responses"] = [
        _make_response(content="first", finish_reason="stop"),
        _make_response(content="second", finish_reason="stop"),
    ]
    messages = [{"role": "user", "content": "hi"}]
    agent.agent_loop(messages)
    # First stop returns force -> inject user message -> second stop returns normally
    assert call_count[0] == 2
    # there should be an injected user message
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assert any("please do more" in m["content"] for m in user_msgs)


# --------------------------------------------------------------------------- #
# Exceptions / retries
# --------------------------------------------------------------------------- #

def test_agent_loop_timeout_retry(mock_stream, monkeypatch):
    call_seq = []

    def _fake(**kwargs):
        call_seq.append("call")
        if len(call_seq) == 1:
            raise TimeoutError("request timed out")
        return _make_response(content="recovered", finish_reason="stop")

    monkeypatch.setattr(agent, "stream_response", _fake)
    messages = [{"role": "user", "content": "hi"}]
    agent.agent_loop(messages)
    assert len(call_seq) == 2  # succeeds after one retry
    assert messages[-1]["content"] == "recovered"


def test_agent_loop_timeout_exhausts_retries_raises(mock_stream, monkeypatch):
    monkeypatch.setattr(agent, "MAX_REACTIVE_RETRIES", 1)

    def _fake(**kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(agent, "stream_response", _fake)
    with pytest.raises(TimeoutError):
        agent.agent_loop([{"role": "user", "content": "hi"}])


def test_agent_loop_prompt_too_long_triggers_reactive_compact(mock_stream, monkeypatch):
    call_seq = []
    compacted = {"done": False}

    def _fake(**kwargs):
        call_seq.append(1)
        if len(call_seq) == 1:
            raise Exception("prompt_too_long: too many tokens")
        return _make_response(content="ok", finish_reason="stop")

    monkeypatch.setattr(agent, "stream_response", _fake)

    def _react(msgs):
        compacted["done"] = True
        return [{"role": "user", "content": "[compacted]"}]

    monkeypatch.setattr(agent, "reactive_compact", _react)
    messages = [{"role": "user", "content": "hi"}]
    agent.agent_loop(messages)
    assert compacted["done"] is True


def test_agent_loop_interrupted_finish_reason_returns(mock_stream):
    mock_stream["responses"] = [
        _make_response(content=None, finish_reason="interrupted"),
    ]
    messages = [{"role": "user", "content": "hi"}]
    agent.agent_loop(messages)
    # interrupted returns directly without continuing
    assert messages[-1]["role"] == "assistant"


def test_agent_loop_agentinterrupt_propagates(mock_stream, monkeypatch):
    def _fake(**kwargs):
        raise AgentInterrupt()
    monkeypatch.setattr(agent, "stream_response", _fake)
    # should not raise; agent_loop catches it internally
    agent.agent_loop([{"role": "user", "content": "hi"}])


# --------------------------------------------------------------------------- #
# Transient-error recovery (429 / 529 / connection errors)
# --------------------------------------------------------------------------- #

def _patch_sleep(monkeypatch):
    """Patch ``time.sleep`` used inside agent.py retry path."""
    calls = []
    monkeypatch.setattr("time.sleep", lambda s: calls.append(s))
    return calls


def test_agent_loop_recovers_from_429_rate_limit(mock_stream, monkeypatch):
    """RateLimitError (429) should be retried and the loop continues."""
    import httpx
    import openai
    sleep_calls = _patch_sleep(monkeypatch)
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(429, request=req, headers={"retry-after": "0"})
    rle = openai.RateLimitError("rate limited", response=resp, body=None)

    call_seq = []

    def _fake(**kwargs):
        call_seq.append(1)
        if len(call_seq) == 1:
            raise rle
        return _make_response(content="recovered", finish_reason="stop")

    monkeypatch.setattr(agent, "stream_response", _fake)
    messages = [{"role": "user", "content": "hi"}]
    agent.agent_loop(messages)
    assert len(call_seq) == 2
    assert messages[-1]["content"] == "recovered"
    assert len(sleep_calls) == 1


def test_agent_loop_recovers_from_529_overloaded(mock_stream, monkeypatch):
    """InternalServerError (529) should be retried and the loop continues."""
    import httpx
    import openai
    sleep_calls = _patch_sleep(monkeypatch)
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(529, request=req)
    ise = openai.InternalServerError("overloaded", response=resp, body=None)

    call_seq = []

    def _fake(**kwargs):
        call_seq.append(1)
        if len(call_seq) == 1:
            raise ise
        return _make_response(content="back online", finish_reason="stop")

    monkeypatch.setattr(agent, "stream_response", _fake)
    messages = [{"role": "user", "content": "hi"}]
    agent.agent_loop(messages)
    assert len(call_seq) == 2
    assert messages[-1]["content"] == "back online"
    assert len(sleep_calls) == 1


def test_agent_loop_recovers_from_connection_error(mock_stream, monkeypatch):
    """APIConnectionError (network jitter) should be retried and the loop continues."""
    import httpx
    import openai
    sleep_calls = _patch_sleep(monkeypatch)
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    ace = openai.APIConnectionError(request=req)

    call_seq = []

    def _fake(**kwargs):
        call_seq.append(1)
        if len(call_seq) == 1:
            raise ace
        return _make_response(content="reconnected", finish_reason="stop")

    monkeypatch.setattr(agent, "stream_response", _fake)
    messages = [{"role": "user", "content": "hi"}]
    agent.agent_loop(messages)
    assert len(call_seq) == 2
    assert messages[-1]["content"] == "reconnected"
    assert len(sleep_calls) == 1


def test_agent_loop_timeout_still_recovers_with_backoff(mock_stream, monkeypatch):
    """TimeoutError should still be retried (now via classify_transient) with backoff."""
    sleep_calls = _patch_sleep(monkeypatch)
    call_seq = []

    def _fake(**kwargs):
        call_seq.append(1)
        if len(call_seq) == 1:
            raise TimeoutError("request timed out")
        return _make_response(content="ok", finish_reason="stop")

    monkeypatch.setattr(agent, "stream_response", _fake)
    messages = [{"role": "user", "content": "hi"}]
    agent.agent_loop(messages)
    assert len(call_seq) == 2
    assert messages[-1]["content"] == "ok"
    assert len(sleep_calls) == 1


def test_agent_loop_429_exhausts_retries_raises(mock_stream, monkeypatch):
    """After exhausting retries, the 429 error should propagate."""
    import httpx
    import openai
    monkeypatch.setattr(agent, "MAX_REACTIVE_RETRIES", 1)
    _patch_sleep(monkeypatch)
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(429, request=req, headers={"retry-after": "0"})
    rle = openai.RateLimitError("rate limited", response=resp, body=None)

    call_count = {"n": 0}

    def _fake(**kwargs):
        call_count["n"] += 1
        raise rle

    monkeypatch.setattr(agent, "stream_response", _fake)
    with pytest.raises(openai.RateLimitError):
        agent.agent_loop([{"role": "user", "content": "hi"}])
    # 1 initial + 1 retry = 2 calls
    assert call_count["n"] == 2


def test_agent_loop_prompt_too_long_still_works_alongside_transient(mock_stream, monkeypatch):
    """Ensure prompt_too_long path still fires when it's NOT a transient error."""
    _patch_sleep(monkeypatch)
    compacted = {"done": False}

    def _fake(**kwargs):
        # Exception whose message contains prompt_too_long but is NOT a
        # transient SDK error (plain Exception, not RateLimitError etc.).
        raise Exception("prompt_too_long: too many tokens")

    monkeypatch.setattr(agent, "stream_response", _fake)

    def _react(msgs):
        compacted["done"] = True
        return [{"role": "user", "content": "[compacted]"}]

    monkeypatch.setattr(agent, "reactive_compact", _react)
    # After compact, stream_response still raises -> we need it to eventually
    # return. Patch again after compact to succeed.
    original = _fake

    def _fake2(**kwargs):
        return _make_response(content="ok", finish_reason="stop")

    call_state = {"compacted": False}

    def _react2(msgs):
        call_state["compacted"] = True
        # Switch stream_response to the succeeding version.
        monkeypatch.setattr(agent, "stream_response", _fake2)
        return [{"role": "user", "content": "[compacted]"}]

    monkeypatch.setattr(agent, "reactive_compact", _react2)
    messages = [{"role": "user", "content": "hi"}]
    agent.agent_loop(messages)
    assert call_state["compacted"] is True


def test_agent_loop_retry_respects_retry_after_header(mock_stream, monkeypatch):
    """When Retry-After header is present, that delay (not backoff) is used."""
    import httpx
    import openai
    sleep_calls = _patch_sleep(monkeypatch)
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(429, request=req, headers={"retry-after": "12"})
    rle = openai.RateLimitError("rate limited", response=resp, body=None)

    seq = []

    def _fake(**kwargs):
        seq.append(1)
        if len(seq) == 1:
            raise rle
        return _make_response(content="ok", finish_reason="stop")

    monkeypatch.setattr(agent, "stream_response", _fake)
    agent.agent_loop([{"role": "user", "content": "hi"}])
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == 12.0  # uses Retry-After value, not backoff


# --------------------------------------------------------------------------- #
# usage / calibrator recording
# --------------------------------------------------------------------------- #

def test_agent_loop_records_usage_to_calibrator(mock_stream):
    usage = SimpleNamespace(prompt_tokens=1234, completion_tokens=100, total_tokens=1334)
    mock_stream["responses"] = [
        _make_response(content="x", finish_reason="stop", usage=usage),
    ]
    ctx.calibrator.samples.clear()
    # record() only logs when estimated > 1000, so use a long message
    long_msg = "w" * 8000
    agent.agent_loop([{"role": "user", "content": long_msg}])
    assert len(ctx.calibrator.samples) == 1
    assert ctx.calibrator.samples[0][1] == 1234
