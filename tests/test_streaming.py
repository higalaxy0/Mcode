"""Streaming response wrapper tests.

Covers ``mcodecore.streaming``:
- ``StreamMessage.model_dump`` ensures assistant messages contain a ``content``
  key (corresponds to the memory backend-requires-content-field /
  sanitize-tail-only-check notes);
- ``stream_response`` aggregates content + tool_calls fragments;
- handling of stream truncation (finish_reason=None but with partial tool_call).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mcodecore import streaming


# --------------------------------------------------------------------------- #
# StreamMessage.model_dump  -- content-field fix
# --------------------------------------------------------------------------- #

def test_assistant_with_only_tool_calls_has_content_key():
    """Backend requirement: an assistant message must carry a content key even when it only has tool_calls."""
    m = streaming.StreamMessage({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "bash", "arguments": "{}"}}],
    })
    d = m.model_dump(exclude_none=True)
    assert "content" in d
    assert d["content"] == ""
    assert "tool_calls" in d


def test_assistant_with_content_keeps_content():
    m = streaming.StreamMessage({"role": "assistant", "content": "hello"})
    d = m.model_dump(exclude_none=True)
    assert d["content"] == "hello"


def test_user_message_unaffected_by_content_fix():
    """The content-field fix only targets assistant messages; user/tool are unaffected."""
    m = streaming.StreamMessage({"role": "user", "content": "hi"})
    d = m.model_dump(exclude_none=True)
    assert d["content"] == "hi"


def test_tool_message_keeps_tool_call_id():
    m = streaming.StreamMessage({"role": "tool", "tool_call_id": "c1",
                                 "content": "result"})
    d = m.model_dump(exclude_none=True)
    assert d["role"] == "tool"
    assert d["tool_call_id"] == "c1"


def test_exclude_none_false_keeps_none():
    m = streaming.StreamMessage({"role": "assistant", "content": None})
    d = m.model_dump(exclude_none=False)
    # When None is not excluded, content stays None (the fix branch is not triggered)
    assert d.get("content") is None


# --------------------------------------------------------------------------- #
# dataclass structure
# --------------------------------------------------------------------------- #

def test_tool_call_function_defaults():
    f = streaming.ToolCallFunction()
    assert f.name == ""
    assert f.arguments == ""


def test_tool_call_defaults():
    tc = streaming.ToolCall()
    assert tc.type == "function"
    assert tc.index == 0


def test_stream_response_default_empty():
    r = streaming.StreamResponse()
    assert r.choices == []
    assert r.usage is None


# --------------------------------------------------------------------------- #
# stream_response -- verify aggregation logic with a fake client
# --------------------------------------------------------------------------- #

def _chunk(content=None, tool_call=None, finish_reason=None, usage=None):
    """Build a fake OpenAI stream chunk."""
    delta = SimpleNamespace()
    delta.content = content
    delta.tool_calls = None
    if tool_call is not None:
        tc = SimpleNamespace()
        tc.id = tool_call.get("id")
        tc.index = tool_call.get("index", 0)
        tc.function = SimpleNamespace()
        tc.function.name = tool_call.get("name")
        tc.function.arguments = tool_call.get("arguments")
        delta.tool_calls = [tc]
    choice = SimpleNamespace()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = SimpleNamespace()
    chunk.usage = usage
    chunk.choices = [choice]
    return chunk


@pytest.fixture
def fake_stream(monkeypatch):
    """Returns a function that injects a predefined chunk list to mock the client."""
    def _install(chunks):
        def _create(*args, **kwargs):
            return iter(chunks)
        monkeypatch.setattr(streaming.client.chat.completions, "create", _create)
    return _install


def test_stream_response_aggregates_content(fake_stream, capsys):
    chunks = [
        _chunk(content="Hel"),
        _chunk(content="lo"),
        _chunk(finish_reason="stop", usage=SimpleNamespace(prompt_tokens=10)),
    ]
    fake_stream(chunks)
    resp = streaming.stream_response(model="m", messages=[])
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.content == "Hello"
    assert resp.usage.prompt_tokens == 10


def test_stream_response_aggregates_tool_calls(fake_stream, capsys):
    chunks = [
        _chunk(tool_call={"id": "c1", "index": 0, "name": "bash",
                          "arguments": '{"comm'}),
        _chunk(tool_call={"index": 0, "arguments": 'and":"ls"}'}),
        _chunk(finish_reason="tool_calls"),
    ]
    fake_stream(chunks)
    resp = streaming.stream_response(model="m", messages=[])
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "tool_calls"
    assert msg.tool_calls is not None and len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.id == "c1"
    assert tc.function.name == "bash"
    assert tc.function.arguments == '{"command":"ls"}'
    # content-field fix
    assert msg.model_dump(exclude_none=True)["content"] == ""


def test_stream_response_stops_with_no_tool_no_content(fake_stream):
    chunks = [_chunk(finish_reason="stop")]
    fake_stream(chunks)
    resp = streaming.stream_response(model="m", messages=[])
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.content is None


def test_stream_response_tool_calls_generate_id_when_missing(fake_stream):
    chunks = [
        _chunk(tool_call={"index": 0, "name": "read_file", "arguments": "{}"}),
        _chunk(finish_reason="tool_calls"),
    ]
    fake_stream(chunks)
    resp = streaming.stream_response(model="m", messages=[])
    tc = resp.choices[0].message.tool_calls[0]
    assert tc.id and tc.id.startswith("call_")  # fallback-generated id


def test_stream_response_sets_stream_options_default(fake_stream):
    captured = {}

    def _create(*args, **kwargs):
        captured.update(kwargs)
        return iter([_chunk(finish_reason="stop")])

    import mcodecore.streaming as sm
    sm.client.chat.completions.create = _create
    streaming.stream_response(model="m", messages=[])
    assert captured.get("stream") is True
    assert captured["stream_options"] == {"include_usage": True}


# --------------------------------------------------------------------------- #
# Transient-error classification helpers
# --------------------------------------------------------------------------- #

def test_classify_transient_rate_limit():
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    err = openai.RateLimitError("rate limited", response=resp, body=None)
    assert streaming.classify_transient(err) is True


def test_classify_transient_internal_server_error_529():
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(529, request=req)
    err = openai.InternalServerError("overloaded", response=resp, body=None)
    assert streaming.classify_transient(err) is True


def test_classify_transient_connection_error():
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    err = openai.APIConnectionError(request=req)
    assert streaming.classify_transient(err) is True


def test_classify_transient_timeout_error():
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    err = openai.APITimeoutError(request=req)
    assert streaming.classify_transient(err) is True


def test_classify_transient_non_transient_error():
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    err = openai.BadRequestError("prompt too long", response=resp, body=None)
    assert streaming.classify_transient(err) is False


def test_classify_transient_bare_timeout():
    assert streaming.classify_transient(TimeoutError("operation timed out")) is True


def test_classify_transient_bare_connection_error():
    assert streaming.classify_transient(ConnectionError("connection reset")) is True


def test_retry_after_seconds_from_header():
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(429, request=req, headers={"retry-after": "7"})
    err = openai.RateLimitError("rate limited", response=resp, body=None)
    assert streaming.retry_after_seconds(err) == 7.0


def test_retry_after_seconds_from_ms_header():
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(429, request=req, headers={"retry-after-ms": "3500"})
    err = openai.RateLimitError("rate limited", response=resp, body=None)
    assert streaming.retry_after_seconds(err) == 3.5


def test_retry_after_seconds_none_when_absent():
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(500, request=req)
    err = openai.InternalServerError("err", response=resp, body=None)
    assert streaming.retry_after_seconds(err) is None


def test_retry_after_seconds_none_for_connection_error():
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    err = openai.APIConnectionError(request=req)
    assert streaming.retry_after_seconds(err) is None


def test_backoff_delay_grows_and_capped():
    delays = [streaming.backoff_delay(i) for i in range(10)]
    # Base jitter range is [0, 0.5]; base*2**attempt.
    # delay = min(30, 2**attempt) + jitter([0,0.5])
    assert delays[0] < 1.6           # 1.0 + up to 0.5
    assert delays[5] >= 30.0          # capped at 30 + jitter
    assert delays[9] >= 30.0          # still capped


# --------------------------------------------------------------------------- #
# stream_response retry behavior
# --------------------------------------------------------------------------- #

def _patch_time_sleep(monkeypatch):
    """Patch ``time.sleep`` in the streaming module so tests don't actually wait."""
    calls = []
    monkeypatch.setattr(streaming.time, "sleep", lambda s: calls.append(s))
    return calls


def test_stream_response_retries_on_rate_limit(monkeypatch):
    """First create() raises RateLimitError, second succeeds -> aggregated result."""
    import httpx
    import openai
    sleep_calls = _patch_time_sleep(monkeypatch)
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp429 = httpx.Response(429, request=req, headers={"retry-after": "0"})
    rle = openai.RateLimitError("rate limited", response=resp429, body=None)

    call_count = {"n": 0}

    def _create(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise rle
        return iter([_chunk(content="OK"), _chunk(finish_reason="stop")])

    import mcodecore.streaming as sm
    sm.client.chat.completions.create = _create
    resp = streaming.stream_response(model="m", messages=[])
    assert resp.choices[0].message.content == "OK"
    assert resp.choices[0].finish_reason == "stop"
    assert call_count["n"] == 2
    assert len(sleep_calls) == 1


def test_stream_response_retries_on_529_overloaded(monkeypatch):
    """InternalServerError (529) should be retried."""
    import httpx
    import openai
    sleep_calls = _patch_time_sleep(monkeypatch)
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp529 = httpx.Response(529, request=req)
    ise = openai.InternalServerError("overloaded", response=resp529, body=None)

    call_count = {"n": 0}

    def _create(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ise
        return iter([_chunk(content="recovered"), _chunk(finish_reason="stop")])

    import mcodecore.streaming as sm
    sm.client.chat.completions.create = _create
    resp = streaming.stream_response(model="m", messages=[])
    assert resp.choices[0].message.content == "recovered"
    assert call_count["n"] == 2
    assert len(sleep_calls) == 1


def test_stream_response_retries_on_connection_error(monkeypatch):
    """APIConnectionError (network jitter) should be retried."""
    import httpx
    import openai
    sleep_calls = _patch_time_sleep(monkeypatch)
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    ace = openai.APIConnectionError(request=req)

    call_count = {"n": 0}

    def _create(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ace
        return iter([_chunk(content="back"), _chunk(finish_reason="stop")])

    import mcodecore.streaming as sm
    sm.client.chat.completions.create = _create
    resp = streaming.stream_response(model="m", messages=[])
    assert resp.choices[0].message.content == "back"
    assert call_count["n"] == 2


def test_stream_response_retries_midstream_break(monkeypatch):
    """A transient error raised *after* the stream has started should be retried."""
    sleep_calls = _patch_time_sleep(monkeypatch)
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp529 = httpx.Response(529, request=req)
    ise = openai.InternalServerError("overloaded", response=resp529, body=None)

    call_count = {"n": 0}

    def _create(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Emit one chunk then break mid-stream.
            return _raise_after(ise)
        return iter([_chunk(content="full"), _chunk(finish_reason="stop")])

    import mcodecore.streaming as sm
    sm.client.chat.completions.create = _create
    resp = streaming.stream_response(model="m", messages=[])
    # On retry the full content replaces the partial one.
    assert resp.choices[0].message.content == "full"
    assert call_count["n"] == 2


def test_stream_response_raises_after_exhausting_retries(monkeypatch):
    """If all retries are exhausted the last error propagates."""
    sleep_calls = _patch_time_sleep(monkeypatch)
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp429 = httpx.Response(429, request=req, headers={"retry-after": "0"})
    rle = openai.RateLimitError("rate limited", response=resp429, body=None)

    def _create(*args, **kwargs):
        raise rle

    import mcodecore.streaming as sm
    sm.client.chat.completions.create = _create
    with pytest.raises(openai.RateLimitError):
        streaming.stream_response(model="m", messages=[])
    # MAX_STREAM_RETRIES=3 means 1 initial + 3 retries = 4 total calls.
    assert len(sleep_calls) == streaming.MAX_STREAM_RETRIES


def test_stream_response_does_not_retry_non_transient(monkeypatch):
    """A non-transient error (e.g. BadRequestError) should not be retried."""
    sleep_calls = _patch_time_sleep(monkeypatch)
    import httpx
    import openai
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    bae = openai.BadRequestError("invalid request", response=resp, body=None)

    call_count = {"n": 0}

    def _create(*args, **kwargs):
        call_count["n"] += 1
        raise bae

    import mcodecore.streaming as sm
    sm.client.chat.completions.create = _create
    with pytest.raises(openai.BadRequestError):
        streaming.stream_response(model="m", messages=[])
    assert call_count["n"] == 1
    assert len(sleep_calls) == 0


def _raise_after(exc):
    """Helper: a generator that yields a valid chunk then raises *exc* on next
    iteration (simulating a mid-stream break).

    The first yielded value is a real chunk (consumed normally); when the
    caller asks for the next value the generator raises *exc*.
    """
    yield _chunk(content="partial")
    raise exc


# --------------------------------------------------------------------------- #
# stream truncation: max_tokens mid-arguments (finish_reason="length")
# and connection drop (finish_reason=None) with partial tool_calls
# --------------------------------------------------------------------------- #

def test_max_tokens_truncates_tool_call_arguments(fake_stream):
    """max_tokens hit mid-arguments -> finish_reason='length' + truncated JSON.

    The partial tool_calls must be dropped to avoid orphaned tool_calls
    (assistant msg with tool_calls but no matching tool results -> 400 on
    next turn).  finish_reason must be 'interrupted'.
    """
    chunks = [
        _chunk(tool_call={"id": "call_1", "index": 0, "name": "bash",
                          "arguments": '{"command": "dir'}),
        _chunk(finish_reason="length"),  # max_tokens hit, arguments truncated
    ]
    fake_stream(chunks)
    resp = streaming.stream_response(model="m", messages=[])
    msg = resp.choices[0].message
    # finish_reason must reflect the interruption
    assert resp.choices[0].finish_reason == "interrupted"
    # tool_calls must be dropped entirely (no orphaned tool_calls)
    assert msg.tool_calls is None or len(msg.tool_calls) == 0
    # A note must be added so the LLM knows to re-issue the call
    dumped = msg.model_dump(exclude_none=True)
    assert "interrupted" in (dumped.get("content") or "")


def test_connection_drop_truncates_tool_call_arguments(fake_stream):
    """Connection cut (finish_reason=None) + partial tool_calls -> interrupted."""
    chunks = [
        _chunk(tool_call={"id": "call_1", "index": 0, "name": "bash",
                          "arguments": '{"command": "dir'}),
        # No finish_reason chunk at all -- connection simply dropped
    ]
    fake_stream(chunks)
    resp = streaming.stream_response(model="m", messages=[])
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "interrupted"
    assert msg.tool_calls is None or len(msg.tool_calls) == 0


def test_max_tokens_on_plain_content_not_interrupted(fake_stream):
    """max_tokens on plain text (no tool_calls) -> finish_reason='length',
    not 'interrupted'.  The truncated text is kept as content."""
    chunks = [
        _chunk(content="This is a long response that got cut"),
        _chunk(finish_reason="length"),
    ]
    fake_stream(chunks)
    resp = streaming.stream_response(model="m", messages=[])
    # length is a valid finish_reason, NOT interrupted
    assert resp.choices[0].finish_reason == "length"
    assert resp.choices[0].message.content == "This is a long response that got cut"
    # No tool_calls
    assert resp.choices[0].message.tool_calls is None


def test_max_tokens_partial_content_and_tool_call(fake_stream):
    """max_tokens with both content and partial tool_call -> interrupted,
    content is preserved alongside the interrupted note."""
    chunks = [
        _chunk(content="Let me run "),
        _chunk(tool_call={"id": "call_1", "index": 0, "name": "bash",
                          "arguments": '{"command": "dir'}),
        _chunk(finish_reason="length"),
    ]
    fake_stream(chunks)
    resp = streaming.stream_response(model="m", messages=[])
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "interrupted"
    # Content is preserved
    assert "Let me run" in (msg.content or "")
    # Tool calls dropped
    assert msg.tool_calls is None or len(msg.tool_calls) == 0
    # Interrupted note appended
    assert "interrupted" in (msg.content or "")


def test_completed_tool_calls_not_interrupted(fake_stream):
    """finish_reason='tool_calls' with complete arguments -> not interrupted,
    tool_calls are kept intact."""
    chunks = [
        _chunk(tool_call={"id": "call_1", "index": 0, "name": "bash",
                          "arguments": '{"command": "dir /b"}'}),
        _chunk(finish_reason="tool_calls"),
    ]
    fake_stream(chunks)
    resp = streaming.stream_response(model="m", messages=[])
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "tool_calls"
    assert msg.tool_calls is not None and len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.arguments == '{"command": "dir /b"}'


def test_max_tokens_multiple_partial_tool_calls_all_dropped(fake_stream):
    """max_tokens with multiple partial tool_calls -> all dropped."""
    chunks = [
        _chunk(tool_call={"id": "call_1", "index": 0, "name": "bash",
                          "arguments": '{"command": "dir'}),
        _chunk(tool_call={"id": "call_2", "index": 1, "name": "read_file",
                          "arguments": '{"path": "test'}),
        _chunk(finish_reason="length"),
    ]
    fake_stream(chunks)
    resp = streaming.stream_response(model="m", messages=[])
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "interrupted"
    assert msg.tool_calls is None or len(msg.tool_calls) == 0

