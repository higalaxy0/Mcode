"""Tests for context compaction and memory loading fixes.

Covers:
  P0: Teamagent run loop compaction (auto-compact + reactive compact)
  P1: Subagent loop compaction (auto-compact + reactive compact)
  P2: Subagent memory loading

These tests use the same ScriptedClient pattern as test_teammate_e2e_real.py
to mock LLM responses deterministically without network calls.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from mcodecore import tools  # must be imported first to avoid circular import
from mcodecore import teammates, bus, tasks as task_mod, subagent
from mcodecore.bus import run_send_message
from mcodecore.config import client as real_client, CONTEXT_LIMIT
from mcodecore.context import ctx
from mcodecore.compact import estimate_tokens_messages


# --------------------------------------------------------------------------- #
# Fake LLM objects (same pattern as test_teammate_e2e_real.py)
# --------------------------------------------------------------------------- #

class FakeToolCall:
    def __init__(self, cid, name, arguments="{}"):
        self.id = cid
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=arguments)
        self.index = 0

    def model_dump(self, exclude_none=True):
        return {"id": self.id, "type": self.type,
                "function": {"name": self.function.name,
                             "arguments": self.function.arguments}}


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant"}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": tc.type,
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        return d


class FakeChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, message, finish_reason):
        self.choices = [FakeChoice(message, finish_reason)]
        self.usage = None


class ScriptedClient:
    """Fake ``client.chat.completions.create`` returning scripted responses."""

    def __init__(self, script):
        self._script = list(script)
        self._idx = 0
        self._lock = threading.Lock()
        self.calls = []

    def create(self, *args, **kwargs):
        self.calls.append(kwargs)
        with self._lock:
            if self._idx >= len(self._script):
                raise StopIteration("script exhausted")
            item = self._script[self._idx]
            self._idx += 1
        if callable(item) and not isinstance(item, BaseException):
            item = item()
        if isinstance(item, BaseException):
            raise item
        return item


def _install_fake_client(monkeypatch, script):
    fake = ScriptedClient(script)
    monkeypatch.setattr(real_client.chat.completions, "create", fake.create)
    return fake


def _msg(content):
    return FakeMessage(content=content)


def _tc(cid, name, args=None):
    return FakeToolCall(cid, name, json.dumps(args or {}))


def _resp_tool(msg_content, tool_calls):
    return FakeResponse(
        FakeMessage(content=msg_content, tool_calls=tool_calls),
        finish_reason="tool_calls")


def _resp_stop(content):
    return FakeResponse(_msg(content), finish_reason="stop")


def _wait_teammate(name, timeout=10.0):
    evt = ctx.active_teammates.get(name)
    assert evt is not None, f"teammate {name} not registered"
    assert evt.wait(timeout=timeout), \
        f"teammate {name} did not finish in {timeout}s"
    time.sleep(0.05)


def _lead_inbox():
    return ctx.bus.read_inbox("lead")


def _find_msg(inbox, msg_type=None, from_agent=None, content_contains=None):
    results = []
    for m in inbox:
        if msg_type is not None and m.get("type") != msg_type:
            continue
        if from_agent is not None and m.get("from") != from_agent:
            continue
        if content_contains is not None and content_contains not in str(m.get("content", "")):
            continue
        results.append(m)
    return results


@pytest.fixture(autouse=True)
def _fast_idle(monkeypatch):
    monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)
    import mcodecore.teammates as _tm2
    monkeypatch.setattr(_tm2.time, "sleep", lambda s: None)


# --------------------------------------------------------------------------- #
# P0: Teamagent compaction
# --------------------------------------------------------------------------- #

def test_teammate_auto_compact_on_large_context(monkeypatch):
    """When the teammate's context exceeds CONTEXT_LIMIT, auto-compact is
    triggered before the LLM call.  The teammate still finishes normally.

    We force a huge context by lowering CONTEXT_LIMIT in the teammates
    module's namespace, then adding many messages to the teammate's
    conversation via inbox flooding.

    Note: compact_history calls summarize_history which itself calls the
    LLM, so the script must include a response for the summarization call.
    """
    import mcodecore.teammates as _tm

    # Temporarily lower CONTEXT_LIMIT so compaction triggers quickly
    monkeypatch.setattr(_tm, "CONTEXT_LIMIT", 100)

    # Mock compact_history to verify it gets called
    compact_called = threading.Event()
    original_compact = _tm.compact_history

    def _tracking_compact(messages):
        compact_called.set()
        return original_compact(messages)

    monkeypatch.setattr(_tm, "compact_history", _tracking_compact)

    # The teammate needs enough turns to build up a large context.
    # Each inbox message is injected as a user message, inflating context.
    # We send a large message to inflate context on first read.
    _install_fake_client(monkeypatch, [
        # summarize_history inside compact_history
        _resp_stop("summary"),
        # actual teammate loop -> done
        _resp_stop("done"),
    ])

    teammates.spawn_teammate_thread("p0_compact", "worker",
                                    "x" * 500)  # large prompt
    _wait_teammate("p0_compact")

    assert compact_called.is_set(), \
        "compact_history should have been called when context exceeded limit"


def test_teammate_reactive_compact_on_prompt_too_long(monkeypatch):
    """When the LLM returns a prompt_too_long error, the teammate
    reactively compacts and retries, then succeeds.

    Note: reactive_compact calls summarize_history which itself calls the
    LLM, so the script must include a response for the summarization call.
    """
    import mcodecore.teammates as _tm

    # Lower CONTEXT_LIMIT so auto-compact check passes but LLM still
    # returns prompt_too_long (simulating a model with lower limit)
    monkeypatch.setattr(_tm, "CONTEXT_LIMIT", 999999)

    reactive_called = threading.Event()
    original_reactive = _tm.reactive_compact

    def _tracking_reactive(messages):
        reactive_called.set()
        return original_reactive(messages)

    monkeypatch.setattr(_tm, "reactive_compact", _tracking_reactive)

    _install_fake_client(monkeypatch, [
        # 1st call: actual teammate loop -> prompt_too_long
        Exception("Error: prompt_too_long: input length exceeds limit"),
        # 2nd call: summarize_history inside reactive_compact
        _resp_stop("summary of conversation"),
        # 3rd call: actual teammate loop -> success
        _resp_stop("recovered after compaction"),
    ])

    teammates.spawn_teammate_thread("p0_reactive", "worker", "hello")
    _wait_teammate("p0_reactive")

    assert reactive_called.is_set(), \
        "reactive_compact should have been called on prompt_too_long error"

    lead_msgs = _lead_inbox()
    results = _find_msg(lead_msgs, msg_type="result", from_agent="p0_reactive")
    assert len(results) == 1
    assert "recovered after compaction" in results[0]["content"]


def test_teammate_compaction_does_not_break_normal_flow(monkeypatch):
    """When context is small (below CONTEXT_LIMIT), no compaction occurs
    and the teammate works normally.  This is a regression guard to ensure
    the compaction code path doesn't interfere with normal operation."""
    import mcodecore.teammates as _tm

    compact_called = threading.Event()

    def _tracking_compact(messages):
        compact_called.set()
        return messages

    monkeypatch.setattr(_tm, "compact_history", _tracking_compact)

    _install_fake_client(monkeypatch, [
        _resp_tool(None, [_tc("c1", "bash", {"command": "echo hi"})]),
        _resp_stop("task complete"),
    ])

    teammates.spawn_teammate_thread("p0_normal", "worker", "do work")
    _wait_teammate("p0_normal")

    assert not compact_called.is_set(), \
        "compact_history should NOT be called when context is small"

    lead_msgs = _lead_inbox()
    results = _find_msg(lead_msgs, msg_type="result", from_agent="p0_normal")
    assert len(results) == 1
    assert "task complete" in results[0]["content"]


# --------------------------------------------------------------------------- #
# P1: Subagent compaction
# --------------------------------------------------------------------------- #

def test_subagent_auto_compact_on_large_context(monkeypatch):
    """When the subagent's context exceeds CONTEXT_LIMIT, auto-compact is
    triggered before the LLM call.  The subagent still returns a result.

    Note: compact_history calls summarize_history which itself calls the
    LLM, so the script must include a response for the summarization call.
    """
    import mcodecore.subagent as _sa

    # Lower CONTEXT_LIMIT so compaction triggers
    monkeypatch.setattr(_sa, "CONTEXT_LIMIT", 100)

    compact_called = threading.Event()
    original_compact = _sa.compact_history

    def _tracking_compact(messages):
        compact_called.set()
        return original_compact(messages)

    monkeypatch.setattr(_sa, "compact_history", _tracking_compact)

    _install_fake_client(monkeypatch, [
        # summarize_history inside compact_history
        _resp_stop("summary"),
        # actual subagent loop -> result
        _resp_stop("subagent result"),
    ])

    result = subagent.spawn_subagent("x" * 500)

    assert compact_called.is_set(), \
        "compact_history should have been called when context exceeded limit"
    assert "subagent result" in result


def test_subagent_reactive_compact_on_prompt_too_long(monkeypatch):
    """When the LLM returns a prompt_too_long error, the subagent
    reactively compacts and retries, then succeeds.

    Note: reactive_compact calls summarize_history which itself calls the
    LLM, so the script must include a response for the summarization call.
    """
    import mcodecore.subagent as _sa

    monkeypatch.setattr(_sa, "CONTEXT_LIMIT", 999999)

    reactive_called = threading.Event()
    original_reactive = _sa.reactive_compact

    def _tracking_reactive(messages):
        reactive_called.set()
        return original_reactive(messages)

    monkeypatch.setattr(_sa, "reactive_compact", _tracking_reactive)

    _install_fake_client(monkeypatch, [
        # 1st call: actual subagent loop -> prompt_too_long
        Exception("Error: prompt_too_long: input length exceeds limit"),
        # 2nd call: summarize_history inside reactive_compact
        _resp_stop("summary of conversation"),
        # 3rd call: actual subagent loop -> success
        _resp_stop("recovered"),
    ])

    result = subagent.spawn_subagent("test subtask")

    assert reactive_called.is_set(), \
        "reactive_compact should have been called on prompt_too_long error"
    assert "recovered" in result


def test_subagent_compaction_does_not_break_normal_flow(monkeypatch):
    """When context is small, no compaction occurs and subagent works
    normally.  Regression guard."""
    import mcodecore.subagent as _sa

    compact_called = threading.Event()

    def _tracking_compact(messages):
        compact_called.set()
        return messages

    monkeypatch.setattr(_sa, "compact_history", _tracking_compact)

    _install_fake_client(monkeypatch, [
        _resp_stop("normal result"),
    ])

    result = subagent.spawn_subagent("simple task")

    assert not compact_called.is_set(), \
        "compact_history should NOT be called when context is small"
    assert "normal result" in result


def test_subagent_empty_choices_guard(monkeypatch):
    """If the LLM returns empty choices (content filter, etc.), the
    subagent should break gracefully and return a result, not crash."""
    empty_response = FakeResponse(
        FakeMessage(content=None), finish_reason="stop")
    empty_response.choices = []  # force empty choices

    _install_fake_client(monkeypatch, [empty_response])

    result = subagent.spawn_subagent("test")

    # Should return something, not crash
    assert isinstance(result, str)


# --------------------------------------------------------------------------- #
# P2: Subagent memory loading
# --------------------------------------------------------------------------- #

def test_subagent_loads_memories(monkeypatch):
    """The subagent should call _load_memories_async and _await_memories
    to inject relevant memories into the first user message."""
    import mcodecore.subagent as _sa

    load_called = threading.Event()
    await_called = threading.Event()

    original_load = _sa._load_memories_async

    def _tracking_load(messages):
        load_called.set()
        return original_load(messages)

    original_await = _sa._await_memories

    def _tracking_await(holder):
        await_called.set()
        return original_await(holder)

    monkeypatch.setattr(_sa, "_load_memories_async", _tracking_load)
    monkeypatch.setattr(_sa, "_await_memories", _tracking_await)

    _install_fake_client(monkeypatch, [
        _resp_stop("done"),
    ])

    result = subagent.spawn_subagent("test task")

    assert load_called.is_set(), \
        "_load_memories_async should have been called"
    assert await_called.is_set(), \
        "_await_memories should have been called"


def test_subagent_memory_content_injected_into_request(monkeypatch):
    """When memories are available, their content is prepended to the
    first user message in the request sent to the LLM."""
    import mcodecore.subagent as _sa

    # Mock memory loading to return known content
    monkeypatch.setattr(_sa, "_load_memories_async",
                        lambda msgs: ["MEMORY_CONTENT", None])

    # _await_memories should return the injected content
    monkeypatch.setattr(_sa, "_await_memories",
                        lambda holder: "MEMORY_CONTENT")

    fake = _install_fake_client(monkeypatch, [
        _resp_stop("done"),
    ])

    subagent.spawn_subagent("original task")

    # Check the request messages sent to the LLM
    assert len(fake.calls) >= 1
    sent_messages = fake.calls[0]["messages"]

    # Find the user message and verify memory content was prepended
    user_msgs = [m for m in sent_messages if m.get("role") == "user"]
    assert len(user_msgs) >= 1
    assert "MEMORY_CONTENT" in user_msgs[-1]["content"]
    assert "original task" in user_msgs[-1]["content"]


def test_subagent_no_memory_does_not_crash(monkeypatch):
    """When no memories are available (empty string), the subagent should
    still work normally without crashing."""
    import mcodecore.subagent as _sa

    monkeypatch.setattr(_sa, "_load_memories_async",
                        lambda msgs: ["", None])
    monkeypatch.setattr(_sa, "_await_memories",
                        lambda holder: "")

    _install_fake_client(monkeypatch, [
        _resp_stop("no memory needed"),
    ])

    result = subagent.spawn_subagent("test")

    assert "no memory needed" in result


def test_subagent_memory_loading_is_called_once_per_subagent(monkeypatch):
    """_load_memories_async is called exactly once at subagent startup."""
    import mcodecore.subagent as _sa

    call_count = [0]
    original_load = _sa._load_memories_async

    def _counting_load(messages):
        call_count[0] += 1
        return original_load(messages)

    monkeypatch.setattr(_sa, "_load_memories_async", _counting_load)

    _install_fake_client(monkeypatch, [
        _resp_tool(None, [_tc("c1", "bash", {"command": "echo hi"})]),
        _resp_stop("done"),
    ])

    subagent.spawn_subagent("multi turn task")

    assert call_count[0] == 1, \
        f"_load_memories_async should be called exactly once, " \
        f"got {call_count[0]}"


# --------------------------------------------------------------------------- #
# L1-L3 progressive compaction tiers (tool_result_budget + snip_compact +
# micro_compact) added to teamagent and subagent loops
# --------------------------------------------------------------------------- #

def test_teammate_calls_snip_micro_tool_budget(monkeypatch):
    """The teammate loop should call tool_result_budget, snip_compact, and
    micro_compact on every turn, not just compact_history."""
    import mcodecore.teammates as _tm

    calls = {"tool_budget": 0, "snip": 0, "micro": 0}

    monkeypatch.setattr(_tm, "tool_result_budget",
                        lambda msgs: (calls.__setitem__("tool_budget", calls["tool_budget"] + 1), msgs)[1])
    monkeypatch.setattr(_tm, "snip_compact",
                        lambda msgs: (calls.__setitem__("snip", calls["snip"] + 1), msgs)[1])
    monkeypatch.setattr(_tm, "micro_compact",
                        lambda msgs: (calls.__setitem__("micro", calls["micro"] + 1), msgs)[1])

    _install_fake_client(monkeypatch, [
        _resp_tool(None, [_tc("c1", "bash", {"command": "echo hi"})]),
        _resp_stop("done"),
    ])

    teammates.spawn_teammate_thread("p0_tiers", "worker", "do work")
    _wait_teammate("p0_tiers")

    assert calls["tool_budget"] >= 2, "tool_result_budget should run every turn"
    assert calls["snip"] >= 2, "snip_compact should run every turn"
    assert calls["micro"] >= 2, "micro_compact should run every turn"


def test_subagent_calls_snip_micro_tool_budget(monkeypatch):
    """The subagent loop should call tool_result_budget, snip_compact, and
    micro_compact on every turn."""
    import mcodecore.subagent as _sa

    calls = {"tool_budget": 0, "snip": 0, "micro": 0}

    monkeypatch.setattr(_sa, "tool_result_budget",
                        lambda msgs: (calls.__setitem__("tool_budget", calls["tool_budget"] + 1), msgs)[1])
    monkeypatch.setattr(_sa, "snip_compact",
                        lambda msgs: (calls.__setitem__("snip", calls["snip"] + 1), msgs)[1])
    monkeypatch.setattr(_sa, "micro_compact",
                        lambda msgs: (calls.__setitem__("micro", calls["micro"] + 1), msgs)[1])

    _install_fake_client(monkeypatch, [
        _resp_tool(None, [_tc("c1", "bash", {"command": "echo hi"})]),
        _resp_stop("done"),
    ])

    subagent.spawn_subagent("multi turn task")

    assert calls["tool_budget"] >= 2, "tool_result_budget should run every turn"
    assert calls["snip"] >= 2, "snip_compact should run every turn"
    assert calls["micro"] >= 2, "micro_compact should run every turn"


def test_teammate_memory_injected_after_compaction(monkeypatch):
    """When memories are available, the teammate should inject them into
    the last user message *after* compaction runs, so the memory content
    is not lost when compaction rebuilds the message list."""
    import mcodecore.teammates as _tm

    monkeypatch.setattr(_tm, "_load_memories_async",
                        lambda msgs: ["MEMORY_CONTENT", None])
    monkeypatch.setattr(_tm, "_await_memories",
                        lambda holder: "MEMORY_CONTENT")

    fake = _install_fake_client(monkeypatch, [
        _resp_stop("done"),
    ])

    teammates.spawn_teammate_thread("p0_mem_compact", "worker", "do work")
    _wait_teammate("p0_mem_compact")

    assert len(fake.calls) >= 1
    sent_messages = fake.calls[0]["messages"]
    user_msgs = [m for m in sent_messages if m.get("role") == "user"]
    assert len(user_msgs) >= 1
    assert "MEMORY_CONTENT" in user_msgs[-1]["content"]


def test_subagent_memory_injected_after_compaction(monkeypatch):
    """Same as above but for subagent: memory injection must survive the
    compaction pipeline."""
    import mcodecore.subagent as _sa

    monkeypatch.setattr(_sa, "_load_memories_async",
                        lambda msgs: ["MEMORY_CONTENT", None])
    monkeypatch.setattr(_sa, "_await_memories",
                        lambda holder: "MEMORY_CONTENT")

    fake = _install_fake_client(monkeypatch, [
        _resp_stop("done"),
    ])

    subagent.spawn_subagent("original task")

    assert len(fake.calls) >= 1
    sent_messages = fake.calls[0]["messages"]
    user_msgs = [m for m in sent_messages if m.get("role") == "user"]
    assert len(user_msgs) >= 1
    assert "MEMORY_CONTENT" in user_msgs[-1]["content"]
    assert "original task" in user_msgs[-1]["content"]


# --------------------------------------------------------------------------- #
# System prompt injection after compaction (P0 fix)
#
# The lead agent keeps its system prompt OUT of the messages list and injects
# it into request_messages[0] AFTER compaction.  Teamagent and subagent used
# to append system into messages, so compact_history / reactive_compact
# would summarise and discard it.  These tests verify the fix.
# --------------------------------------------------------------------------- #

def test_teammate_system_prompt_injected_after_compaction(monkeypatch):
    """The teammate's system prompt must be at request_messages[0] and
    must NOT be present in the conversation messages list."""
    fake = _install_fake_client(monkeypatch, [
        _resp_stop("done"),
    ])

    teammates.spawn_teammate_thread("sys_prompt_t", "worker", "do work")
    _wait_teammate("sys_prompt_t")

    assert len(fake.calls) >= 1
    sent = fake.calls[0]["messages"]
    # System prompt must be at position 0
    assert sent[0]["role"] == "system"
    assert "sys_prompt_t" in sent[0]["content"]
    # No other system messages should appear in the list
    system_count = sum(1 for m in sent if m.get("role") == "system")
    assert system_count == 1, f"Expected exactly 1 system message, got {system_count}"


def test_subagent_system_prompt_injected_after_compaction(monkeypatch):
    """The subagent's system prompt must be at request_messages[0] and
    must NOT be present in the conversation messages list."""
    fake = _install_fake_client(monkeypatch, [
        _resp_stop("done"),
    ])

    subagent.spawn_subagent("test task")

    assert len(fake.calls) >= 1
    sent = fake.calls[0]["messages"]
    # System prompt must be at position 0
    assert sent[0]["role"] == "system"
    # No other system messages should appear
    system_count = sum(1 for m in sent if m.get("role") == "system")
    assert system_count == 1, f"Expected exactly 1 system message, got {system_count}"


def test_teammate_system_prompt_survives_compaction(monkeypatch):
    """When auto-compact triggers, the system prompt must still be present
    in the next LLM call's request_messages."""
    from mcodecore.config import CONTEXT_LIMIT as _CL
    import mcodecore.teammates as _tm

    # Force auto-compact on every turn by setting the threshold to 0
    monkeypatch.setattr(_tm, "CONTEXT_LIMIT", 1)

    # Mock compact_history to return a single compacted user message
    monkeypatch.setattr(_tm, "compact_history",
                        lambda msgs: [{"role": "user", "content": "[Compacted] summary"}])

    fake = _install_fake_client(monkeypatch, [
        _resp_tool(None, [_tc("c1", "bash", {"command": "echo hi"})]),
        _resp_stop("done"),
    ])

    teammates.spawn_teammate_thread("sys_survive_t", "worker", "do work")
    _wait_teammate("sys_survive_t")

    # Second LLM call (after compaction) should still have system at [0]
    assert len(fake.calls) >= 2
    sent2 = fake.calls[1]["messages"]
    assert sent2[0]["role"] == "system"
    assert "sys_survive_t" in sent2[0]["content"]


def test_subagent_system_prompt_survives_compaction(monkeypatch):
    """When auto-compact triggers in subagent, system prompt must still be
    present in the next LLM call."""
    import mcodecore.subagent as _sa

    monkeypatch.setattr(_sa, "CONTEXT_LIMIT", 1)
    monkeypatch.setattr(_sa, "compact_history",
                        lambda msgs: [{"role": "user", "content": "[Compacted] summary"}])

    fake = _install_fake_client(monkeypatch, [
        _resp_tool(None, [_tc("c1", "bash", {"command": "echo hi"})]),
        _resp_stop("done"),
    ])

    subagent.spawn_subagent("test task")

    assert len(fake.calls) >= 2
    sent2 = fake.calls[1]["messages"]
    assert sent2[0]["role"] == "system"


def test_teammate_system_prompt_survives_reactive_compact(monkeypatch):
    """When reactive_compact triggers (prompt_too_long error), the system
    prompt must still be present in the retry call's messages."""
    import mcodecore.teammates as _tm

    monkeypatch.setattr(_tm, "reactive_compact",
                        lambda msgs: [{"role": "user", "content": "[Reactive compact] summary"}])

    fake = _install_fake_client(monkeypatch, [
        _resp_tool(None, [_tc("c1", "bash", {"command": "echo hi"})]),
        # Simulate prompt_too_long error
        Exception("prompt_too_long: maximum context length exceeded"),
        _resp_stop("done"),
    ])

    teammates.spawn_teammate_thread("sys_reactive_t", "worker", "do work")
    _wait_teammate("sys_reactive_t")

    # The retry call (3rd LLM call) should have system at [0]
    assert len(fake.calls) >= 3
    retry_messages = fake.calls[2]["messages"]
    assert retry_messages[0]["role"] == "system"
    assert "sys_reactive_t" in retry_messages[0]["content"]


def test_subagent_system_prompt_survives_reactive_compact(monkeypatch):
    """When reactive_compact triggers in subagent, system prompt must still
    be present in the retry call."""
    import mcodecore.subagent as _sa

    monkeypatch.setattr(_sa, "reactive_compact",
                        lambda msgs: [{"role": "user", "content": "[Reactive compact] summary"}])

    fake = _install_fake_client(monkeypatch, [
        _resp_tool(None, [_tc("c1", "bash", {"command": "echo hi"})]),
        Exception("prompt_too_long: maximum context length exceeded"),
        _resp_stop("done"),
    ])

    subagent.spawn_subagent("test task")

    assert len(fake.calls) >= 3
    retry_messages = fake.calls[2]["messages"]
    assert retry_messages[0]["role"] == "system"
