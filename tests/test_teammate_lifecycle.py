"""Teammate lifecycle / lead-notification tests.

This suite does **not** modify production code.  It drives the real
``spawn_teammate_thread`` / ``run`` logic with a **mocked LLM client**
so every teammate exit path can be exercised deterministically.

The central question investigated:

    "A teammate exits but the lead agent never finds out — is it because
     the teammate never sent a message, because the lead's mailbox was
     flooded / unread, or some other reason?"

Exit paths exercised (see ``teammates.run``):

  P1 normal finish      — finish_reason != tool_calls → ``bus.send(result)``
  P2 LLM call failure    — exception exhausted retries → ``bus.send(error)``
                           then ``return`` (finally → evt.set, status finished)
  P3 idle timeout        — ``idle_poll`` returns "timeout" → breaks out of
                           while → ``bus.send(result)``
  P4 shutdown request    — ``idle_poll`` / handle_inbox returns shutdown →
                           breaks → ``bus.send(result)``
  P5 unexpected exception — any exception *before* the final ``bus.send``
                           (e.g. a tool handler raising) → ``finally`` runs
                           (evt.set + status finished) but **NO message**
                           is sent to lead.  This is the silent-exit path.

Lead-side detection (``_drain_inbox`` in ``agent.py``):

  - finished teammates are detected via ``evt.is_set()`` (event-based,
    independent of the mailbox).
  - lead's inbox is drained via ``consume_lead_inbox``.

So the two notification channels are:
  (a) the ``threading.Event`` (always set in ``finally``) — but only
      observed when the lead's REPL calls ``_drain_inbox``.
  (b) a message in lead's mailbox — only sent on P1/P2/P3/P4, **not** P5.

These tests pin down exactly which channel fires for each path and
whether a mailbox flood can mask the notification.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcodecore import teammates, bus
from mcodecore.config import client as real_client
from mcodecore.context import ctx


# --------------------------------------------------------------------------- #
# Helpers: fake LLM responses + mock client
# --------------------------------------------------------------------------- #

class FakeToolCall:
    """Mimics the OpenAI SDK tool-call object."""
    def __init__(self, cid, name, arguments="{}"):
        self.id = cid
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=arguments)
        self.index = 0

    def model_dump(self, exclude_none=True):
        d = {"id": self.id, "type": self.type,
             "function": {"name": self.function.name,
                           "arguments": self.function.arguments}}
        return d


class FakeMessage:
    """Mimics ``ChatCompletionMessage``."""
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
    """A fake ``client.chat.completions.create`` that returns scripted
    responses (or raises scripted exceptions) in sequence.

    Each item in the script is either:
      - a ``FakeResponse``  → returned as-is
      - an ``Exception``    → raised
      - a callable          → called with ``**kwargs``; its return used
    After the script is exhausted, raises ``StopIteration``.
    """

    def __init__(self, script):
        self._script = list(script)
        self._idx = 0
        self.calls = []

    def chat_completions_create(self, *args, **kwargs):
        return self.create(*args, **kwargs)

    def create(self, *args, **kwargs):
        self.calls.append(kwargs)
        if self._idx >= len(self._script):
            raise StopIteration("script exhausted")
        item = self._script[self._idx]
        self._idx += 1
        if isinstance(item, BaseException):
            raise item
        if callable(item) and not isinstance(item, FakeResponse):
            return item(**kwargs)
        return item


def _install_fake_client(monkeypatch, script):
    """Replace ``client.chat.completions.create`` in every module that
    imported it (``config.client`` is the single source object; teammates
    imports ``client`` from config, and so does ``teammates.client``)."""
    fake = ScriptedClient(script)
    # teammates.py does:  from .config import LLM_MODEL, client
    # so patch the attribute on the real client object's completions.
    monkeypatch.setattr(real_client.chat.completions, "create", fake.create)
    return fake


def _msg(content):
    return FakeMessage(content=content)


def _tc(cid, name, args=None):
    return FakeToolCall(cid, name, json.dumps(args or {}))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _fast_idle(monkeypatch):
    """Speed up ``idle_poll`` so idle-timeout tests don't take 60s.

    ``idle_poll`` uses ``range(IDLE_TIMEOUT // IDLE_POLL_INTERVAL)`` which
    requires **integers** (Python 3 ``range`` rejects floats), so the
    patched values must be ``int``.
    """
    monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)
    # Skip backoff sleeps in retry paths so tests are fast.
    import mcodecore.teammates as _tm
    monkeypatch.setattr(_tm.time, "sleep", lambda s: None)


def _wait_teammate(name, timeout=5.0):
    """Block until the teammate's event is set (it finished)."""
    evt = ctx.active_teammates.get(name)
    assert evt is not None, f"teammate {name} not registered"
    assert evt.wait(timeout=timeout), f"teammate {name} did not finish in {timeout}s"


def _load_team_history(name):
    """Read the JSONL team-history file for *name* and return a list of
    event dicts ``{"ts": ..., "event": ..., "detail": ...}``.

    The history dir is ``WORKDIR / .team_history`` (set inside
    ``teammates.run``), and ``WORKDIR`` is monkey-patched by
    ``isolate_paths`` to ``tmp_path`` in tests.
    """
    from mcodecore import teammates as _tm
    # log_team_history writes to TEAM_HISTORY_DIR (session-scoped:
    # .team_history/<sid>/<name>.jsonl), not the legacy flat layout.
    history_file = _tm.TEAM_HISTORY_DIR / f"{name}.jsonl"
    if not history_file.exists():
        return []
    records = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _tool_outputs(name, tool_name):
    """Return a list of output strings for every call to *tool_name* by
    teammate *name*, extracted from the team history."""
    return [r["detail"]["output"]
            for r in _load_team_history(name)
            if r["event"] == "tool_called"
            and r["detail"].get("tool") == tool_name]


# --------------------------------------------------------------------------- #
# P1 — normal finish: lead DOES get a "result" message
# --------------------------------------------------------------------------- #

def test_normal_finish_sends_result_message(monkeypatch):
    """Teammate finishes normally (non-tool-call finish_reason).

    Expected: a ``type=result`` message lands in lead's inbox AND the event
    is set.  Lead learns of the exit via *both* channels.
    """
    _install_fake_client(monkeypatch, [
        FakeResponse(_msg("all done"), finish_reason="stop"),
    ])
    teammates.spawn_teammate_thread("t1", "worker", "say hello")
    _wait_teammate("t1")

    # channel (a): event set + registry status
    assert ctx.teammate_registry["t1"]["status"] == "finished"
    # channel (b): lead inbox has a result message
    lead_inbox = ctx.bus.read_inbox("lead")
    result_msgs = [m for m in lead_inbox if m.get("type") == "result"]
    assert len(result_msgs) == 1
    assert "all done" in result_msgs[0]["content"]


# --------------------------------------------------------------------------- #
# P2 — LLM call failure: lead DOES get an error message
# --------------------------------------------------------------------------- #

def test_llm_failure_sends_error_message(monkeypatch):
    """The LLM raises a transient exception (retries exhausted).

    RuntimeError("connection refused") contains "connection" so it IS now
    classified as transient and retried up to MAX_REACTIVE_RETRIES=3.
    After exhausting retries, ``run`` sends a ``LLM API error`` message to
    lead, then returns.
    """
    _install_fake_client(monkeypatch, [
        RuntimeError("connection refused"),
        RuntimeError("connection refused"),
        RuntimeError("connection refused"),
        RuntimeError("connection refused"),  # MAX_REACTIVE_RETRIES=3, 4th -> exhausted
    ])
    teammates.spawn_teammate_thread("t2", "worker", "do stuff")
    _wait_teammate("t2", timeout=10)

    assert ctx.teammate_registry["t2"]["status"] == "finished"
    lead_inbox = ctx.bus.read_inbox("lead")
    err_msgs = [m for m in lead_inbox if m.get("type") == "LLM API error"]
    assert len(err_msgs) == 1, f"expected 1 error msg, got {err_msgs}"
    assert "connection refused" in err_msgs[0]["content"]

def test_llm_timeout_exhausts_retries_then_sends_error(monkeypatch):
    """Timeout exceptions ARE retried (up to MAX_REACTIVE_RETRIES=3).

    After exhausting retries, an error message is sent to lead.
    """
    _install_fake_client(monkeypatch, [
        TimeoutError("request timed out"),
        TimeoutError("request timed out"),
        TimeoutError("request timed out"),
        TimeoutError("request timed out"),  # 4th → retries exhausted
    ])
    teammates.spawn_teammate_thread("t2b", "worker", "do stuff")
    _wait_teammate("t2b", timeout=10)

    assert ctx.teammate_registry["t2b"]["status"] == "finished"
    lead_inbox = ctx.bus.read_inbox("lead")
    err_msgs = [m for m in lead_inbox if m.get("type") == "LLM API error"]
    assert len(err_msgs) == 1


# --------------------------------------------------------------------------- #
# P3 — idle timeout: lead DOES get a result message
# --------------------------------------------------------------------------- #

def test_idle_timeout_sends_result_message(monkeypatch):
    """Teammate's first turn finishes (stop), then idle_poll times out.

    Expected: ``run`` falls through to ``bus.send(result)``.
    """
    _install_fake_client(monkeypatch, [
        FakeResponse(_msg("working"), finish_reason="stop"),
    ])
    teammates.spawn_teammate_thread("t3", "worker", "do stuff")
    _wait_teammate("t3", timeout=10)

    assert ctx.teammate_registry["t3"]["status"] == "finished"
    lead_inbox = ctx.bus.read_inbox("lead")
    result_msgs = [m for m in lead_inbox if m.get("type") == "result"]
    assert len(result_msgs) == 1


# --------------------------------------------------------------------------- #
# P4 — shutdown request: lead DOES get a result message + shutdown_response
# --------------------------------------------------------------------------- #

def test_shutdown_request_sends_shutdown_and_result(monkeypatch):
    """Lead sends a shutdown_request; teammate approves then finishes.

    Expected: a ``shutdown_response`` (protocol) message + a ``result`` message.
    """
    _install_fake_client(monkeypatch, [
        FakeResponse(_msg("ok"), finish_reason="stop"),
    ])
    teammates.spawn_teammate_thread("t4", "worker", "do stuff")
    # Give the thread a moment to enter the idle poll
    time.sleep(0.05)
    # Send shutdown
    bus.run_request_shutdown("t4")
    _wait_teammate("t4", timeout=10)

    assert ctx.teammate_registry["t4"]["status"] == "finished"
    lead_inbox = ctx.bus.read_inbox("lead")
    # protocol response consumed by consume_lead_inbox routing; but raw messages
    # are returned by read_inbox (read_inbox doesn't route — only consume does).
    types = [m.get("type") for m in lead_inbox]
    assert "shutdown_response" in types
    assert "result" in types


# --------------------------------------------------------------------------- #
# P5 — SILENT EXIT: exception during tool execution, NO message to lead
# --------------------------------------------------------------------------- #

def test_tool_handler_exception_silent_exit(monkeypatch):
    """A tool handler raises an unexpected exception during execution.

    After Fix #6 (wrap teammate tool calls in try/except, mirroring the
    lead agent's error handling), any tool exception is surfaced to the
    LLM as an error string ``Error: ...`` and the teammate continues its
    loop.  The teammate then calls ``finish`` to report the result.

    The OLD behavior (pre-Fix-#6) was: the exception escaped to the
    outer ``except`` and a ``crashed`` notification was sent, killing the
    teammate.  This is no longer the case - tool errors are recoverable.
    """
    _install_fake_client(monkeypatch, [
        # Turn 1: teammate calls a tool whose handler will explode
        FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc("c1", "read_file", {"path": "x.txt"})]),
            finish_reason="tool_calls",
        ),
        # Turn 2: the error was fed back; teammate calls finish to report
        FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc("c2", "finish", {"summary": "handled tool error"})]),
            finish_reason="tool_calls",
        ),
    ])
    # Replace the real read_file handler with one that raises.
    from mcodecore import fsops
    def _boom(**kwargs):
        raise ValueError("simulated catastrophic failure in tool")
    monkeypatch.setattr(fsops, "run_read", _boom)

    teammates.spawn_teammate_thread("t5", "worker", "read a file")
    _wait_teammate("t5", timeout=5)

    # Event IS set and registry IS marked finished (finally block ran)
    assert ctx.teammate_registry["t5"]["status"] == "finished"
    # After Fix #6: no crashed notification - tool error is recovered.
    # The teammate delivered a result via the finish tool instead.
    lead_inbox = ctx.bus.read_inbox("lead")
    crashed = [m for m in lead_inbox
               if m.get("from") == "t5" and m.get("type") == "crashed"]
    assert len(crashed) == 0, (
        f"expected no crashed message (tool error should be recovered), "
        f"got {len(crashed)}; full inbox: "
        f"{[(m.get('from'), m.get('type')) for m in lead_inbox]}")
    results = [m for m in lead_inbox
               if m.get("from") == "t5" and m.get("type") == "result"]
    assert len(results) == 1, (
        f"expected 1 result message from t5 (via finish tool), got "
        f"{len(results)}; full inbox: "
        f"{[(m.get('from'), m.get('type')) for m in lead_inbox]}")
    assert "handled tool error" in results[0]["content"]


def test_parse_tool_args_exception_silent_exit(monkeypatch):
    """If ``parse_tool_args`` raises (it normally doesn't, but any
    unexpected error in the tool-dispatch section), the exception escapes.

    We simulate by making the tool_call arguments non-JSON in a way that
    parse_tool_args handles gracefully — so instead we patch parse_tool_args
    to raise, demonstrating the same silent-exit behavior.
    """
    _install_fake_client(monkeypatch, [
        FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc("c1", "bash", {"command": "echo hi"})]),
            finish_reason="tool_calls",
        ),
    ])

    def _bad_parse(*a, **kw):
        raise RuntimeError("simulated parse failure")
    # teammates.py imports parse_tool_args at module top-level (line 17),
    # so patch it on the teammates module, not on utils.
    monkeypatch.setattr(teammates, "parse_tool_args", _bad_parse)

    teammates.spawn_teammate_thread("t5b", "worker", "do stuff")
    _wait_teammate("t5b", timeout=5)

    assert ctx.teammate_registry["t5b"]["status"] == "finished"
    # After fix (改进2): parse failure now triggers the crashed notification.
    lead_inbox = ctx.bus.read_inbox("lead")
    crashed = [m for m in lead_inbox
               if m.get("from") == "t5b" and m.get("type") == "crashed"]
    assert len(crashed) == 1, (
        f"expected 1 crashed message from t5b, got {len(crashed)}; "
        f"full inbox: {[(m.get('from'), m.get('type')) for m in lead_inbox]}")
    assert "RuntimeError" in crashed[0]["content"]
    assert "simulated parse failure" in crashed[0]["content"]


# --------------------------------------------------------------------------- #
# Lead-side detection: _drain_inbox picks up finished via event even
# when mailbox is empty (the P5 scenario)
# --------------------------------------------------------------------------- #

def test_lead_drain_inbox_detects_finished_without_message(monkeypatch):
    """Even when a teammate exits silently (no mailbox message), the lead's
    ``_drain_inbox`` detects it via ``evt.is_set()``.

    After Fix #6, a tool exception is **not** fatal - the error is fed
    back to the LLM as ``Error: ...``.  In this test the script has only
    one response, so the second LLM call raises ``StopIteration`` which
    is caught and reported as an ``LLM API error`` message.  The key
    invariant: the event is set (teammate is finished) so the lead can
    detect the exit even if the mailbox has only an error message.
    """
    # Force tool error -> LLM error (script exhausted)
    _install_fake_client(monkeypatch, [
        FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc("c1", "read_file", {"path": "x"})]),
            finish_reason="tool_calls",
        ),
    ])
    from mcodecore import fsops
    monkeypatch.setattr(fsops, "run_read", lambda **kw: (_ for _ in ()).throw(ValueError("boom")))

    teammates.spawn_teammate_thread("t6", "worker", "go")
    _wait_teammate("t6")

    # Simulate what _drain_inbox does (agent.py):
    finished = [n for n, e in list(ctx.active_teammates.items()) if e.is_set()]
    assert "t6" in finished, "event-based detection should catch exit"
    # After Fix #6: tool error is recovered (fed to LLM), not crashed.
    # The teammate still exits (because script is exhausted -> LLM API
    # error), so there IS a mailbox message - but it's an LLM API error,
    # not a crashed notification.
    lead_inbox = ctx.bus.read_inbox("lead")
    crashed = [m for m in lead_inbox
               if m.get("from") == "t6" and m.get("type") == "crashed"]
    assert len(crashed) == 0, (
        f"expected no crashed notification (tool error recovered), "
        f"got {len(crashed)}; inbox: "
        f"{[(m.get('from'), m.get('type')) for m in lead_inbox]}")
    # The teammate should have SOME mailbox message (LLM API error from
    # the exhausted script), proving it didn't exit completely silently.
    assert len(lead_inbox) >= 1, (
        "expected at least one mailbox message for t6 (LLM API error)")

# --------------------------------------------------------------------------- #
# Mailbox flooding: does the lead lose messages?
# --------------------------------------------------------------------------- #

def test_lead_inbox_survives_flood(monkeypatch):
    """Many messages queued in lead's mailbox before the lead reads them.

    ``MessageBus`` uses an atomic-rename JSONL append — there is no cap,
    no eviction.  All messages survive.  So mailbox *flooding does not
    cause message loss*; the risk is only that the lead never *reads*
    them (doesn't call check_inbox / _drain_inbox).
    """
    _install_fake_client(monkeypatch, [
        FakeResponse(_msg("done"), finish_reason="stop"),
    ])
    # Pre-flood lead's inbox with 200 ordinary messages
    for i in range(200):
        ctx.bus.send("flood", "lead", f"flood msg {i}")

    teammates.spawn_teammate_thread("t7", "worker", "go")
    _wait_teammate("t7")

    lead_inbox = ctx.bus.read_inbox("lead")
    flood = [m for m in lead_inbox if m["from"] == "flood"]
    results = [m for m in lead_inbox if m.get("type") == "result"]
    assert len(flood) == 200, f"lost flood messages: got {len(flood)}"
    assert len(results) == 1, "teammate result should still be present"
    # Order preserved: flood messages first, then result (append order)
    assert lead_inbox[0]["content"] == "flood msg 0"
    assert lead_inbox[-1]["type"] == "result"


def test_concurrent_read_inbox_no_duplication(monkeypatch):
    """Two threads calling ``read_inbox`` concurrently.

    ``MessageBus.read_inbox`` uses an atomic-rename pattern, but the
    ``tmp.read_text()`` call (bus.py:55) is **outside** the try/except
    that guards the rename.  Under concurrent readers on Windows this
    can raise ``FileNotFoundError`` - a robustness gap, though it does
    NOT cause message duplication or loss (the surviving reader gets
    all messages; the erroring reader gets none).

    This test verifies the *content* guarantee (no dup, no loss) while
    tolerating the transient ``FileNotFoundError`` as an observed
    (not fixed) behaviour.

    Uses a *unique* agent name (not ``"lead"``) to avoid cross-test
    interference from daemon teammate threads that may still be writing
    to the shared ``lead`` mailbox.
    """
    from mcodecore.bus import MessageBus
    mb = ctx.bus
    agent = "concur_test_agent"
    for i in range(100):
        mb.send("src", agent, f"m{i}")

    collected = []
    barrier = threading.Barrier(2)
    fs_errors = []

    def reader():
        try:
            barrier.wait(timeout=2)
            for _ in range(10):
                msgs = mb.read_inbox(agent)
                collected.extend(msgs)
        except FileNotFoundError as e:
            # Known gap: tmp.read_text() raced with another reader.
            fs_errors.append(e)
        except Exception as e:
            fs_errors.append(e)

    t1 = threading.Thread(target=reader)
    t2 = threading.Thread(target=reader)
    t1.start(); t2.start()
    t1.join(timeout=5); t2.join(timeout=5)

    # Content guarantee: every unique message appears exactly once across
    # both readers combined (one reader wins the rename, the other gets []).
    contents = [m["content"] for m in collected]
    expected = sorted(f"m{i}" for i in range(100))
    # Deduplicate then compare - if atomic rename worked, no dups.
    assert len(set(contents)) == len(contents), (
        f"duplicate messages detected: {len(contents)} msgs, "
        f"{len(set(contents))} unique")
    assert sorted(contents) == expected, (
        f"loss detected: got {len(contents)} msgs, expected 100")
    assert len(contents) == 100


# --------------------------------------------------------------------------- #
# The real gap: lead is busy and never calls _drain_inbox
# --------------------------------------------------------------------------- #

def test_lead_busy_does_not_notice_silent_exit(monkeypatch):
    """Simulate the lead being inside a long ``agent_loop`` (blocked on
    a slow tool) while a teammate exits silently.

    While blocked, ``ctx.active_teammates`` still contains the teammate
    (its event was set, but ``_drain_inbox`` hasn't run to pop it).
    The teammate *appears* active to any code that only checks
    ``active_teammates`` membership, even though it already died.

    This is the crux: the lead's awareness is **pull-based** (only at
    REPL boundaries), so a teammate dying mid-turn is invisible until
    the lead returns to the REPL.
    """
    _install_fake_client(monkeypatch, [
        FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc("c1", "read_file", {"path": "x"})]),
            finish_reason="tool_calls",
        ),
    ])
    from mcodecore import fsops
    monkeypatch.setattr(fsops, "run_read", lambda **kw: (_ for _ in ()).throw(ValueError("boom")))

    teammates.spawn_teammate_thread("t8", "worker", "go")
    _wait_teammate("t8")

    # At this point, the teammate thread is dead, but because _drain_inbox
    # has NOT run, active_teammates still lists it as "present".
    assert "t8" in ctx.active_teammates, (
        "teammate still in active_teammates until _drain_inbox pops it")
    evt = ctx.active_teammates["t8"]
    assert evt.is_set(), "event IS set, but nobody polled it"

    # Registry status IS "finished" (set in finally), so a smarter lead
    # that checks teammate_registry would notice immediately.
    assert ctx.teammate_registry["t8"]["status"] == "finished"


# --------------------------------------------------------------------------- #
# Event-set BEFORE mailbox message: ordering of finally vs. send
# --------------------------------------------------------------------------- #

def test_event_set_in_finally_even_if_send_fails(monkeypatch):
    """If ``bus.send(result)`` itself raises (e.g. disk full), the
    ``finally`` still sets the event.  So event-based detection is more
    robust than mailbox-based detection.
    """
    _install_fake_client(monkeypatch, [
        FakeResponse(_msg("done"), finish_reason="stop"),
    ])
    # Make bus.send blow up on the final result send.  We can't easily
    # target only the final send, so make ALL sends fail and verify the
    # event is still set.
    original_send = ctx.bus.send
    call_log = []

    def _flaky_send(*args, **kwargs):
        call_log.append(args)
        raise OSError("disk full")

    monkeypatch.setattr(ctx.bus, "send", _flaky_send)

    teammates.spawn_teammate_thread("t9", "worker", "go")
    # The send in the LLM-error path or the result path raises; but run()'s
    # try-block doesn't wrap the final send in its own try, so the OSError
    # propagates to finally.  finally sets the event.
    _wait_teammate("t9", timeout=10)

    assert ctx.active_teammates["t9"].is_set(), (
        "event must be set in finally even if bus.send raised")
    assert ctx.teammate_registry["t9"]["status"] == "finished"


# =========================================================================== #
# PART 2 - Additional silent-exit paths discovered in deeper review
# =========================================================================== #
#
# The LLM try/except (lines 176-212) protects ONLY the ``create`` call.
# Everything AFTER line 213 (response.choices[0], trigger_hooks,
# tool dispatch, idle_poll) is UNPROTECTED.  Any exception there escapes
# run()'s outer try -> finally (event + registry) but NO message to lead.
#
# Worse: if the ``finally`` block itself raises, the event is NEVER set
# and lead loses BOTH channels.
# =========================================================================== #


def _crashed_exit_assertions(name):
    """Shared assertions for a crash-exit path (after 改进2 fix).

    Before the fix these paths exited silently (empty mailbox).  After
    改进2 the outer ``except Exception`` sends a ``crashed`` notification,
    so the lead always learns via both the event channel AND mailbox.
    """
    _wait_teammate(name, timeout=8)
    assert ctx.teammate_registry[name]["status"] == "finished", (
        "finally should mark status finished")
    lead_inbox = ctx.bus.read_inbox("lead")
    crashed = [m for m in lead_inbox
               if m.get("from") == name and m.get("type") == "crashed"]
    assert len(crashed) == 1, (
        f"expected 1 crashed message from {name}, got {len(crashed)}; "
        f"full inbox: {[(m.get('from'), m.get('type')) for m in lead_inbox]}")
    return crashed[0]

# Backward-compat alias for any remaining references.
def _silent_exit_assertions(name):
    return _crashed_exit_assertions(name)


# --------------------------------------------------------------------------- #
# P5a - empty ``choices`` from the LLM (content-filter / API quirk) -> IndexError
# --------------------------------------------------------------------------- #

def test_empty_choices_handled_gracefully(monkeypatch):
    """Some LLM providers return an empty ``choices`` list (content-filter,
    rate-limit soft-fail, etc.) instead of raising.

    After fix (改进4): ``run`` now guards ``if not response.choices`` and
    breaks out of the loop with an empty assistant message, then sends a
    ``result`` to the lead.  The teammate no longer crashes with
    ``IndexError`` on ``choices[0]``.
    """
    empty_resp = SimpleNamespace(
        choices=[],
        usage=None,
    )
    _install_fake_client(monkeypatch, [empty_resp])
    teammates.spawn_teammate_thread("ta", "worker", "go")
    _wait_teammate("ta", timeout=8)

    assert ctx.teammate_registry["ta"]["status"] == "finished"
    lead_inbox = ctx.bus.read_inbox("lead")
    results = [m for m in lead_inbox
               if m.get("from") == "ta" and m.get("type") == "result"]
    assert len(results) == 1, (
        f"empty choices should produce a result, not a crash; "
        f"got: {[(m.get('from'), m.get('type')) for m in lead_inbox]}")
    # No crashed message - this path is now handled gracefully.
    crashed = [m for m in lead_inbox
               if m.get("from") == "ta" and m.get("type") == "crashed"]
    assert crashed == [], "empty choices should not crash the teammate"


# --------------------------------------------------------------------------- #
# P5b - PreToolUse hook raises -> silent exit
# --------------------------------------------------------------------------- #

def test_pretooluse_hook_exception_does_not_kill_teammate(monkeypatch):
    """After fix (改进3): ``trigger_hooks("PreToolUse", ...)`` is wrapped in
    its own try/except.  A buggy hook that raises no longer kills the
    teammate - the hook is skipped (graceful degradation) and the tool
    still executes.  The teammate finishes normally with a ``result``.
    """
    _install_fake_client(monkeypatch, [
        # Turn 1: tool call with a crashing hook
        FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc("c1", "bash", {"command": "echo hi"})]),
            finish_reason="tool_calls",
        ),
        # Turn 2: normal finish (the tool ran despite the hook crash)
        FakeResponse(_msg("done"), finish_reason="stop"),
    ])
    from mcodecore import hooks
    def _boom_hook(*args, **kwargs):
        raise RuntimeError("hook exploded")
    monkeypatch.setattr(hooks, "trigger_hooks", _boom_hook)
    monkeypatch.setattr(teammates, "trigger_hooks", _boom_hook)

    teammates.spawn_teammate_thread("tb", "worker", "go")
    _wait_teammate("tb", timeout=8)

    assert ctx.teammate_registry["tb"]["status"] == "finished"
    lead_inbox = ctx.bus.read_inbox("lead")
    results = [m for m in lead_inbox
               if m.get("from") == "tb" and m.get("type") == "result"]
    assert len(results) == 1, (
        f"teammate should finish normally (hook caught), got: "
        f"{[(m.get('from'), m.get('type')) for m in lead_inbox]}")
    # No crashed message - the hook was caught, not the teammate.
    crashed = [m for m in lead_inbox
               if m.get("from") == "tb" and m.get("type") == "crashed"]
    assert crashed == [], "PreToolUse hook error should not crash teammate"


# --------------------------------------------------------------------------- #
# P5c - PostToolUse hook raises -> silent exit (after tool already ran!)
# --------------------------------------------------------------------------- #

def test_posttooluse_hook_exception_does_not_kill_teammate(monkeypatch):
    """After fix (改进3): ``trigger_hooks("PostToolUse", ...)`` is also
    wrapped.  The tool already ran; a PostToolUse hook crash is logged
    and skipped, the tool result is kept, and the teammate survives.
    """
    _install_fake_client(monkeypatch, [
        # Turn 1: tool call
        FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc("c1", "bash", {"command": "echo hi"})]),
            finish_reason="tool_calls",
        ),
        # Turn 2: normal finish
        FakeResponse(_msg("done"), finish_reason="stop"),
    ])

    # PreToolUse must pass; PostToolUse must explode.
    call_count = {"pre": 0}
    from mcodecore import hooks
    def _selective_hook(event, *args, **kwargs):
        if event == "PreToolUse":
            call_count["pre"] += 1
            return None  # not blocked
        raise RuntimeError("post-hook exploded")
    monkeypatch.setattr(teammates, "trigger_hooks", _selective_hook)

    teammates.spawn_teammate_thread("tc", "worker", "go")
    _wait_teammate("tc", timeout=8)

    assert ctx.teammate_registry["tc"]["status"] == "finished"
    assert call_count["pre"] >= 1, "PreToolUse should have run before PostToolUse"
    lead_inbox = ctx.bus.read_inbox("lead")
    results = [m for m in lead_inbox
               if m.get("from") == "tc" and m.get("type") == "result"]
    assert len(results) == 1, (
        f"teammate should finish normally (post-hook caught), got: "
        f"{[(m.get('from'), m.get('type')) for m in lead_inbox]}")
    crashed = [m for m in lead_inbox
               if m.get("from") == "tc" and m.get("type") == "crashed"]
    assert crashed == [], "PostToolUse hook error should not crash teammate"


# --------------------------------------------------------------------------- #
# P5d - idle_poll raises (e.g. task store I/O failure) -> silent exit
# --------------------------------------------------------------------------- #

def test_idle_poll_exception_sends_crashed(monkeypatch):
    """``idle_poll`` calls ``scan_unclaimed_tasks()`` / ``claim_task()``
    which touch the task store on disk.  If the task store raises
    (corrupted JSON, permission denied, etc.), the exception escapes
    ``idle_poll`` and then ``run``.

    After fix (改进2): the outer ``except Exception`` catches it and
    sends a ``crashed`` notification, so the lead learns that the
    teammate died during idle (rather than vanishing silently).
    """
    _install_fake_client(monkeypatch, [
        # First turn: respond normally, break out of the for-loop.
        FakeResponse(_msg("started work"), finish_reason="stop"),
    ])
    from mcodecore import bus as bus_mod
    def _boom_idle(*args, **kwargs):
        raise OSError("task store corrupted")
    monkeypatch.setattr(bus_mod, "idle_poll", _boom_idle)
    monkeypatch.setattr(teammates, "idle_poll", _boom_idle)

    teammates.spawn_teammate_thread("td", "worker", "go")
    msg = _crashed_exit_assertions("td")
    assert "OSError" in msg["content"]
    assert "task store corrupted" in msg["content"]


# --------------------------------------------------------------------------- #
# P5e - log_team_history raises INSIDE finally -> event NEVER set
# --------------------------------------------------------------------------- #

def test_finally_log_failure_does_not_prevent_event_set(monkeypatch):
    """After fix (改进1): the ``finally`` block now sets the event and
    marks the registry status BEFORE calling ``log_team_history``, and
    ``log_team_history`` itself is wrapped in try/except.  So even if
    logging raises (e.g. ``json.dumps`` on non-serializable hook output),
    the event is still set and the registry still says ``finished``.

    Before the fix this was the WORST case: a log failure left the event
    UNSET and status ``running`` forever - the lead thought the teammate
    was still alive.  Now the event channel is guaranteed.
    """
    _install_fake_client(monkeypatch, [
        FakeResponse(_msg("done"), finish_reason="stop"),
    ])
    # truncate is called inside log_team_history's detail construction.
    # Making it raise guarantees the "finished" log in finally explodes.
    from mcodecore import utils
    call_count = {"n": 0}
    def _boom_truncate(*args, **kwargs):
        call_count["n"] += 1
        # Let early logs (spawned, llm_response) succeed; explode on the
        # "finished" log which happens in finally.
        if call_count["n"] > 3:
            raise RuntimeError("truncate boom in finally")
        return "ok"
    monkeypatch.setattr(teammates, "truncate", _boom_truncate)

    teammates.spawn_teammate_thread("te", "worker", "go")
    _wait_teammate("te", timeout=8)

    evt = ctx.active_teammates.get("te")
    assert evt is not None
    # After 改进1: event IS set even though logging failed.
    assert evt.is_set(), (
        "event must be set even if log_team_history raised - "
        "evt.set() now runs before logging")
    assert ctx.teammate_registry["te"]["status"] == "finished", (
        "registry status must be 'finished' even if logging failed")


# --------------------------------------------------------------------------- #
# P5f - log_team_history raises but AFTER evt.set would be fine:
#       verify current ordering is log-BEFORE-set (the dangerous order)
# --------------------------------------------------------------------------- #

def test_finally_ordering_event_set_before_log(monkeypatch):
    """After fix (改进1): verify the source ordering in ``finally`` is now
    ``evt.set()`` BEFORE ``log_team_history`` (the safe order).  This
    guarantees a logging failure cannot prevent the event from being set.

    We inspect the source lines of ``run``'s finally block.
    """
    import inspect, re
    src = inspect.getsource(teammates.spawn_teammate_thread)
    finally_idx = src.index("finally:")
    # Search for actual *calls*, not mentions in comments.
    # evt.set() is a call like "evt.set()".
    set_match = re.search(r'\bevt\.set\(\)', src[finally_idx:])
    set_idx = set_match.start()
    # log_team_history( is an actual call (not a comment mention).
    log_match = re.search(r'log_team_history\(', src[finally_idx + set_idx:])
    log_idx = set_idx + log_match.start()
    assert set_idx < log_idx, (
        "evt.set() must be called BEFORE log_team_history in finally "
        "(fixed order) - so a log failure cannot prevent event "
        "notification")


# --------------------------------------------------------------------------- #
# P5g - 50-turn cap: teammate exhausts ``range(50)`` then falls to
#       idle_poll; if idle_poll times out it IS reported (result msg).
#       But the 50-turn cap itself is silent if the last turn had tool_calls
#       (no final assistant message with content).
# --------------------------------------------------------------------------- #

def test_fifty_turn_cap_without_final_answer_sends_result(monkeypatch):
    """If a teammate loops 50 tool-call turns without ever producing a
    non-tool-call finish, the ``for _ in range(50)`` exhausts, falls to
    ``idle_poll``.  On timeout, ``result = messages[-1]["content"]`` is
    likely the last tool output (not ``None`` since bash always returns
    a string), so that content is sent as the result.  If it were
    ``None``/empty, the fallback "stopped after 50 turns" string fires.

    Either way this path IS reported (a ``result`` message is sent), so
    the 50-turn cap is NOT a silent-exit cause - unlike the unguarded
    exceptions above.

    We filter inbox messages by ``from == "tg"`` to avoid interference
    from daemon teammate threads leaked from prior tests writing to the
    shared ``lead`` mailbox.
    """
    # 50 tool-call responses, never a "stop"
    script = [
        FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc(f"c{i}", "bash", {"command": "echo"})]),
            finish_reason="tool_calls",
        )
        for i in range(50)
    ]
    _install_fake_client(monkeypatch, script)
    teammates.spawn_teammate_thread("tg", "worker", "go")
    _wait_teammate("tg", timeout=15)

    assert ctx.teammate_registry["tg"]["status"] == "finished"
    lead_inbox = ctx.bus.read_inbox("lead")
    result_msgs = [m for m in lead_inbox
                   if m.get("type") == "result" and m.get("from") == "tg"]
    assert len(result_msgs) == 1, (
        f"expected 1 result from tg, got {len(result_msgs)}; "
        f"full inbox types: {[(m.get('from'), m.get('type')) for m in lead_inbox]}")
    # Content is either the last tool output or the 50-turn fallback.
    assert result_msgs[0]["content"]


# --------------------------------------------------------------------------- #
# P6 - Misclassified exit: _await_memories failure reported as "LLM API error"
# --------------------------------------------------------------------------- #

def test_memory_failure_misreported_as_llm_error(monkeypatch):
    """``_await_memories`` (line 178) runs INSIDE the LLM try/except.
    If it raises, the except branch fires and sends an ``LLM API error``
    message - even though the LLM call itself never happened.

    This is NOT silent (lead does get a message), but it is MISCLASSIFIED:
    the lead sees "LLM call failed" when the real cause is a memory-load
    failure.  In practice this can mislead debugging.
    """
    _install_fake_client(monkeypatch, [
        # This response will never be reached because _await_memories
        # raises first, inside the try.
        FakeResponse(_msg("done"), finish_reason="stop"),
    ])
    from mcodecore import memory
    def _boom_await(*args, **kwargs):
        raise RuntimeError("memory store unreadable")
    monkeypatch.setattr(memory, "_await_memories", _boom_await)
    monkeypatch.setattr(teammates, "_await_memories", _boom_await)

    teammates.spawn_teammate_thread("th", "worker", "go")
    _wait_teammate("th", timeout=8)

    assert ctx.teammate_registry["th"]["status"] == "finished"
    lead_inbox = ctx.bus.read_inbox("lead")
    err_msgs = [m for m in lead_inbox if m.get("type") == "LLM API error"]
    assert len(err_msgs) == 1, "memory failure should be (mis)reported as LLM error"
    assert "memory store unreadable" in err_msgs[0]["content"]


# --------------------------------------------------------------------------- #
# P7 - Lead never calls _drain_inbox while blocked on input()
# --------------------------------------------------------------------------- #

def test_lead_repl_blocked_on_input_cannot_detect_exit(monkeypatch):
    """``_drain_inbox`` is only called after ``input()`` returns (REPL
    boundary, lines 192-204).  If the lead is blocked on ``input()``
    waiting for user typing, and a teammate exits during that wait,
    the lead cannot detect it until the user presses Enter.

    We simulate: teammate dies, then check that BEFORE a drain call,
    ``active_teammates`` still holds the dead teammate (event set but
    not popped).  This mirrors the real gap between REPL turns.
    """
    # Silent exit via empty choices
    empty_resp = SimpleNamespace(choices=[], usage=None)
    _install_fake_client(monkeypatch, [empty_resp])
    teammates.spawn_teammate_thread("ti", "worker", "go")
    _wait_teammate("ti")

    # Lead has NOT called _drain_inbox yet (simulating blocked-on-input).
    # The dead teammate is still listed as "active" - looks alive.
    assert "ti" in ctx.active_teammates
    # But the event reveals death:
    assert ctx.active_teammates["ti"].is_set()
    # Simulate _drain_inbox's finished-detection:
    finished = [n for n, e in list(ctx.active_teammates.items()) if e.is_set()]
    assert "ti" in finished
    # After drain pops it:
    ctx.active_teammates.pop("ti", None)
    assert "ti" not in ctx.active_teammates


# --------------------------------------------------------------------------- #
# Turn-budget tests (Fix #1A + #1C)
# --------------------------------------------------------------------------- #

def test_claim_refused_when_turn_budget_low(monkeypatch):
    """Fix #1C: when remaining turns < CLAIM_MIN_TURNS, the claim_task
    tool handler returns a refusal message instead of actually claiming.

    We script 45 tool-call turns (each calls 'bash echo'), then one
    final tool-call turn that tries claim_task.  At that point
    turns_used == 45, remaining == 5 < CLAIM_MIN_TURNS(10), so the
    claim must be refused.

    We verify by inspecting the claim_task tool **output** in the team
    history (not the task status), because idle_poll may auto-claim the
    task afterwards -- that is a separate code path.
    """
    from mcodecore import tasks as tasks_mod
    t = tasks_mod.create_task("target-for-claim")
    script = []
    # 45 bash tool-calls
    for i in range(45):
        script.append(FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc(f"b{i}", "bash", {"command": "echo hi"})]),
            finish_reason="tool_calls",
        ))
    # 46th turn: try to claim a task
    script.append(FakeResponse(
        FakeMessage(content=None, tool_calls=[_tc("c1", "claim_task", {"task_id": t.id})]),
        finish_reason="tool_calls",
    ))
    # 47th: stop
    script.append(FakeResponse(_msg("done"), finish_reason="stop"))

    _install_fake_client(monkeypatch, script)
    teammates.spawn_teammate_thread("tb1", "worker", "go")
    _wait_teammate("tb1", timeout=15)

    # The direct claim_task call must have been refused.
    claim_outputs = _tool_outputs("tb1", "claim_task")
    assert len(claim_outputs) >= 1, "expected at least one claim_task call"
    refusal = claim_outputs[0]
    assert "Cannot claim" in refusal, \
        f"expected refusal message, got: {refusal}"
    assert "turns left in budget" in refusal, \
        f"expected turn-budget refusal, got: {refusal}"


def test_claim_allowed_when_turn_budget_sufficient(monkeypatch):
    """Fix #1C: when remaining turns >= CLAIM_MIN_TURNS, claim proceeds
    normally.  We use 5 bash turns (remaining=45 >= 10), then claim."""
    from mcodecore import tasks as tasks_mod
    t = tasks_mod.create_task("claim-allowed")
    script = []
    for i in range(5):
        script.append(FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc(f"b{i}", "bash", {"command": "echo hi"})]),
            finish_reason="tool_calls",
        ))
    script.append(FakeResponse(
        FakeMessage(content=None, tool_calls=[_tc("c1", "claim_task", {"task_id": t.id})]),
        finish_reason="tool_calls",
    ))
    script.append(FakeResponse(_msg("done"), finish_reason="stop"))

    _install_fake_client(monkeypatch, script)
    teammates.spawn_teammate_thread("tb2", "worker", "go")
    _wait_teammate("tb2", timeout=15)

    t2 = tasks_mod.load_task(t.id)
    assert t2.owner == "tb2", f"task should be claimed by tb2, owner={t2.owner}"
    assert t2.status == "in_progress"


def test_turn_renewal_when_owning_inprogress(monkeypatch):
    """Fix #1A: after TURN_BUDGET(50) turns, if the worker still owns
    an in_progress task, the budget is renewed by TURN_BUDGET_RENEWAL(20).

    We script 50 bash tool-calls, then claim_task on turn 51 (within
    the renewed budget), complete_task, then stop.

    Without renewal, the worker would hit idle_poll after 50 turns
    and timeout (since idle is patched to 1s).  With renewal, it gets
    20 more turns and can complete the task.

    Note: after 50 bash turns the inner loop exits, and idle_poll may
    auto-claim the task before the scripted claim_task tool fires.
    Either way, the task must end up completed.
    """
    from mcodecore import tasks as tasks_mod
    t = tasks_mod.create_task("renewal-target")
    script = []
    # 50 bash tool-calls to exhaust the soft cap
    for i in range(50):
        script.append(FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc(f"b{i}", "bash", {"command": "echo hi"})]),
            finish_reason="tool_calls",
        ))
    # Turn 51: claim the task (may show "already owned" if idle_poll
    # auto-claimed it first; either way the task is owned by tb3).
    script.append(FakeResponse(
        FakeMessage(content=None, tool_calls=[_tc("c1", "claim_task", {"task_id": t.id})]),
        finish_reason="tool_calls",
    ))
    # Turn 52: complete the task
    script.append(FakeResponse(
        FakeMessage(content=None, tool_calls=[_tc("ct1", "complete_task", {"task_id": t.id})]),
        finish_reason="tool_calls",
    ))
    # Turn 53: stop
    script.append(FakeResponse(_msg("all done"), finish_reason="stop"))

    _install_fake_client(monkeypatch, script)
    teammates.spawn_teammate_thread("tb3", "worker", "go")
    _wait_teammate("tb3", timeout=20)

    # The task must be completed (not stuck in_progress)
    t2 = tasks_mod.load_task(t.id)
    assert t2.status == "completed", f"task should be completed, status={t2.status}"
    assert t2.owner == "tb3"


def test_hard_cap_prevents_infinite_renewal(monkeypatch):
    """Fix #1A + #4: TURN_BUDGET_HARD_CAP(100) prevents infinite renewal.

    We script exactly 100 tool-call responses (1 claim + 99 bash).
    After 100 total turns, no more renewal is granted.  The worker
    falls through to idle_poll which returns "work" (it owns a task),
    but Fix #4 breaks the loop at the hard cap so it does not spin
    forever.  The worker finishes cleanly and sends a result.
    """
    from mcodecore import tasks as tasks_mod
    t = tasks_mod.create_task("hardcap-target")
    script = []
    # Turn 1: claim
    script.append(FakeResponse(
        FakeMessage(content=None, tool_calls=[_tc("c0", "claim_task", {"task_id": t.id})]),
        finish_reason="tool_calls",
    ))
    # Turns 2-100: bash (99 more, total 100)
    for i in range(99):
        script.append(FakeResponse(
            FakeMessage(content=None, tool_calls=[_tc(f"b{i}", "bash", {"command": "echo hi"})]),
            finish_reason="tool_calls",
        ))

    _install_fake_client(monkeypatch, script)
    teammates.spawn_teammate_thread("tb4", "worker", "go")
    _wait_teammate("tb4", timeout=30)

    # Worker should have finished cleanly (not crashed via StopIteration)
    assert ctx.teammate_registry["tb4"]["status"] == "finished"
    # Task should still be in_progress (never completed)
    lead_inbox = ctx.bus.read_inbox("lead")
    result_msgs = [m for m in lead_inbox
                   if m.get("type") == "result" and m.get("from") == "tb4"]
    assert len(result_msgs) >= 1
