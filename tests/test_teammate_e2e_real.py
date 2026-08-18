"""End-to-end teammate lifecycle tests.

This suite exercises the **full teammate lifecycle** using the real
``spawn_teammate_thread`` / ``run`` code path with a **mocked LLM
client** (so tests are deterministic and don't hit the network).  Every
public API that the lead and teammate use is driven through its real
implementation:

    spawn_teammate_thread     -- lead summons a teammate
    _teammate_submit_plan     -- teammate submits a plan (via tool call)
    run_review_plan           -- lead approves / rejects the plan
    run_send_message          -- lead <-> teammate messaging
    run_request_shutdown      -- lead requests graceful shutdown
    consume_lead_inbox        -- lead drains its mailbox
    _drain_inbox (simulated)  -- lead detects finished teammates via event
    create_task / claim_task  -- task board interaction
    complete_task             -- teammate completes a claimed task

Test coverage matrix:

    Phase          | Test(s)
    ----------------|--------------------------------------------------
    A. Spawn        | test_spawn_registers_teammate
                   | test_duplicate_spawn_rejected
    B. Messaging    | test_lead_to_teammate_message
                   | test_teammate_to_lead_message_via_tool
                   | test_bidirectional_messaging_roundtrip
    C. Plan flow    | test_plan_submit_and_approve
                   | test_plan_submit_and_reject
    D. Task board   | test_teammate_claims_and_completes_task
                   | test_idle_poll_auto_claims_unclaimed_task
    E. Exit paths   | test_normal_finish_lead_detects_both_channels
                   | test_idle_timeout_exit
                   | test_llm_failure_exit
                   | test_shutdown_request_exit
                   | test_tool_exception_crashed_notification
    F. Lead detect  | test_lead_detects_finished_via_event
                   | test_lead_busy_does_not_notice_until_drain
                   | test_multiple_teammates_concurrent
    G. Regression   | test_mailbox_flood_does_not_mask_notification
                   | test_finished_teammate_cleaned_from_registry
                   | test_teammate_uses_file_tools
                   | test_teammate_chain_via_messages
                   | test_shutdown_during_idle_with_pending_inbox

Design principles:
  1.  **No production code changes** -- only test code.
  2.  **Mocked LLM** via ``ScriptedClient`` so each turn is deterministic.
  3.  **Real bus / tasks / context** -- the actual ``MessageBus``,
      ``Task`` board, and ``ctx`` singleton are used (paths isolated by
      the ``isolate_paths`` autouse fixture in ``conftest.py``).
  4.  **Fast idle** -- ``IDLE_POLL_INTERVAL`` / ``IDLE_TIMEOUT`` are
      shortened so idle-timeout tests finish in ~1s instead of 60s.
  5.  **Synchronisation events** for plan-approval tests: the scripted
      LLM's second turn blocks until the lead has sent the approval,
      eliminating the race between teammate re-entry and lead action.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from mcodecore import teammates, bus, tasks as task_mod
from mcodecore.bus import (
    _teammate_submit_plan, run_review_plan, run_send_message,
    run_request_shutdown, consume_lead_inbox,
)
from mcodecore.config import client as real_client
from mcodecore.context import ctx


# --------------------------------------------------------------------------- #
# Fake LLM objects
# --------------------------------------------------------------------------- #

class FakeToolCall:
    """Mimics the OpenAI SDK tool-call object."""

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
    responses (or raises scripted exceptions) in FIFO sequence.

    Each item in the script is either:
      - a ``FakeResponse``  -> returned as-is
      - an ``Exception``    -> raised
      - a ``callable``      -> called with no args; its return value
        is used as the item (allows blocking on an Event for
        synchronisation, or dynamic response generation)

    Thread-safety: a lock protects the shared index, so concurrent
    teammates consume items in FIFO order of their ``create()`` calls.
    """

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
        # If the item is callable (e.g. a lambda that blocks on an
        # event), call it to resolve the actual response/exception.
        if callable(item) and not isinstance(item, BaseException):
            item = item()
        if isinstance(item, BaseException):
            raise item
        return item


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _install_fake_client(monkeypatch, script):
    """Replace ``client.chat.completions.create`` with a scripted fake."""
    fake = ScriptedClient(script)
    monkeypatch.setattr(real_client.chat.completions, "create", fake.create)
    return fake


def _msg(content):
    return FakeMessage(content=content)


def _tc(cid, name, args=None):
    return FakeToolCall(cid, name, json.dumps(args or {}))


def _resp_tool(msg_content, tool_calls):
    """Build a tool-call response (finish_reason='tool_calls')."""
    return FakeResponse(
        FakeMessage(content=msg_content, tool_calls=tool_calls),
        finish_reason="tool_calls")


def _resp_stop(content):
    """Build a normal stop response (finish_reason='stop')."""
    return FakeResponse(_msg(content), finish_reason="stop")


def _wait_teammate(name, timeout=10.0):
    """Block until the teammate's event is set (it finished)."""
    evt = ctx.active_teammates.get(name)
    assert evt is not None, f"teammate {name} not registered"
    assert evt.wait(timeout=timeout), \
        f"teammate {name} did not finish in {timeout}s"
    # Give the finally block a moment to complete registry/status update.
    time.sleep(0.05)


def _lead_inbox():
    """Read lead's inbox (destructive -- clears the mailbox)."""
    return ctx.bus.read_inbox("lead")


def _peek_lead_inbox():
    """Read lead's inbox WITHOUT clearing it (non-destructive peek).

    Useful for polling until a message arrives, then consuming via
    ``_lead_inbox`` or ``consume_lead_inbox``.
    """
    # NOTE: use ``bus.MAILBOX_DIR`` (patched by conftest's isolate_paths),
    # NOT ``config.MAILBOX_DIR`` (import-time binding, not patched).
    inbox = bus.MAILBOX_DIR / "lead.jsonl"
    if not inbox.exists():
        return []
    try:
        text = inbox.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _find_msg(inbox, msg_type=None, from_agent=None, content_contains=None):
    """Find messages matching criteria in an inbox list."""
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


def _simulate_drain_inbox(history=None):
    """Simulate ``agent._drain_inbox``: detect finished teammates,
    consume lead inbox, return (finished_names, inbox_msgs).

    This mirrors ``agent.py`` lines 159-174 exactly.
    """
    finished = [name for name, evt in list(ctx.active_teammates.items())
                if evt.is_set()]
    for name in finished:
        ctx.active_teammates.pop(name, None)
        if name in ctx.teammate_registry:
            ctx.teammate_registry[name]["status"] = "finished"

    inbox_msgs = consume_lead_inbox(route_protocol=True)
    return finished, inbox_msgs


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _fast_idle(monkeypatch):
    """Speed up ``idle_poll`` so idle-timeout tests finish quickly.

    ``idle_poll`` uses ``range(IDLE_TIMEOUT // IDLE_POLL_INTERVAL)``
    which requires integers, so patched values must be ``int``.
    """
    monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)
    # Skip backoff sleeps in retry paths so tests are fast.
    import mcodecore.teammates as _tm2
    monkeypatch.setattr(_tm2.time, "sleep", lambda s: None)


# --------------------------------------------------------------------------- #
# A. Spawn
# --------------------------------------------------------------------------- #

def test_spawn_registers_teammate(monkeypatch):
    """Spawning a teammate registers it in ``active_teammates`` and
    ``teammate_registry`` with status ``running``."""
    _install_fake_client(monkeypatch, [_resp_stop("done")])
    teammates.spawn_teammate_thread("e2e_a1", "worker", "hello")

    assert "e2e_a1" in ctx.active_teammates
    assert ctx.teammate_registry["e2e_a1"]["role"] == "worker"
    assert ctx.teammate_registry["e2e_a1"]["status"] == "running"
    _wait_teammate("e2e_a1")


def test_duplicate_spawn_rejected(monkeypatch):
    """Spawning a teammate with an existing name returns an error message
    and does NOT create a second thread."""
    _install_fake_client(monkeypatch, [_resp_stop("done")])
    teammates.spawn_teammate_thread("e2e_a2", "worker", "hello")
    result = teammates.spawn_teammate_thread("e2e_a2", "worker", "hello")
    assert "already exists" in result
    _wait_teammate("e2e_a2")


# --------------------------------------------------------------------------- #
# B. Messaging
# --------------------------------------------------------------------------- #

def test_lead_to_teammate_message(monkeypatch):
    """Lead sends a message to a teammate; the teammate receives it in
    its inbox and finishes normally."""
    _install_fake_client(monkeypatch, [_resp_stop("acknowledged")])
    teammates.spawn_teammate_thread("e2e_b1", "worker", "start")

    # Send a message from lead to teammate
    run_send_message("e2e_b1", "please review the code")
    _wait_teammate("e2e_b1")

    assert ctx.teammate_registry["e2e_b1"]["status"] == "finished"


def test_teammate_to_lead_message_via_tool(monkeypatch):
    """Teammate uses the ``send_message`` tool to send a message to lead.
    The message lands in lead's mailbox with type ``message``."""
    _install_fake_client(monkeypatch, [
        # Turn 1: teammate calls send_message tool
        _resp_tool(None, [_tc("c1", "send_message",
                              {"to": "lead", "content": "I found a bug!"})]),
        # Turn 2: teammate finishes
        _resp_stop("task complete"),
    ])
    teammates.spawn_teammate_thread("e2e_b2", "worker", "report findings")
    _wait_teammate("e2e_b2")

    lead_msgs = _lead_inbox()
    # Should have: the send_message result + the final result
    sent = _find_msg(lead_msgs, msg_type="message", from_agent="e2e_b2")
    assert len(sent) == 1
    assert "I found a bug!" in sent[0]["content"]

    results = _find_msg(lead_msgs, msg_type="result", from_agent="e2e_b2")
    assert len(results) == 1
    assert "task complete" in results[0]["content"]


def test_bidirectional_messaging_roundtrip(monkeypatch):
    """Full roundtrip: lead sends message -> teammate reads and responds
    -> lead receives response.

    The teammate's first LLM turn produces a tool_call to send_message.
    Before that turn, the lead sends a message to the teammate.  The
    teammate's inbox-read loop picks it up and injects it into the
    conversation, then the teammate responds.
    """
    _install_fake_client(monkeypatch, [
        # Turn 1: teammate responds to lead's message with send_message
        _resp_tool(None, [_tc("c1", "send_message",
                              {"to": "lead", "content": "got your message"})]),
        # Turn 2: teammate finishes
        _resp_stop("done"),
    ])
    teammates.spawn_teammate_thread("e2e_b3", "worker", "start")

    # Lead sends a message to the teammate
    run_send_message("e2e_b3", "what is the status?")
    _wait_teammate("e2e_b3")

    lead_msgs = _lead_inbox()
    responses = _find_msg(lead_msgs, msg_type="message", from_agent="e2e_b3")
    assert len(responses) == 1
    assert "got your message" in responses[0]["content"]


# --------------------------------------------------------------------------- #
# C. Plan approval flow
# --------------------------------------------------------------------------- #

def test_plan_submit_and_approve(monkeypatch):
    """Teammate submits a plan via the ``submit_plan`` tool.  Lead
    approves it.  The teammate receives the approval and continues.

    Synchronisation: the 2nd LLM call is wrapped in a lambda that blocks
    on an Event until the lead has approved the plan.  This eliminates
    the race between the teammate re-entering its loop and the lead
    sending the approval.
    """
    approved_evt = threading.Event()

    def _wait_then_stop():
        """Block until lead approves, then return the finish response."""
        approved_evt.wait(timeout=10)
        return _resp_stop("plan executed successfully")

    _install_fake_client(monkeypatch, [
        # Turn 1: teammate submits a plan
        _resp_tool(None, [_tc("c1", "submit_plan",
                              {"plan": "Step 1: read files. Step 2: write code."})]),
        # Turn 2: blocks until approval arrives, then returns a stop
        # response.  BUT the approval_response also lands in the
        # teammate's inbox, so idle_poll picks it up and returns "work",
        # giving the teammate a 3rd LLM turn.
        _wait_then_stop,
        # Turn 3: final finish
        _resp_stop("plan executed successfully"),
    ])
    teammates.spawn_teammate_thread("e2e_c1", "worker", "do work")

    # Wait for the plan to arrive in lead's inbox (non-destructive peek)
    deadline = time.time() + 5
    plan_msg = None
    while time.time() < deadline:
        msgs = _peek_lead_inbox()
        plans = _find_msg(msgs, msg_type="plan_approval_request")
        if plans:
            plan_msg = plans[0]
            break
        time.sleep(0.05)

    assert plan_msg is not None, "plan_approval_request not received"
    req_id = plan_msg["metadata"]["request_id"]
    assert req_id in ctx.pending_requests
    assert ctx.pending_requests[req_id].status == "pending"

    # Consume the plan_approval_request so it doesn't linger in inbox
    _lead_inbox()  # drain plan request

    # Lead approves the plan (sends plan_approval_response to teammate)
    result = run_review_plan(req_id, approve=True, feedback="looks good")
    assert "approved" in result
    assert ctx.pending_requests[req_id].status == "approved"

    # Unblock the teammate's 2nd LLM turn
    approved_evt.set()

    _wait_teammate("e2e_c1")

    # Teammate should have finished with a result
    lead_msgs = _lead_inbox()
    results = _find_msg(lead_msgs, msg_type="result", from_agent="e2e_c1")
    assert len(results) == 1
    assert "plan executed successfully" in results[0]["content"]


def test_plan_submit_and_reject(monkeypatch):
    """Teammate submits a plan; lead rejects it with feedback.  The
    teammate receives the rejection feedback and finishes."""
    rejected_evt = threading.Event()

    def _wait_then_stop():
        rejected_evt.wait(timeout=10)
        return _resp_stop("will revise")

    _install_fake_client(monkeypatch, [
        # Turn 1: teammate submits a plan
        _resp_tool(None, [_tc("c1", "submit_plan", {"plan": "bad plan"})]),
        # Turn 2: blocks until rejection arrives, then returns a stop
        # response.  idle_poll finds the rejection_response and returns
        # "work", giving a 3rd LLM turn.
        _wait_then_stop,
        # Turn 3: final finish
        _resp_stop("will revise"),
    ])
    teammates.spawn_teammate_thread("e2e_c2", "worker", "do work")

    # Wait for plan (non-destructive peek)
    deadline = time.time() + 5
    plan_msg = None
    while time.time() < deadline:
        msgs = _peek_lead_inbox()
        plans = _find_msg(msgs, msg_type="plan_approval_request")
        if plans:
            plan_msg = plans[0]
            break
        time.sleep(0.05)

    assert plan_msg is not None
    req_id = plan_msg["metadata"]["request_id"]

    # Consume the plan_approval_request
    _lead_inbox()  # drain

    # Lead rejects
    result = run_review_plan(req_id, approve=False, feedback="needs more detail")
    assert "rejected" in result
    assert ctx.pending_requests[req_id].status == "rejected"

    # Unblock the teammate's 2nd LLM turn
    rejected_evt.set()

    _wait_teammate("e2e_c2")

    lead_msgs = _lead_inbox()
    results = _find_msg(lead_msgs, msg_type="result", from_agent="e2e_c2")
    assert len(results) == 1


# --------------------------------------------------------------------------- #
# D. Task board interaction
# --------------------------------------------------------------------------- #

def test_teammate_claims_and_completes_task(monkeypatch):
    """Teammate uses ``claim_task`` and ``complete_task`` tools.  The
    task transitions pending -> in_progress -> completed."""
    # Create a task on the board
    task = task_mod.create_task("E2E test task", "do something")
    task_id = task.id

    _install_fake_client(monkeypatch, [
        # Turn 1: teammate claims the task
        _resp_tool(None, [_tc("c1", "claim_task", {"task_id": task_id})]),
        # Turn 2: teammate completes the task
        _resp_tool(None, [_tc("c2", "complete_task", {"task_id": task_id})]),
        # Turn 3: teammate finishes
        _resp_stop("task done"),
    ])
    teammates.spawn_teammate_thread("e2e_d1", "worker", "work on task")
    _wait_teammate("e2e_d1")

    # Verify task transitions
    t = task_mod.load_task(task_id)
    assert t.status == "completed"
    assert t.owner == "e2e_d1"

    # Teammate should have sent a result
    lead_msgs = _lead_inbox()
    results = _find_msg(lead_msgs, msg_type="result", from_agent="e2e_d1")
    assert len(results) == 1


def test_idle_poll_auto_claims_unclaimed_task(monkeypatch):
    """When a teammate finishes its first turn and enters ``idle_poll``,
    if there are unclaimed tasks on the board, it auto-claims one and
    returns to work."""
    # Create an unclaimed task BEFORE spawning the teammate
    task = task_mod.create_task("Auto-claim task", "auto work")
    task_id = task.id

    _install_fake_client(monkeypatch, [
        # Turn 1: teammate responds and finishes (breaks out of for-loop)
        _resp_stop("initial work done"),
        # Turn 2: after idle_poll auto-claims and returns "work",
        # teammate gets another LLM turn and finishes
        _resp_stop("auto-claimed task done"),
    ])
    teammates.spawn_teammate_thread("e2e_d2", "worker", "start")
    _wait_teammate("e2e_d2", timeout=15)

    # The task should have been auto-claimed by the teammate
    t = task_mod.load_task(task_id)
    assert t.owner == "e2e_d2"
    assert t.status in ("in_progress", "completed")


# --------------------------------------------------------------------------- #
# E. Exit paths
# --------------------------------------------------------------------------- #

def test_normal_finish_lead_detects_both_channels(monkeypatch):
    """P1: normal finish.  Lead detects via BOTH event channel (status
    finished) AND mailbox (result message)."""
    _install_fake_client(monkeypatch, [_resp_stop("all done")])
    teammates.spawn_teammate_thread("e2e_e1", "worker", "hello")
    _wait_teammate("e2e_e1")

    # Channel 1: event-based
    assert ctx.teammate_registry["e2e_e1"]["status"] == "finished"

    # Channel 2: mailbox
    lead_msgs = _lead_inbox()
    results = _find_msg(lead_msgs, msg_type="result", from_agent="e2e_e1")
    assert len(results) == 1
    assert "all done" in results[0]["content"]

    # Simulate _drain_inbox: should detect the finished teammate
    finished, inbox_msgs = _simulate_drain_inbox()
    assert "e2e_e1" in finished
    assert "e2e_e1" not in ctx.active_teammates  # cleaned up


def test_idle_timeout_exit(monkeypatch):
    """P3: idle timeout.  Teammate finishes first turn, enters idle_poll,
    times out, and sends a result to lead."""
    _install_fake_client(monkeypatch, [_resp_stop("started")])
    teammates.spawn_teammate_thread("e2e_e3", "worker", "hello")
    _wait_teammate("e2e_e3", timeout=10)

    assert ctx.teammate_registry["e2e_e3"]["status"] == "finished"
    lead_msgs = _lead_inbox()
    results = _find_msg(lead_msgs, msg_type="result", from_agent="e2e_e3")
    assert len(results) == 1


def test_llm_failure_exit(monkeypatch):
    """P2: LLM call failure.  Teammate sends an error message to lead
    and exits.

    ConnectionError("API is down") is transient -> retried up to
    MAX_REACTIVE_RETRIES=3, then error message is sent.
    """
    _install_fake_client(monkeypatch, [
        ConnectionError("API is down"),
        ConnectionError("API is down"),
        ConnectionError("API is down"),
        ConnectionError("API is down"),  # 4th -> retries exhausted
    ])
    teammates.spawn_teammate_thread("e2e_e2", "worker", "hello")
    _wait_teammate("e2e_e2", timeout=10)

    assert ctx.teammate_registry["e2e_e2"]["status"] == "finished"
    lead_msgs = _lead_inbox()
    errors = _find_msg(lead_msgs, msg_type="LLM API error", from_agent="e2e_e2")
    assert len(errors) == 1
    assert "API is down" in errors[0]["content"]


def test_shutdown_request_exit(monkeypatch):
    """P4: shutdown request.  Lead sends a shutdown_request; the teammate
    approves it and exits gracefully."""
    _install_fake_client(monkeypatch, [
        # Turn 1: teammate starts and finishes (breaks out of for-loop)
        _resp_stop("working"),
    ])
    teammates.spawn_teammate_thread("e2e_e4", "worker", "hello")

    # Wait for teammate to finish its first turn and enter idle_poll
    time.sleep(0.3)

    # Lead requests shutdown
    result = run_request_shutdown("e2e_e4")
    assert "Shutdown request sent" in result

    _wait_teammate("e2e_e4", timeout=10)

    assert ctx.teammate_registry["e2e_e4"]["status"] == "finished"
    lead_msgs = _lead_inbox()
    # Should have a shutdown_response
    shutdowns = _find_msg(lead_msgs, msg_type="shutdown_response",
                          from_agent="e2e_e4")
    assert len(shutdowns) == 1
    assert shutdowns[0]["metadata"]["approve"] is True


def test_tool_exception_crashed_notification(monkeypatch):
    """P5 (after Fix #5): a tool handler that raises is now caught and
    fed back to the LLM as an error string (instead of crashing). The
    teammate continues, the LLM calls finish, and a result message is
    delivered -- no 'crashed' notification."""
    _install_fake_client(monkeypatch, [
        # Turn 1: teammate calls bash, which we'll make raise
        _resp_tool(None, [_tc("c1", "bash", {"command": "echo hi"})]),
        # Turn 2: error fed back, LLM calls finish to report
        _resp_tool(None, [_tc("c2", "finish", {"summary": "tool error handled"})]),
    ])

    # Make run_bash raise during this test
    import mcodecore.fsops as fsops
    def _boom_bash(*args, **kwargs):
        raise RuntimeError("simulated tool crash")
    monkeypatch.setattr(fsops, "run_bash", _boom_bash)

    teammates.spawn_teammate_thread("e2e_e5", "worker", "do work")
    _wait_teammate("e2e_e5", timeout=10)

    assert ctx.teammate_registry["e2e_e5"]["status"] == "finished"
    lead_msgs = _lead_inbox()

    # The teammate should NOT have crashed -- it should have delivered
    # a result message via the finish tool.
    results = _find_msg(lead_msgs, msg_type="result", from_agent="e2e_e5")
    assert len(results) == 1, (
        f"expected 1 result message from e2e_e5 (via finish after tool "
        f"error), got {len(results)}; inbox types: "
        f"{[(m['from'], m['type']) for m in lead_msgs]}"
    )
    # No crashed notification should be present
    crashed = _find_msg(lead_msgs, msg_type="crashed", from_agent="e2e_e5")
    assert len(crashed) == 0, (
        f"tool error should be fed back, not cause a crash; "
        f"found crashed: {crashed}"
    )


# --------------------------------------------------------------------------- #
# F. Lead detection
# --------------------------------------------------------------------------- #

def test_lead_detects_finished_via_event(monkeypatch):
    """Lead's ``_drain_inbox`` detects a finished teammate purely via
    the event channel, even if the mailbox is empty."""
    _install_fake_client(monkeypatch, [_resp_stop("done")])
    teammates.spawn_teammate_thread("e2e_f1", "worker", "hello")

    # Consume the result message before drain (simulate lead was busy
    # and a previous drain already ate the mailbox)
    _lead_inbox()  # drain and discard

    _wait_teammate("e2e_f1")

    # Now simulate drain: event-based detection should still work
    finished, inbox_msgs = _simulate_drain_inbox()
    assert "e2e_f1" in finished
    assert "e2e_f1" not in ctx.active_teammates


def test_lead_busy_does_not_notice_until_drain(monkeypatch):
    """When the lead is 'busy' (not calling _drain_inbox), a teammate
    can finish and the lead won't notice until it drains.  This tests
    the timing window: the event IS set immediately, but detection only
    happens when drain runs."""
    _install_fake_client(monkeypatch, [_resp_stop("done")])
    teammates.spawn_teammate_thread("e2e_f2", "worker", "hello")
    _wait_teammate("e2e_f2")

    # Before drain: teammate is finished (event set) but still in
    # active_teammates (lead hasn't drained yet)
    assert "e2e_f2" in ctx.active_teammates
    assert ctx.teammate_registry["e2e_f2"]["status"] == "finished"

    # After drain: cleaned up
    _simulate_drain_inbox()
    assert "e2e_f2" not in ctx.active_teammates


def test_multiple_teammates_concurrent(monkeypatch):
    """Multiple teammates run concurrently.  Each is independently
    detected by the lead's drain.

    All teammates share the same FIFO script queue (thread-safe).  Each
    teammate consumes one response and finishes.
    """
    _install_fake_client(monkeypatch, [
        _resp_stop("done"),
        _resp_stop("done"),
        _resp_stop("done"),
    ])
    # Spawn all three teammates concurrently — the bus write-lock
    # now guarantees no message is lost when multiple teammates
    # send their results to the lead's inbox simultaneously.
    teammates.spawn_teammate_thread("e2e_f3a", "worker", "hello")
    teammates.spawn_teammate_thread("e2e_f3b", "worker", "hello")
    teammates.spawn_teammate_thread("e2e_f3c", "worker", "hello")

    _wait_teammate("e2e_f3a")
    _wait_teammate("e2e_f3b")
    _wait_teammate("e2e_f3c")

    # All three should be finished
    for name in ("e2e_f3a", "e2e_f3b", "e2e_f3c"):
        assert ctx.teammate_registry[name]["status"] == "finished"

    # Read inbox BEFORE drain (drain consumes the inbox)
    lead_msgs = _lead_inbox()
    for name in ("e2e_f3a", "e2e_f3b", "e2e_f3c"):
        results = _find_msg(lead_msgs, msg_type="result", from_agent=name)
        assert len(results) == 1, f"expected result from {name}"
        assert "done" in results[0]["content"]

    # Drain should detect all three (inbox already consumed, but
    # event-based detection still works)
    finished, _ = _simulate_drain_inbox()
    assert set(finished) == {"e2e_f3a", "e2e_f3b", "e2e_f3c"}


# --------------------------------------------------------------------------- #
# G. Regression / edge cases
# --------------------------------------------------------------------------- #

def test_mailbox_flood_does_not_mask_notification(monkeypatch):
    """If the lead's mailbox receives many messages (flood), the
    teammate's result message is still present and detectable.

    This disproves the 'mailbox overflow' hypothesis: ``read_inbox``
    returns ALL messages, and ``_drain_inbox`` processes them all.
    """
    _install_fake_client(monkeypatch, [
        # Turn 1: teammate sends 3 messages to lead
        _resp_tool(None, [
            _tc("c1", "send_message", {"to": "lead", "content": "msg 1"}),
        ]),
        _resp_tool(None, [
            _tc("c2", "send_message", {"to": "lead", "content": "msg 2"}),
        ]),
        _resp_tool(None, [
            _tc("c3", "send_message", {"to": "lead", "content": "msg 3"}),
        ]),
        # Turn 4: finish
        _resp_stop("final result"),
    ])
    teammates.spawn_teammate_thread("e2e_g1", "worker", "flood test")
    _wait_teammate("e2e_g1", timeout=15)

    lead_msgs = _lead_inbox()
    # All messages present
    assert len(lead_msgs) >= 4  # 3 send_message + 1 result
    # Result is there
    results = _find_msg(lead_msgs, msg_type="result", from_agent="e2e_g1")
    assert len(results) == 1
    assert "final result" in results[0]["content"]


def test_finished_teammate_cleaned_from_registry(monkeypatch):
    """After ``_drain_inbox`` detects a finished teammate, it is removed
    from ``active_teammates``.  The registry entry remains (with status
    ``finished``) for historical reference."""
    _install_fake_client(monkeypatch, [_resp_stop("done")])
    teammates.spawn_teammate_thread("e2e_g2", "worker", "hello")
    _wait_teammate("e2e_g2")

    assert "e2e_g2" in ctx.active_teammates  # before drain
    _simulate_drain_inbox()
    assert "e2e_g2" not in ctx.active_teammates  # after drain
    assert ctx.teammate_registry["e2e_g2"]["status"] == "finished"


def test_teammate_uses_file_tools(monkeypatch):
    """Teammate uses ``write_file`` and ``read_file`` tools to actually
    do work.  Verifies the file-system tool handlers work inside the
    teammate thread."""
    _install_fake_client(monkeypatch, [
        # Turn 1: teammate writes a file
        _resp_tool(None, [_tc("c1", "write_file",
                              {"path": "e2e_test_output.txt",
                               "content": "hello from teammate"})]),
        # Turn 2: teammate reads it back
        _resp_tool(None, [_tc("c2", "read_file",
                              {"path": "e2e_test_output.txt"})]),
        # Turn 3: teammate finishes
        _resp_stop("file operations complete"),
    ])
    teammates.spawn_teammate_thread("e2e_g3", "worker", "write and read a file")
    _wait_teammate("e2e_g3", timeout=15)

    # The file should exist (in the isolated tmp workspace).
    # NOTE: use ``fsops.WORKDIR`` (which conftest patches to tmp_path),
    # NOT ``config.WORKDIR`` (import-time binding, not patched).
    from mcodecore.fsops import WORKDIR as FSOPS_WORKDIR
    out_file = FSOPS_WORKDIR / "e2e_test_output.txt"
    assert out_file.exists()
    assert "hello from teammate" in out_file.read_text()

    # Teammate should have sent a result
    lead_msgs = _lead_inbox()
    results = _find_msg(lead_msgs, msg_type="result", from_agent="e2e_g3")
    assert len(results) == 1


def test_teammate_chain_via_messages(monkeypatch):
    """Two teammates communicate indirectly through the lead: teammate A
    sends a result, lead (simulated) forwards it to teammate B, B finishes.

    Both teammates share a FIFO script queue.  A runs first (consumes
    item 0), finishes, then B is spawned (consumes item 1), finishes.
    """
    _install_fake_client(monkeypatch, [
        # Teammate A: finishes with a result
        _resp_stop("analysis complete: found 3 issues"),
        # Teammate B: finishes with a result
        _resp_stop("fixes applied"),
    ])
    teammates.spawn_teammate_thread("e2e_g4a", "analyst", "analyze code")
    _wait_teammate("e2e_g4a")

    # Lead receives A's result
    lead_msgs = _lead_inbox()
    a_results = _find_msg(lead_msgs, msg_type="result", from_agent="e2e_g4a")
    assert len(a_results) == 1
    a_content = a_results[0]["content"]

    # Lead spawns teammate B and forwards A's result
    run_send_message("e2e_g4b", f"From analyst: {a_content}")
    teammates.spawn_teammate_thread("e2e_g4b", "fixer", "fix issues")
    _wait_teammate("e2e_g4b")

    lead_msgs = _lead_inbox()
    b_results = _find_msg(lead_msgs, msg_type="result", from_agent="e2e_g4b")
    assert len(b_results) == 1
    assert "fixes applied" in b_results[0]["content"]


def test_shutdown_during_idle_with_pending_inbox(monkeypatch):
    """If a teammate has pending inbox messages AND receives a shutdown
    request, the shutdown takes priority (handle_inbox returns True for
    shutdown_request, breaking the inbox loop)."""
    _install_fake_client(monkeypatch, [
        # Turn 1: teammate starts and finishes (enters idle_poll)
        _resp_stop("started"),
    ])
    teammates.spawn_teammate_thread("e2e_g5", "worker", "hello")

    # Wait for teammate to enter idle
    time.sleep(0.3)

    # Send both a regular message and a shutdown request
    run_send_message("e2e_g5", "regular message")
    run_request_shutdown("e2e_g5")

    _wait_teammate("e2e_g5", timeout=10)

    assert ctx.teammate_registry["e2e_g5"]["status"] == "finished"
    lead_msgs = _lead_inbox()
    # Should have shutdown_response
    shutdowns = _find_msg(lead_msgs, msg_type="shutdown_response",
                          from_agent="e2e_g5")
    assert len(shutdowns) == 1
