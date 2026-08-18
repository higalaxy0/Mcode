"""Message bus and protocol state machine tests.

Covers ``mcodecore.bus``:
MessageBus send/read_inbox, the ProtocolState dataclass, match_response,
consume_lead_inbox, idle_poll (return-value semantics only), and protocol
operation handlers.
"""

from __future__ import annotations

import pytest

from mcodecore import bus
from mcodecore.context import ctx


@pytest.fixture
def mb():
    return bus.MessageBus()


# --------------------------------------------------------------------------- #
# MessageBus basic send/read
# --------------------------------------------------------------------------- #

def test_messagebus_send_and_read(mb):
    mb.send("alice", "bob", "hello")
    inbox = mb.read_inbox("bob")
    assert len(inbox) == 1
    assert inbox[0]["from"] == "alice"
    assert inbox[0]["content"] == "hello"


def test_messagebus_read_clears_inbox(mb):
    mb.send("a", "b", "m1")
    mb.read_inbox("b")
    second = mb.read_inbox("b")
    assert second == []


def test_messagebus_empty_inbox(mb):
    assert mb.read_inbox("nobody") == []


def test_messagebus_message_type_default(mb):
    mb.send("a", "b", "msg")
    m = mb.read_inbox("b")[0]
    assert m["type"] == "message"


def test_messagebus_custom_type(mb):
    mb.send("a", "b", "plan text", "plan_approval_request",
            {"request_id": "req_1"})
    m = mb.read_inbox("b")[0]
    assert m["type"] == "plan_approval_request"
    assert m["metadata"]["request_id"] == "req_1"


def test_messagebus_multiple_messages_order(mb):
    for i in range(3):
        mb.send("a", "b", f"m{i}")
    inbox = mb.read_inbox("b")
    assert [m["content"] for m in inbox] == ["m0", "m1", "m2"]


def test_messagebus_message_has_timestamp(mb):
    mb.send("a", "b", "x")
    m = mb.read_inbox("b")[0]
    assert "ts" in m and isinstance(m["ts"], float)


# --------------------------------------------------------------------------- #
# ProtocolState dataclass
# --------------------------------------------------------------------------- #

def test_protocol_state_fields():
    ps = bus.ProtocolState(
        request_id="req_1", type="plan_approval",
        sender="tm1", target="lead", status="pending", payload="my plan")
    assert ps.request_id == "req_1"
    assert ps.type == "plan_approval"
    assert ps.status == "pending"
    assert ps.payload == "my plan"
    assert ps.created_at > 0  # default factory


# --------------------------------------------------------------------------- #
# match_response
# --------------------------------------------------------------------------- #

def test_match_response_unknown_request_id():
    """Prints a warning and returns None for an unknown request_id."""
    ctx.pending_requests.clear()
    result = bus.match_response("plan_approval_response", "req_unknown", True)
    assert result is None


def test_match_response_type_mismatch():
    state = bus.ProtocolState(
        request_id="req_2", type="shutdown",
        sender="lead", target="tm1", status="pending", payload="")
    ctx.pending_requests["req_2"] = state
    bus.match_response("plan_approval_response", "req_2", True)
    assert state.status == "pending"  # type mismatch, status unchanged


def test_match_response_shutdown_approved():
    state = bus.ProtocolState(
        request_id="req_3", type="shutdown",
        sender="lead", target="tm1", status="pending", payload="")
    ctx.pending_requests["req_3"] = state
    bus.match_response("shutdown_response", "req_3", True)
    assert state.status == "approved"


def test_match_response_plan_rejected():
    state = bus.ProtocolState(
        request_id="req_4", type="plan_approval",
        sender="tm1", target="lead", status="pending", payload="plan")
    ctx.pending_requests["req_4"] = state
    bus.match_response("plan_approval_response", "req_4", False)
    assert state.status == "rejected"


# --------------------------------------------------------------------------- #
# consume_lead_inbox
# --------------------------------------------------------------------------- #

def test_consume_lead_inbox_empty():
    assert bus.consume_lead_inbox() == []


def test_consume_lead_inbox_returns_messages():
    ctx.bus.send("alice", "lead", "hello lead")
    msgs = bus.consume_lead_inbox(route_protocol=False)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "hello lead"


def test_consume_lead_inbox_routes_protocol_response():
    state = bus.ProtocolState(
        request_id="req_5", type="plan_approval",
        sender="tm1", target="lead", status="pending", payload="plan")
    ctx.pending_requests["req_5"] = state
    ctx.bus.send("tm1", "lead", "approved", "plan_approval_response",
                 {"request_id": "req_5", "approve": True})
    bus.consume_lead_inbox(route_protocol=True)
    assert state.status == "approved"


# --------------------------------------------------------------------------- #
# run_send_message / run_check_inbox
# --------------------------------------------------------------------------- #

def test_run_send_message_delivers():
    res = bus.run_send_message("bob", "hi there")
    assert "Sent" in res
    assert "bob" in res
    inbox = ctx.bus.read_inbox("bob")
    assert len(inbox) == 1
    assert inbox[0]["content"] == "hi there"


def test_run_check_inbox_returns_messages():
    ctx.bus.send("alice", "lead", "hello lead")
    res = bus.run_check_inbox(include_read=False)
    assert "hello lead" in res


def test_run_check_inbox_empty():
    res = bus.run_check_inbox(include_read=False)
    assert "empty" in res


def test_run_check_inbox_string_bool_coercion():
    ctx.bus.send("alice", "lead", "msg")
    res = bus.run_check_inbox(include_read="true")
    assert "msg" in res


# --------------------------------------------------------------------------- #
# Protocol operation handlers
# --------------------------------------------------------------------------- #

def test_run_request_shutdown_creates_state_and_sends():
    res = bus.run_request_shutdown("teammate1")
    assert "teammate1" in res
    # state was registered
    states = list(ctx.pending_requests.values())
    assert any(s.type == "shutdown" and s.target == "teammate1" for s in states)
    # teammate received the shutdown_request
    inbox = ctx.bus.read_inbox("teammate1")
    assert any(m["type"] == "shutdown_request" for m in inbox)


def test_run_request_plan_sends_message():
    res = bus.run_request_plan("teammate2", "do the thing")
    assert "submit a plan" in res
    inbox = ctx.bus.read_inbox("teammate2")
    assert any("do the thing" in m["content"] for m in inbox)


def test_run_review_plan_approve():
    # First have the teammate submit a plan
    sub = bus._teammate_submit_plan("teammate3", "my plan steps")
    rid = [k for k, v in ctx.pending_requests.items()
           if v.sender == "teammate3"][0]
    res = bus.run_review_plan(rid, approve=True, feedback="good")
    assert "approved" in res
    assert ctx.pending_requests[rid].status == "approved"
    # teammate received the approval result
    inbox = ctx.bus.read_inbox("teammate3")
    assert any(m["type"] == "plan_approval_response" for m in inbox)


def test_run_review_plan_reject():
    bus._teammate_submit_plan("teammate4", "plan")
    rid = [k for k, v in ctx.pending_requests.items()
           if v.sender == "teammate4"][0]
    res = bus.run_review_plan(rid, approve=False, feedback="redo it")
    assert "rejected" in res
    assert ctx.pending_requests[rid].status == "rejected"


def test_run_review_plan_unknown_id():
    res = bus.run_review_plan("req_nonexistent", approve=True)
    assert "not found" in res


def test_run_review_plan_already_processed():
    bus._teammate_submit_plan("teammate5", "plan")
    rid = [k for k, v in ctx.pending_requests.items()
           if v.sender == "teammate5"][0]
    bus.run_review_plan(rid, approve=True)
    res = bus.run_review_plan(rid, approve=False)
    assert "already" in res


def test_teammate_submit_plan_registers_state():
    res = bus._teammate_submit_plan("tm6", "steps")
    assert "submitted" in res.lower()
    states = [v for v in ctx.pending_requests.values() if v.sender == "tm6"]
    assert len(states) == 1
    assert states[0].type == "plan_approval"
    assert states[0].payload == "steps"


# --------------------------------------------------------------------------- #
# idle_poll semantics (no real LLM call; only verifies timeout when there are no tasks or messages)
# Note: idle_poll sleeps IDLE_TIMEOUT//IDLE_POLL_INTERVAL times; the default 60//5=12 * 5s=60s is too slow, so monkeypatch shortens it here.
# --------------------------------------------------------------------------- #

def test_idle_poll_empty_returns_timeout(monkeypatch):
    monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)
    res = bus.idle_poll("solo", [], "solo", "worker")
    assert res == "timeout"


def test_idle_poll_shutdown_request_returns_shutdown(monkeypatch):
    monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)
    # Pre-place a shutdown_request
    ctx.bus.send("lead", "solo", "shutdown please", "shutdown_request",
                 {"request_id": "req_sd"})
    res = bus.idle_poll("solo", [], "solo", "worker")
    assert res == "shutdown"


# --------------------------------------------------------------------------- #
# Concurrent send – regression test for Windows data-loss bug
#
# Before the fix ``send`` used ``open(path, "a")`` without a lock.  On
# Windows this is **not** atomic: multiple threads appending to the same
# inbox concurrently lost messages (up to ~37 %) and occasionally produced
# corrupted (interleaved) JSON lines.  The fix adds a global
# ``_write_lock`` to serialise appends.
# --------------------------------------------------------------------------- #

def test_concurrent_send_no_data_loss(mb):
    """Concurrent ``send`` to the same inbox must not lose messages."""
    import threading

    N_THREADS = 10
    N_PER_THREAD = 50
    expected = N_THREADS * N_PER_THREAD
    barrier = threading.Barrier(N_THREADS)

    def worker(tid):
        barrier.wait()
        for i in range(N_PER_THREAD):
            mb.send(f"t{tid}", "lead", f"msg-{tid}-{i}", "message")

    threads = [threading.Thread(target=worker, args=(t,))
               for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    msgs = mb.read_inbox("lead")
    assert len(msgs) == expected, (
        f"Expected {expected} messages, got {len(msgs)} "
        f"({expected - len(msgs)} lost)"
    )


def test_concurrent_send_no_corruption(mb):
    """Concurrent ``send`` must not produce corrupted JSON lines."""
    import threading
    import json

    N_THREADS = 8
    N_PER_THREAD = 40
    barrier = threading.Barrier(N_THREADS)

    def worker(tid):
        barrier.wait()
        for i in range(N_PER_THREAD):
            # include metadata to make lines longer (more likely to
            # interleave if the lock were absent)
            mb.send(f"t{tid}", "shared", f"payload-{tid}-{i}",
                    "message", {"seq": i, "worker": tid})

    threads = [threading.Thread(target=worker, args=(t,))
               for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    msgs = mb.read_inbox("shared")
    # Every message must be valid JSON (read_inbox already parses them;
    # if it returned without raising, none were corrupted)
    for m in msgs:
        assert "from" in m
        assert "content" in m
        assert "metadata" in m
    assert len(msgs) == N_THREADS * N_PER_THREAD


def test_concurrent_send_different_inboxes(mb):
    """Concurrent ``send`` to *different* inboxes must also be safe."""
    import threading

    N = 5
    PER = 30
    barrier = threading.Barrier(N)

    def worker(tid):
        barrier.wait()
        for i in range(PER):
            mb.send(f"src{tid}", f"dst{tid}", f"msg-{i}")

    threads = [threading.Thread(target=worker, args=(t,))
               for t in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for tid in range(N):
        msgs = mb.read_inbox(f"dst{tid}")
        assert len(msgs) == PER, (
            f"dst{tid}: expected {PER}, got {len(msgs)}"
        )


def test_concurrent_send_and_read_interleaved(mb):
    """A reader must never see a half-written line.

    ``read_inbox`` renames the file atomically, so it either sees all
    messages written *before* the rename or none.  Concurrent writers
    appending after the rename go to the new file.  No message is lost
    or corrupted.
    """
    import threading

    N_WRITERS = 4
    PER_WRITER = 25
    received = []
    r_lock = threading.Lock()
    stop = threading.Event()

    def writer(tid):
        for i in range(PER_WRITER):
            mb.send(f"w{tid}", "rbox", f"w{tid}-{i}")

    def reader():
        while not stop.is_set():
            msgs = mb.read_inbox("rbox")
            if msgs:
                with r_lock:
                    received.extend(msgs)

    readers = [threading.Thread(target=reader) for _ in range(2)]
    for r in readers:
        r.start()

    writers = [threading.Thread(target=writer, args=(t,))
               for t in range(N_WRITERS)]
    for w in writers:
        w.start()
    for w in writers:
        w.join()

    # Give readers a final chance to drain
    import time
    time.sleep(0.2)
    stop.set()
    for r in readers:
        r.join()

    # Drain any remaining
    remaining = mb.read_inbox("rbox")
    with r_lock:
        received.extend(remaining)

    expected = N_WRITERS * PER_WRITER
    assert len(received) == expected, (
        f"Expected {expected}, got {len(received)}"
    )


def test_read_inbox_skips_corrupted_line(mb, monkeypatch, tmp_path):
    """``read_inbox`` should skip (not crash on) a corrupted JSON line."""
    import json as _json
    from mcodecore.bus import MAILBOX_DIR

    inbox = MAILBOX_DIR / "corrupt.jsonl"
    # Write one valid + one corrupted + one valid line
    good1 = _json.dumps({"from": "a", "to": "corrupt",
                         "content": "good1", "type": "message",
                         "ts": 1.0, "metadata": {}}) + "\n"
    bad = "{this is not valid json}\n"
    good2 = _json.dumps({"from": "a", "to": "corrupt",
                         "content": "good2", "type": "message",
                         "ts": 2.0, "metadata": {}}) + "\n"
    inbox.write_text(good1 + bad + good2, encoding="utf-8")

    msgs = mb.read_inbox("corrupt")
    assert len(msgs) == 2
    assert msgs[0]["content"] == "good1"
    assert msgs[1]["content"] == "good2"
