"""Multi-task scenario regression test.

This suite exercises the **full multi-task workflow** with multiple
teammates, complex dependency DAGs, and plan-approval flows -- all using
the real ``spawn_teammate_thread`` / ``run`` code path with a **mocked
LLM client** (ScriptedClient) so tests are deterministic.

Scenario topology:
    T1 (root, no deps)
        |
    +---+---+
    |       |
    T2      T3   (both blocked by T1)
    |       |
    +---+---+
        |
       T4      (blocked by T2 AND T3)

Plus plan-approval sub-tests where a worker submits a plan and the
lead approves or rejects.

Design principles (inherited from test_teammate_e2e_real.py):
  1.  Mocked LLM via ``ScriptedClient`` -- deterministic turns.
  2.  Real bus / tasks / context -- actual MessageBus, Task board,
      ctx singleton (paths isolated by conftest's isolate_paths).
  3.  Fast idle -- IDLE_POLL_INTERVAL / IDLE_TIMEOUT shortened,
      time.sleep patched to no-op for speed.
  4.  Synchronisation via blocking callables and threading.Event
      (NOT time.sleep, which is patched to no-op).
  5.  Auto-claim avoidance: when a worker finishes its scripted turns
      and enters idle_poll, any unclaimed+unblocked task on the board
      will be auto-claimed.  Tests avoid this by either:
        - Creating tasks just-in-time (after the previous worker finishes)
        - Pre-claiming tasks with dummy owners
        - Using blocking callables to keep the worker in its LLM call
          until the test is ready for it to exit.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from mcodecore import teammates, bus, tasks as task_mod
from mcodecore.bus import run_review_plan, run_request_shutdown
from mcodecore.config import client as real_client
from mcodecore.context import ctx


# --------------------------------------------------------------------------- #
# Fake LLM objects (same design as test_teammate_e2e_real.py)
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
    """Fake client.chat.completions.create returning scripted responses.

    Thread-safe: concurrent teammates consume items in FIFO order.
    Items can be FakeResponse objects, Exceptions, or zero-arg callables
    that return a FakeResponse (used for blocking/synchronisation).
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
        if callable(item) and not isinstance(item, BaseException):
            item = item()
        if isinstance(item, BaseException):
            raise item
        return item


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

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


def _block_until(evt, response, timeout=10.0):
    """Return a callable that blocks on *evt* then returns *response*.

    Used as a script item to synchronise worker execution with the test
    thread (e.g. wait for lead approval before continuing).
    """
    def _fn():
        evt.wait(timeout=timeout)
        return response
    return _fn


def _wait_teammate(name, timeout=15.0):
    evt = ctx.active_teammates.get(name)
    assert evt is not None, f"teammate {name} not registered"
    assert evt.wait(timeout=timeout), \
        f"teammate {name} did not finish in {timeout}s"
    # Small delay for the finally block to complete
    _spin_wait(lambda: ctx.teammate_registry.get(name, {}).get("status") == "finished",
               timeout=2.0)


def _spin_wait(fn, timeout=5.0, interval=0.01):
    """Poll *fn* until it returns True or timeout.

    Uses threading.Event.wait (NOT time.sleep which is patched to no-op)
    for the polling interval.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        threading.Event().wait(interval)
    return False


def _lead_inbox():
    return ctx.bus.read_inbox("lead")


def _peek_lead_inbox():
    inbox = bus.MAILBOX_DIR / "lead.jsonl"
    if not inbox.exists():
        return []
    try:
        text = inbox.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


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


def _wait_for_plan(timeout=5.0):
    """Poll lead inbox for a plan_approval_request, return the message."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msgs = _peek_lead_inbox()
        plans = _find_msg(msgs, msg_type="plan_approval_request")
        if plans:
            return plans[0]
        threading.Event().wait(0.02)
    return None


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _fast_idle(monkeypatch):
    """Shorten idle poll and patch time.sleep to no-op for speed.

    Note: patching time.sleep affects ALL modules (time is a singleton).
    Tests must use threading.Event or _spin_wait for synchronisation,
    NOT time.sleep.
    """
    monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)
    import mcodecore.teammates as _tm2
    monkeypatch.setattr(_tm2.time, "sleep", lambda s: None)


# --------------------------------------------------------------------------- #
# Test 1: Full DAG -- T1 -> T2,T3 -> T4
# --------------------------------------------------------------------------- #

class TestDagDependency:
    """Verify that task dependencies (blockedBy) are enforced end-to-end.

    DAG:  T1 -> T2, T3 -> T4
        - T1 has no deps, claimable immediately.
        - T2 and T3 are blocked by T1.
        - T4 is blocked by both T2 and T3.
    """

    def test_dag_all_complete(self, monkeypatch):
        """A single worker walks the entire DAG: claims T1, completes it,
        then claims T2, T3 (now unblocked), completes them, then claims
        T4 (now fully unblocked), completes it."""
        t1 = task_mod.create_task("T1-root")
        t2 = task_mod.create_task("T2-child", blockedBy=[t1.id])
        t3 = task_mod.create_task("T3-child", blockedBy=[t1.id])
        t4 = task_mod.create_task("T4-leaf", blockedBy=[t2.id, t3.id])

        # Before T1 completes: T2/T3/T4 should NOT be claimable
        assert not task_mod.can_start(t2.id)
        assert not task_mod.can_start(t3.id)
        assert not task_mod.can_start(t4.id)
        # T1 IS claimable
        assert task_mod.can_start(t1.id)

        _install_fake_client(monkeypatch, [
            _resp_tool(None, [_tc("c1", "claim_task", {"task_id": t1.id})]),
            _resp_tool(None, [_tc("c2", "complete_task", {"task_id": t1.id})]),
            _resp_tool(None, [_tc("c3", "claim_task", {"task_id": t2.id})]),
            _resp_tool(None, [_tc("c4", "complete_task", {"task_id": t2.id})]),
            _resp_tool(None, [_tc("c5", "claim_task", {"task_id": t3.id})]),
            _resp_tool(None, [_tc("c6", "complete_task", {"task_id": t3.id})]),
            _resp_tool(None, [_tc("c7", "claim_task", {"task_id": t4.id})]),
            _resp_tool(None, [_tc("c8", "complete_task", {"task_id": t4.id})]),
            _resp_stop("DAG complete"),
        ])
        teammates.spawn_teammate_thread("dag_w1", "worker", "do the DAG")
        _wait_teammate("dag_w1", timeout=15)

        # All four tasks should be completed
        for t in (t1, t2, t3, t4):
            loaded = task_mod.load_task(t.id)
            assert loaded.status == "completed", \
                f"{t.subject} should be completed, got {loaded.status}"
            assert loaded.owner == "dag_w1"

    def test_blocked_claim_returns_error(self, monkeypatch):
        """Trying to claim a task whose deps are not met returns an error
        message, and the task stays pending.

        dep-A is pre-claimed with a dummy owner so idle_poll won't
        auto-claim it after the worker finishes.
        """
        dep = task_mod.create_task("dep-A")
        child = task_mod.create_task("child-B", blockedBy=[dep.id])
        # Pre-claim dep-A so idle_poll cannot auto-claim it
        task_mod.claim_task(dep.id, owner="dummy")

        _install_fake_client(monkeypatch, [
            _resp_tool(None, [_tc("c1", "claim_task", {"task_id": child.id})]),
            _resp_stop("done"),
        ])
        teammates.spawn_teammate_thread("dag_w2", "worker", "try blocked")
        _wait_teammate("dag_w2", timeout=15)

        # Child should still be pending (claim failed)
        c = task_mod.load_task(child.id)
        assert c.status == "pending"
        assert c.owner is None

    def test_partial_deps_not_enough(self):
        """T4 blocked by [T2, T3].  Completing only T2 is NOT enough;
        T4 cannot be claimed until BOTH deps are completed."""
        t2 = task_mod.create_task("dep2")
        t3 = task_mod.create_task("dep3")
        t4 = task_mod.create_task("child4", blockedBy=[t2.id, t3.id])

        # Complete T2 manually
        task_mod.claim_task(t2.id, owner="manual")
        task_mod.complete_task(t2.id, owner="manual")

        # T4 should still NOT be claimable (T3 not done)
        assert not task_mod.can_start(t4.id)

        # Now complete T3
        task_mod.claim_task(t3.id, owner="manual")
        task_mod.complete_task(t3.id, owner="manual")

        # Now T4 IS claimable
        assert task_mod.can_start(t4.id)


# --------------------------------------------------------------------------- #
# Test 2: Concurrent multi-worker with shared task board
# --------------------------------------------------------------------------- #

class TestMultiWorkerConcurrent:
    """Multiple workers pick up tasks from a shared board."""

    def test_two_workers_pick_two_tasks(self, monkeypatch):
        """Two tasks, two workers.  Worker-1 completes task-alpha, then
        worker-2 completes task-beta.

        task-beta is created AFTER worker-1 finishes to prevent
        idle_poll from auto-claiming it.  Each worker gets its own
        ScriptedClient (re-installed before spawning worker-2).
        """
        ta = task_mod.create_task("task-alpha")

        _install_fake_client(monkeypatch, [
            _resp_tool(None, [_tc("c1", "claim_task", {"task_id": ta.id})]),
            _resp_tool(None, [_tc("c2", "complete_task", {"task_id": ta.id})]),
            _resp_stop("alpha done"),
        ])

        teammates.spawn_teammate_thread("mw_w1", "worker", "do alpha")
        _wait_teammate("mw_w1", timeout=15)

        # Now create task-beta and install worker-2's client
        tb = task_mod.create_task("task-beta")
        _install_fake_client(monkeypatch, [
            _resp_tool(None, [_tc("c3", "claim_task", {"task_id": tb.id})]),
            _resp_tool(None, [_tc("c4", "complete_task", {"task_id": tb.id})]),
            _resp_stop("beta done"),
        ])

        teammates.spawn_teammate_thread("mw_w2", "worker", "do beta")
        _wait_teammate("mw_w2", timeout=15)

        assert task_mod.load_task(ta.id).status == "completed"
        assert task_mod.load_task(tb.id).status == "completed"
        assert task_mod.load_task(ta.id).owner == "mw_w1"
        assert task_mod.load_task(tb.id).owner == "mw_w2"

    def test_double_claim_rejected(self, monkeypatch):
        """If worker A already claimed a task, worker B cannot claim it."""
        t = task_mod.create_task("exclusive-task")
        task_mod.claim_task(t.id, owner="mw_wA")

        _install_fake_client(monkeypatch, [
            _resp_tool(None, [_tc("c1", "claim_task", {"task_id": t.id})]),
            _resp_stop("failed to claim"),
        ])
        teammates.spawn_teammate_thread("mw_wB", "worker", "try claim")
        _wait_teammate("mw_wB", timeout=15)

        # Task should still be owned by wA
        loaded = task_mod.load_task(t.id)
        assert loaded.owner == "mw_wA"
        assert loaded.status == "in_progress"


# --------------------------------------------------------------------------- #
# Test 3: Plan approval flow (submit -> approve/reject -> continue)
# --------------------------------------------------------------------------- #

class TestPlanApprovalFlow:
    """Full plan-approval lifecycle: worker claims task, submits plan,
    lead approves or rejects, worker continues."""

    def test_claim_then_submit_plan_then_approve(self, monkeypatch):
        """Worker claims a task, submits a plan, lead approves, worker
        completes the task."""
        t = task_mod.create_task("plan-task-1")
        approved_evt = threading.Event()

        _install_fake_client(monkeypatch, [
            _resp_tool(None, [_tc("c1", "claim_task", {"task_id": t.id})]),
            _resp_tool(None, [_tc("c2", "submit_plan",
                                  {"plan": "Step 1: design. Step 2: code.",
                                   "task_id": t.id})]),
            # Block until lead approves, then complete
            _block_until(approved_evt,
                         _resp_tool(None, [_tc("c3", "complete_task",
                                               {"task_id": t.id})])),
            _resp_stop("plan approved and executed"),
        ])
        teammates.spawn_teammate_thread("pa_w1", "worker", "do plan task")

        # Wait for plan to arrive in lead inbox
        plan_msg = _wait_for_plan(timeout=5)
        assert plan_msg is not None, "plan_approval_request not received"
        req_id = plan_msg["metadata"]["request_id"]
        assert req_id in ctx.pending_requests
        assert ctx.pending_requests[req_id].status == "pending"

        # Drain the plan request from inbox
        _lead_inbox()

        # Lead approves
        result = run_review_plan(req_id, approve=True, feedback="looks good")
        assert "approved" in result
        assert ctx.pending_requests[req_id].status == "approved"

        # Unblock the worker
        approved_evt.set()
        _wait_teammate("pa_w1", timeout=15)

        # Task should be completed
        assert task_mod.load_task(t.id).status == "completed"
        assert task_mod.load_task(t.id).owner == "pa_w1"

        # Result message from worker
        lead_msgs = _lead_inbox()
        results = _find_msg(lead_msgs, msg_type="result", from_agent="pa_w1")
        assert len(results) == 1
        assert "plan approved and executed" in results[0]["content"]

    def test_plan_rejected_then_revise(self, monkeypatch):
        """Worker submits plan, lead rejects with feedback, worker revises
        and submits again, lead approves."""
        t = task_mod.create_task("plan-task-2")
        rejected_evt = threading.Event()
        approved_evt2 = threading.Event()

        _install_fake_client(monkeypatch, [
            _resp_tool(None, [_tc("c1", "claim_task", {"task_id": t.id})]),
            _resp_tool(None, [_tc("c2", "submit_plan",
                                  {"plan": "Vague plan.",
                                   "task_id": t.id})]),
            # Block until rejection, then resubmit
            _block_until(rejected_evt,
                         _resp_tool(None, [_tc("c3", "submit_plan",
                                               {"plan": "Revised plan.",
                                                "task_id": t.id})])),
            # Block until 2nd approval, then complete
            _block_until(approved_evt2,
                         _resp_tool(None, [_tc("c5", "complete_task",
                                               {"task_id": t.id})])),
            _resp_stop("revised and done"),
        ])
        teammates.spawn_teammate_thread("pa_w2", "worker", "do plan task")

        # First plan: reject
        plan1 = _wait_for_plan(timeout=5)
        assert plan1 is not None
        req1 = plan1["metadata"]["request_id"]
        _lead_inbox()  # drain
        run_review_plan(req1, approve=False, feedback="too vague")
        assert ctx.pending_requests[req1].status == "rejected"
        rejected_evt.set()

        # Second plan: approve
        plan2 = _wait_for_plan(timeout=5)
        assert plan2 is not None
        req2 = plan2["metadata"]["request_id"]
        assert req2 != req1  # different request IDs
        _lead_inbox()  # drain
        run_review_plan(req2, approve=True, feedback="much better")
        assert ctx.pending_requests[req2].status == "approved"
        approved_evt2.set()

        _wait_teammate("pa_w2", timeout=15)

        assert task_mod.load_task(t.id).status == "completed"

    def test_plan_for_unowned_task_rejected(self, monkeypatch):
        """Worker submits a plan for a task it does NOT own.  The plan
        submission should be rejected."""
        t = task_mod.create_task("owned-by-someone-else")
        task_mod.claim_task(t.id, owner="other-worker")

        _install_fake_client(monkeypatch, [
            _resp_tool(None, [_tc("c1", "submit_plan",
                                  {"plan": "my plan",
                                   "task_id": t.id})]),
            _resp_stop("plan rejected"),
        ])
        teammates.spawn_teammate_thread("pa_w3", "worker", "submit bad plan")
        _wait_teammate("pa_w3", timeout=15)

        # No plan_approval_request should have been created
        lead_msgs = _lead_inbox()
        plans = _find_msg(lead_msgs, msg_type="plan_approval_request")
        assert len(plans) == 0


# --------------------------------------------------------------------------- #
# Test 4: Unblocked notification after dependency completion
# --------------------------------------------------------------------------- #

class TestUnblockedNotification:
    """When a task is completed, downstream tasks that become unblocked
    should be reported in the complete_task output."""

    def test_complete_reports_unblocked(self, monkeypatch):
        """Completing T1 should report that T2 (blocked by T1) is now
        unblocked.  T2 is blocked, so idle_poll won't auto-claim it."""
        t1 = task_mod.create_task("root-unblock")
        t2 = task_mod.create_task("child-unblock", blockedBy=[t1.id])

        _install_fake_client(monkeypatch, [
            _resp_tool(None, [_tc("c1", "claim_task", {"task_id": t1.id})]),
            _resp_tool(None, [_tc("c2", "complete_task", {"task_id": t1.id})]),
            _resp_stop("done"),
        ])
        teammates.spawn_teammate_thread("ub_w1", "worker", "complete root")
        _wait_teammate("ub_w1", timeout=15)

        # T1 completed
        assert task_mod.load_task(t1.id).status == "completed"
        # T2 now unblocked
        assert task_mod.can_start(t2.id)

    def test_no_unblocked_when_deps_partial(self, monkeypatch):
        """Completing T2 (one of T4's two deps) should NOT report T4
        as unblocked because T3 is still pending.

        T3 is pre-claimed with a dummy owner so idle_poll won't
        auto-claim it.
        """
        t2 = task_mod.create_task("dep-partial-1")
        t3 = task_mod.create_task("dep-partial-2")
        t4 = task_mod.create_task("child-partial", blockedBy=[t2.id, t3.id])
        # Pre-claim T3 so idle_poll cannot auto-claim it
        task_mod.claim_task(t3.id, owner="dummy")

        _install_fake_client(monkeypatch, [
            _resp_tool(None, [_tc("c1", "claim_task", {"task_id": t2.id})]),
            _resp_tool(None, [_tc("c2", "complete_task", {"task_id": t2.id})]),
            _resp_stop("done"),
        ])
        teammates.spawn_teammate_thread("ub_w2", "worker", "complete one dep")
        _wait_teammate("ub_w2", timeout=15)

        # T4 should NOT be claimable (T3 still not completed)
        assert not task_mod.can_start(t4.id)


# --------------------------------------------------------------------------- #
# Test 5: Full lifecycle -- multi-worker + DAG + plan approval
# --------------------------------------------------------------------------- #

class TestFullLifecycle:
    """End-to-end: two workers collaboratively complete a DAG where one
    task requires plan approval.

    Topology:
        T1 (no deps, no plan)  ->  completed by worker-1
        T2 (blocked by T1, needs plan approval) -> completed by worker-2

    Worker-1 completes T1, then worker-2 claims T2 (now unblocked),
    submits a plan, lead approves, worker-2 completes T2.

    Worker-1 uses a blocking callable to stay alive until worker-2
    is ready, preventing idle_poll from auto-claiming T2.
    """

    def test_two_workers_dag_with_plan(self, monkeypatch):
        t1 = task_mod.create_task("lifecycle-T1")
        t2 = task_mod.create_task("lifecycle-T2", blockedBy=[t1.id])

        w1_done = threading.Event()
        w1_blocked = threading.Event()
        approved_evt = threading.Event()

        # Worker-1 script: claim T1, complete T1, block until done, stop
        def _w1_block():
            w1_blocked.set()
            w1_done.wait(timeout=15.0)
            return _resp_stop("T1 done")

        script_w1 = [
            _resp_tool(None, [_tc("w1c1", "claim_task", {"task_id": t1.id})]),
            _resp_tool(None, [_tc("w1c2", "complete_task", {"task_id": t1.id})]),
            _w1_block,
        ]
        # Worker-2 script: claim T2, submit plan, block for approval, complete, stop
        script_w2 = [
            _resp_tool(None, [_tc("w2c1", "claim_task", {"task_id": t2.id})]),
            _resp_tool(None, [_tc("w2c2", "submit_plan",
                                  {"plan": "Plan for T2.",
                                   "task_id": t2.id})]),
            _block_until(approved_evt,
                         _resp_tool(None, [_tc("w2c3", "complete_task",
                                               {"task_id": t2.id})])),
            _resp_stop("T2 done with approval"),
        ]

        # Install worker-1's client and spawn
        _install_fake_client(monkeypatch, script_w1)
        teammates.spawn_teammate_thread("life_w1", "worker", "do T1")
        # Wait for T1 to be completed AND worker-1 to be blocked
        assert _spin_wait(
            lambda: task_mod.load_task(t1.id).status == "completed",
            timeout=5.0), "T1 not completed in time"
        assert w1_blocked.wait(timeout=5.0), "w1 did not block in time"

        # T2 should be unblocked now
        assert task_mod.can_start(t2.id)

        # Now safe to install worker-2's client (worker-1 won't call create)
        _install_fake_client(monkeypatch, script_w2)
        teammates.spawn_teammate_thread("life_w2", "worker", "do T2")
        # Wait for plan
        plan_msg = _wait_for_plan(timeout=5)
        assert plan_msg is not None, "plan not received from worker-2"
        req_id = plan_msg["metadata"]["request_id"]
        _lead_inbox()  # drain

        # Approve
        run_review_plan(req_id, approve=True, feedback="go ahead")
        approved_evt.set()

        # Wait for worker-2 to finish
        _wait_teammate("life_w2", timeout=15)

        # Now release worker-1
        w1_done.set()
        _wait_teammate("life_w1", timeout=15)

        # Both tasks completed
        assert task_mod.load_task(t1.id).status == "completed"
        assert task_mod.load_task(t2.id).status == "completed"

        # Both workers sent results
        lead_msgs = _lead_inbox()
        r1 = _find_msg(lead_msgs, msg_type="result", from_agent="life_w1")
        r2 = _find_msg(lead_msgs, msg_type="result", from_agent="life_w2")
        assert len(r1) == 1
        assert "T1 done" in r1[0]["content"]
        assert len(r2) == 1
        assert "T2 done with approval" in r2[0]["content"]


# --------------------------------------------------------------------------- #
# Test 6: Auto-claim via idle_poll with dependencies
# --------------------------------------------------------------------------- #

class TestAutoClaimWithDeps:
    """When a worker enters idle_poll and there are unclaimed tasks with
    satisfied dependencies, idle_poll auto-claims one."""

    def test_auto_claim_after_dep_complete(self, monkeypatch):
        """T1 is completed manually.  T2 (blocked by T1) is on the board.
        A worker finishes its first turn, enters idle_poll, and auto-claims
        T2."""
        t1 = task_mod.create_task("auto-dep")
        t2 = task_mod.create_task("auto-child", blockedBy=[t1.id])
        # Complete T1 manually
        task_mod.claim_task(t1.id, owner="manual")
        task_mod.complete_task(t1.id, owner="manual")

        # T2 should be claimable now
        assert task_mod.can_start(t2.id)

        _install_fake_client(monkeypatch, [
            # Turn 1: worker finishes (enters idle_poll)
            _resp_stop("initial"),
            # Turn 2: after auto-claim, worker completes T2
            _resp_tool(None, [_tc("ac1", "complete_task", {"task_id": t2.id})]),
            # Turn 3: stop
            _resp_stop("auto-claimed and done"),
        ])
        teammates.spawn_teammate_thread("ac_w1", "worker", "start")
        _wait_teammate("ac_w1", timeout=15)

        # T2 should be completed by the auto-claiming worker
        loaded = task_mod.load_task(t2.id)
        assert loaded.status == "completed"
        assert loaded.owner == "ac_w1"

    def test_auto_claim_skips_blocked_task(self, monkeypatch):
        """idle_poll should NOT auto-claim a task whose deps are not met.

        T1 is pre-claimed with a dummy owner (so it's not auto-claimable
        either).  T2 is blocked by T1 and cannot be auto-claimed.
        The worker enters idle_poll, finds nothing to claim, and times
        out.
        """
        t1 = task_mod.create_task("skip-dep")
        t2 = task_mod.create_task("skip-child", blockedBy=[t1.id])
        # Pre-claim T1 so idle_poll cannot auto-claim it
        task_mod.claim_task(t1.id, owner="dummy")

        # T2 should NOT be claimable
        assert not task_mod.can_start(t2.id)

        _install_fake_client(monkeypatch, [
            _resp_stop("nothing to do"),
        ])
        teammates.spawn_teammate_thread("ac_w2", "worker", "start")
        _wait_teammate("ac_w2", timeout=15)

        # T2 should still be pending (never claimed)
        loaded = task_mod.load_task(t2.id)
        assert loaded.status == "pending"
        assert loaded.owner is None


# --------------------------------------------------------------------------- #
# Test 7: Three-worker pipeline with fan-out
# --------------------------------------------------------------------------- #

class TestThreeWorkerPipeline:
    """Three workers complete a fan-out DAG:

        T1 -> T2, T3  (T1 done by worker-1, T2 by worker-2, T3 by worker-3)

    Each worker gets its own ScriptedClient so they don't interfere.
    Worker-1 uses a blocking callable to stay alive until workers 2 and 3
    are ready, preventing idle_poll from auto-claiming T2/T3.
    """

    def test_fan_out_three_workers(self, monkeypatch):
        t1 = task_mod.create_task("pipe-T1")
        t2 = task_mod.create_task("pipe-T2", blockedBy=[t1.id])
        t3 = task_mod.create_task("pipe-T3", blockedBy=[t1.id])

        w1_release = threading.Event()
        w2_release = threading.Event()
        w1_blocked = threading.Event()
        w2_blocked = threading.Event()

        # Worker-1 script: claim T1, complete T1, block, stop
        def _w1_block():
            w1_blocked.set()
            w1_release.wait(timeout=10.0)
            return _resp_stop("T1 done")

        script_w1 = [
            _resp_tool(None, [_tc("p1c1", "claim_task", {"task_id": t1.id})]),
            _resp_tool(None, [_tc("p1c2", "complete_task", {"task_id": t1.id})]),
            _w1_block,
        ]
        # Worker-2 script: claim T2, complete T2, block, stop
        def _w2_block():
            w2_blocked.set()
            w2_release.wait(timeout=10.0)
            return _resp_stop("T2 done")

        script_w2 = [
            _resp_tool(None, [_tc("p2c1", "claim_task", {"task_id": t2.id})]),
            _resp_tool(None, [_tc("p2c2", "complete_task", {"task_id": t2.id})]),
            _w2_block,
        ]
        # Worker-3 script: claim T3, complete T3, stop
        script_w3 = [
            _resp_tool(None, [_tc("p3c1", "claim_task", {"task_id": t3.id})]),
            _resp_tool(None, [_tc("p3c2", "complete_task", {"task_id": t3.id})]),
            _resp_stop("T3 done"),
        ]

        # Install worker-1's client first
        _install_fake_client(monkeypatch, script_w1)

        # Worker 1 (blocks after completing T1)
        teammates.spawn_teammate_thread("pipe_w1", "worker", "do T1")
        # Wait until worker-1 has completed T1 AND is blocked in _w1_block.
        # The blocked event guarantees worker-1 has consumed all 3 script
        # items and will not call create() again until released.
        assert _spin_wait(
            lambda: task_mod.load_task(t1.id).status == "completed",
            timeout=5.0), "T1 not completed in time"
        assert w1_blocked.wait(timeout=5.0), "w1 did not block in time"

        # Now safe to install worker-2's client (worker-1 won't call create)
        _install_fake_client(monkeypatch, script_w2)
        teammates.spawn_teammate_thread("pipe_w2", "worker", "do T2")
        assert _spin_wait(
            lambda: task_mod.load_task(t2.id).status == "completed",
            timeout=5.0), "T2 not completed in time"
        assert w2_blocked.wait(timeout=5.0), "w2 did not block in time"

        # Install worker-3's client (worker-2 won't call create)
        _install_fake_client(monkeypatch, script_w3)
        teammates.spawn_teammate_thread("pipe_w3", "worker", "do T3")
        _wait_teammate("pipe_w3", timeout=15)

        # Release worker-2 (T3 is done, no unclaimed tasks)
        w2_release.set()
        _wait_teammate("pipe_w2", timeout=15)

        # Release worker-1 (T2 and T3 done, no unclaimed tasks)
        w1_release.set()
        _wait_teammate("pipe_w1", timeout=15)

        assert task_mod.load_task(t1.id).status == "completed"
        assert task_mod.load_task(t2.id).status == "completed"
        assert task_mod.load_task(t3.id).status == "completed"
        assert task_mod.load_task(t1.id).owner == "pipe_w1"
        assert task_mod.load_task(t2.id).owner == "pipe_w2"
        assert task_mod.load_task(t3.id).owner == "pipe_w3"

        # All three sent results
        lead_msgs = _lead_inbox()
        for name, label in [("pipe_w1", "T1"), ("pipe_w2", "T2"),
                            ("pipe_w3", "T3")]:
            results = _find_msg(lead_msgs, msg_type="result", from_agent=name)
            assert len(results) == 1
            assert f"{label} done" in results[0]["content"]
