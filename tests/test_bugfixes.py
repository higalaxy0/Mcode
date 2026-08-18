"""Regression tests for the bug fixes in tasks.py / bus.py / teammates.py.

Each test targets a specific bug identified during the 10-task DAG run:

  A   - claim_task race condition (concurrent claim must not double-assign)
  B   - idle_poll thundering herd (fall-through + jitter)
  C   - submit_plan no ownership validation
  D   - result extraction (tool message not sent as final result)
  F1  - complete_task no ownership check
  F2  - request_plan / submit_plan protocol mismatch
  F3  - idle_poll doesn't route protocol messages
  F4  - shutdown batch drops messages
  F5  - scan_unclaimed_tasks no exception handling
  F6  - non-message inbox types dropped
  F7  - task ID collision (uuid)
  F8  - can_start double-loads task from disk
  F9  - pending_requests check-then-set not atomic
  F10 - idle_poll redundant params
  #2  - idle_poll cannot resume owned in_progress tasks (task abandonment)
  #3  - worker exit does not release owned tasks (orphaned tasks)
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from mcodecore import tasks, bus
from mcodecore.context import ctx


# =========================================================================== #
# Bug A: claim_task race condition
# =========================================================================== #

class TestClaimRaceCondition:
    """Concurrent claim_task calls must not both succeed on the same task."""

    def test_concurrent_claim_single_winner(self):
        t = tasks.create_task("race-task")
        barrier = threading.Barrier(5)
        results = []

        def claim(owner):
            barrier.wait()
            r = tasks.claim_task(t.id, owner=owner)
            results.append(r)

        threads = [threading.Thread(target=claim, args=(f"agent-{i}",))
                   for i in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        claimed = [r for r in results if "Claimed" in r]
        failed = [r for r in results if "Claimed" not in r]
        assert len(claimed) == 1, (
            f"Expected exactly 1 winner, got {len(claimed)}: {results}")
        assert len(failed) == 4

        loaded = tasks.load_task(t.id)
        assert loaded.status == "in_progress"
        # Owner must be one of the 5 agents
        assert loaded.owner.startswith("agent-")

    def test_concurrent_claim_different_tasks(self):
        """Each of N agents should be able to claim a distinct task."""
        task_ids = [tasks.create_task(f"t-{i}").id for i in range(5)]
        barrier = threading.Barrier(5)
        claimed_ids = []
        lock = threading.Lock()

        def claim(idx):
            barrier.wait()
            r = tasks.claim_task(task_ids[idx], owner=f"agent-{idx}")
            if "Claimed" in r:
                with lock:
                    claimed_ids.append(task_ids[idx])

        threads = [threading.Thread(target=claim, args=(i,))
                   for i in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(claimed_ids) == 5


# =========================================================================== #
# Bug B: idle_poll thundering herd + fall-through
# =========================================================================== #

class TestIdlePollFallThrough:
    """idle_poll should try the next unclaimed task when the first fails."""

    def test_idle_poll_falls_through_to_second_task(self, monkeypatch):
        monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 0)
        monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)

        t1 = tasks.create_task("first")
        t2 = tasks.create_task("second")
        # Pre-claim t1 so idle_poll's first claim attempt fails and it
        # should fall through to t2.
        tasks.claim_task(t1.id, owner="other-agent")

        messages = []
        result = bus.idle_poll("tester", messages, "tester")
        assert result == "work"
        # t2 should have been claimed by "tester"
        loaded = tasks.load_task(t2.id)
        assert loaded.owner == "tester"

    def test_idle_poll_timeout_when_all_claimed(self, monkeypatch):
        monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 0)
        monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)
        t = tasks.create_task("only")
        tasks.claim_task(t.id, owner="someone-else")

        messages = []
        result = bus.idle_poll("starved", messages, "starved")
        assert result == "timeout"


# =========================================================================== #
# Bug C: submit_plan no ownership validation
# =========================================================================== #

class TestSubmitPlanOwnership:
    """submit_plan must reject plans for tasks the submitter doesn't own."""

    def test_submit_plan_rejected_for_unowned_task(self):
        t = tasks.create_task("planme")
        tasks.claim_task(t.id, owner="alpha")
        # beta tries to submit a plan for alpha's task
        res = bus._teammate_submit_plan("beta", "my plan", task_id=t.id)
        assert "Cannot submit plan" in res
        assert "owned by alpha" in res

    def test_submit_plan_accepted_for_owned_task(self):
        t = tasks.create_task("owned")
        tasks.claim_task(t.id, owner="gamma")
        res = bus._teammate_submit_plan("gamma", "my plan", task_id=t.id)
        assert "submitted" in res.lower()

    def test_submit_plan_rejected_for_nonexistent_task(self):
        res = bus._teammate_submit_plan("delta", "plan", task_id="task_nope")
        assert "not found" in res.lower()

    def test_submit_plan_rejected_for_pending_task(self):
        t = tasks.create_task("not-started")
        # Task is still pending (not claimed by anyone)
        res = bus._teammate_submit_plan("epsilon", "plan", task_id=t.id)
        assert "Cannot submit plan" in res
        assert "not in_progress" in res

    def test_submit_plan_without_task_id_still_works(self):
        """Backward compat: no task_id means no ownership check."""
        res = bus._teammate_submit_plan("zeta", "plan")
        assert "submitted" in res.lower()


# =========================================================================== #
# Bug D: result extraction
# =========================================================================== #

class TestResultExtraction:
    """The final result must be an assistant message, not tool output.

    We can't easily test the teammate thread directly, but we can test
    the logic pattern: searching backwards for the last assistant content.
    """

    def test_last_assistant_message_found(self):
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "I'll do it", "tool_calls": []},
            {"role": "tool", "tool_call_id": "x", "content": "tool output here"},
            {"role": "assistant", "content": "Done! Summary of work.", "tool_calls": []},
            {"role": "tool", "tool_call_id": "y", "content": "more tool output"},
        ]
        result = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                result = msg["content"]
                break
        assert result == "Done! Summary of work."

    def test_no_assistant_message_gives_fallback(self):
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "tool", "tool_call_id": "x", "content": "tool output"},
        ]
        result = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                result = msg["content"]
                break
        assert result == ""  # caller will use fallback string


# =========================================================================== #
# Bug F-1: complete_task no ownership check
# =========================================================================== #

class TestCompleteTaskOwnership:
    """complete_task must reject completion by non-owners."""

    def test_complete_by_non_owner_rejected(self):
        t = tasks.create_task("complete-me")
        tasks.claim_task(t.id, owner="alpha")
        res = tasks.complete_task(t.id, owner="beta")
        assert "cannot complete" in res.lower()
        assert "owned by alpha" in res
        # Task should still be in_progress
        loaded = tasks.load_task(t.id)
        assert loaded.status == "in_progress"

    def test_complete_by_owner_succeeds(self):
        t = tasks.create_task("my-task")
        tasks.claim_task(t.id, owner="gamma")
        res = tasks.complete_task(t.id, owner="gamma")
        assert "Completed" in res
        assert tasks.load_task(t.id).status == "completed"

    def test_complete_without_owner_still_works(self):
        """Backward compat: no owner param means no check (lead-side)."""
        t = tasks.create_task("lead-task")
        tasks.claim_task(t.id, owner="agent")
        res = tasks.complete_task(t.id)
        assert "Completed" in res


# =========================================================================== #
# Bug F-2: request_plan / submit_plan protocol mismatch
# =========================================================================== #

class TestProtocolLinking:
    """run_request_plan should create a ProtocolState (tracked)."""

    def test_request_plan_creates_protocol_state(self):
        res = bus.run_request_plan("teammate-x", "task-xyz")
        assert "req:" in res
        states = [v for v in ctx.pending_requests.values()
                  if v.type == "plan_approval" and v.target == "teammate-x"]
        assert len(states) == 1
        assert states[0].task_id == "task-xyz"

    def test_submit_plan_state_has_task_id(self):
        t = tasks.create_task("linked")
        tasks.claim_task(t.id, owner="worker")
        bus._teammate_submit_plan("worker", "my plan", task_id=t.id)
        states = [v for v in ctx.pending_requests.values()
                  if v.sender == "worker"]
        assert len(states) == 1
        assert states[0].task_id == t.id


# =========================================================================== #
# Bug F-3 + F-4: idle_poll protocol routing + batch shutdown
# =========================================================================== #

class TestIdlePollProtocolRouting:
    """idle_poll should route plan_approval_response and handle batch shutdown."""

    def test_idle_poll_routes_plan_approval(self, monkeypatch):
        monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 0)
        monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)
        # Send a plan_approval_response to the agent's inbox
        ctx.bus.send("lead", "planned-agent", "Approved!",
                     "plan_approval_response",
                     {"request_id": "req_test", "approve": True})
        messages = []
        result = bus.idle_poll("planned-agent", messages, "planned-agent")
        assert result == "work"
        # The message should have been structured as [Plan approved]
        assert any("[Plan approved]" in m.get("content", "")
                   for m in messages)

    def test_idle_poll_batch_shutdown_preserves_messages(self, monkeypatch):
        """When shutdown arrives, other messages in the same batch
        should still be surfaced."""
        monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 0)
        monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)
        # Send a regular message AND a shutdown in the same batch
        ctx.bus.send("lead", "batch-agent", "important info", "message")
        ctx.bus.send("lead", "batch-agent", "shutdown please",
                     "shutdown_request", {"request_id": "req_sd"})
        messages = []
        result = bus.idle_poll("batch-agent", messages, "batch-agent")
        assert result == "shutdown"
        # The regular message should still be in messages
        assert any("important info" in m.get("content", "")
                   for m in messages)


# =========================================================================== #
# Bug F-5: scan_unclaimed_tasks no exception handling
# =========================================================================== #

class TestScanCorruptFile:
    """scan_unclaimed_tasks should skip corrupt files, not crash."""

    def test_scan_skips_corrupt_task_file(self):
        from mcodecore.tasks import TASKS_DIR
        # Create a valid task
        good = tasks.create_task("good-task")
        # Create a corrupt task file
        corrupt_path = TASKS_DIR / "task_corrupt.json"
        corrupt_path.write_text("{this is not valid json}", encoding="utf-8")
        # scan should not crash
        unclaimed = tasks.scan_unclaimed_tasks()
        ids = [t["id"] for t in unclaimed]
        assert good.id in ids
        # corrupt file should not appear
        assert "task_corrupt" not in ids


# =========================================================================== #
# Bug F-6: non-message inbox types dropped
# =========================================================================== #

class TestNonMessageInboxTypes:
    """Teammates should surface non-protocol inbox messages of any type."""

    def test_non_message_types_collected(self):
        """Simulate the filter logic used in teammates.py main loop."""
        inbox = [
            {"type": "message", "content": "hello"},
            {"type": "result", "content": "some result"},
            {"type": "LLM API error", "content": "error info"},
            {"type": "shutdown_request", "content": "shutdown"},
            {"type": "plan_approval_response", "content": "approved"},
        ]
        _protocol_types = {"shutdown_request", "plan_approval_response"}
        non_protocol = [m for m in inbox
                        if m.get("type") not in _protocol_types]
        # Should include message, result, AND LLM API error
        types = [m["type"] for m in non_protocol]
        assert "message" in types
        assert "result" in types
        assert "LLM API error" in types
        assert "shutdown_request" not in types
        assert "plan_approval_response" not in types


# =========================================================================== #
# Bug F-7: task ID collision
# =========================================================================== #

class TestTaskIdCollision:
    """Task IDs should be unique even when created in the same second."""

    def test_ids_unique_for_burst_creation(self):
        ids = [tasks.create_task(f"burst-{i}").id for i in range(100)]
        assert len(ids) == len(set(ids)), "Duplicate task IDs detected"

    def test_id_uses_uuid_format(self):
        t = tasks.create_task("format-check")
        # Should be task_ followed by 12 hex chars
        assert t.id.startswith("task_")
        suffix = t.id[5:]
        assert len(suffix) == 12
        int(suffix, 16)  # must be valid hex


# =========================================================================== #
# Bug F-8: can_start double-loads task from disk
# =========================================================================== #

class TestCanStartNoDoubleLoad:
    """_deps_ready should accept a Task object (no extra disk load)."""

    def test_deps_ready_with_task_object(self):
        dep = tasks.create_task("dep")
        dep.status = "in_progress"
        tasks.save_task(dep)
        tasks.complete_task(dep.id)
        child = tasks.create_task("child", blockedBy=[dep.id])
        # _deps_ready should work with the task object directly
        assert tasks._deps_ready(child) is True

    def test_deps_ready_false_for_uncompleted(self):
        dep = tasks.create_task("uncompleted-dep")
        child = tasks.create_task("child2", blockedBy=[dep.id])
        assert tasks._deps_ready(child) is False


# =========================================================================== #
# Bug F-9: pending_requests check-then-set not atomic
# =========================================================================== #

class TestPendingRequestsAtomicity:
    """run_review_plan check-then-set should be safe under concurrency."""

    def test_concurrent_review_does_not_double_process(self):
        """Two concurrent review_plan calls on the same request_id
        should result in only one succeeding."""
        bus._teammate_submit_plan("worker", "plan")
        rid = [k for k, v in ctx.pending_requests.items()
               if v.sender == "worker"][0]
        barrier = threading.Barrier(2)
        results = []

        def review(approve):
            barrier.wait()
            r = bus.run_review_plan(rid, approve=approve)
            results.append(r)

        t1 = threading.Thread(target=review, args=(True,))
        t2 = threading.Thread(target=review, args=(False,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # One should process the request, the other should see "already"
        processed = [r for r in results
                     if r.startswith("Plan ") and "already" not in r]
        already = [r for r in results if "already" in r]
        assert len(processed) == 1, f"Expected 1 processed, got {processed}"
        assert len(already) == 1, f"Expected 1 'already', got {already}"


# =========================================================================== #
# Bug F-10: idle_poll redundant params
# =========================================================================== #

class TestIdlePollSignature:
    """idle_poll should work with 3 required args (role optional)."""

    def test_idle_poll_works_without_role(self, monkeypatch):
        monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 0)
        monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)
        # Should work with just 3 args (role defaults to "")
        result = bus.idle_poll("solo2", [], "solo2")
        assert result == "timeout"


# =========================================================================== #
# Fix #2: idle_poll resumes owned in_progress tasks
# =========================================================================== #

class TestIdlePollResumeOwned:
    """idle_poll should resume the agent's own in_progress tasks before
    scanning for new unclaimed ones.

    This prevents task abandonment when a worker exhausts its turn budget
    with tasks still in_progress (e.g. it claimed two tasks but only
    completed one before hitting the 50-turn cap).
    """

    def test_idle_poll_resumes_owned_inprogress(self, monkeypatch):
        """When the agent owns an in_progress task and the inbox is
        empty, idle_poll should return 'work' and inject a <resume>
        message so the LLM picks up the task."""
        monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 0)
        monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)

        t = tasks.create_task("resume-me")
        tasks.claim_task(t.id, owner="stuck-worker")

        messages: list = []
        result = bus.idle_poll("stuck-worker", messages, "stuck-worker")
        assert result == "work"
        # A <resume> message must have been injected
        assert any("<resume>" in m.get("content", "") for m in messages)

    def test_idle_poll_resumes_multiple_owned_tasks(self, monkeypatch):
        """Multiple owned in_progress tasks should all be listed in the
        <resume> message."""
        monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 0)
        monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)

        t1 = tasks.create_task("task-A")
        t2 = tasks.create_task("task-B")
        tasks.claim_task(t1.id, owner="greedy-worker")
        tasks.claim_task(t2.id, owner="greedy-worker")

        messages: list = []
        result = bus.idle_poll("greedy-worker", messages, "greedy-worker")
        assert result == "work"
        resume_msg = [m for m in messages if "<resume>" in m.get("content", "")]
        assert len(resume_msg) == 1
        # Both task subjects should be mentioned
        assert "task-A" in resume_msg[0]["content"]
        assert "task-B" in resume_msg[0]["content"]

    def test_idle_poll_does_not_resume_other_agents_tasks(self, monkeypatch):
        """Owned tasks of a *different* agent must not trigger resume."""
        monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 0)
        monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)

        t = tasks.create_task("someone-elses")
        tasks.claim_task(t.id, owner="alpha")

        messages: list = []
        # beta should NOT pick up alpha's in_progress task via resume
        result = bus.idle_poll("beta", messages, "beta")
        assert result == "timeout"
        assert not any("<resume>" in m.get("content", "") for m in messages)

    def test_idle_poll_resume_takes_priority_over_unclaimed(self, monkeypatch):
        """If the agent owns an in_progress task AND there are unclaimed
        tasks, resume should win (own tasks first)."""
        monkeypatch.setattr(bus, "IDLE_POLL_INTERVAL", 0)
        monkeypatch.setattr(bus, "IDLE_TIMEOUT", 1)

        owned = tasks.create_task("owned-task")
        tasks.claim_task(owned.id, owner="multi-worker")
        # Also create an unclaimed task
        unclaimed = tasks.create_task("unclaimed-task")

        messages: list = []
        result = bus.idle_poll("multi-worker", messages, "multi-worker")
        assert result == "work"
        # The resume message should reference the owned task, not auto-claim
        assert "<resume>" in messages[-1]["content"]
        assert "owned-task" in messages[-1]["content"]
        # The unclaimed task should still be pending/unclaimed
        assert tasks.load_task(unclaimed.id).status == "pending"
        assert tasks.load_task(unclaimed.id).owner is None


# =========================================================================== #
# Fix #2: list_owned_inprogress + release_task
# =========================================================================== #

class TestListOwnedInprogress:
    """list_owned_inprogress should return only the agent's in_progress tasks."""

    def test_returns_owned_inprogress_tasks(self):
        t1 = tasks.create_task("owned-1")
        t2 = tasks.create_task("owned-2")
        t3 = tasks.create_task("pending-other")
        tasks.claim_task(t1.id, owner="alpha")
        tasks.claim_task(t2.id, owner="alpha")
        # t3 is pending, not claimed

        owned = tasks.list_owned_inprogress("alpha")
        ids = [t["id"] for t in owned]
        assert t1.id in ids
        assert t2.id in ids
        assert t3.id not in ids

    def test_excludes_other_owners(self):
        t1 = tasks.create_task("mine")
        t2 = tasks.create_task("theirs")
        tasks.claim_task(t1.id, owner="alpha")
        tasks.claim_task(t2.id, owner="beta")

        owned = tasks.list_owned_inprogress("alpha")
        ids = [t["id"] for t in owned]
        assert t1.id in ids
        assert t2.id not in ids

    def test_excludes_completed(self):
        t = tasks.create_task("done")
        tasks.claim_task(t.id, owner="alpha")
        tasks.complete_task(t.id, owner="alpha")

        owned = tasks.list_owned_inprogress("alpha")
        assert t.id not in [x["id"] for x in owned]

    def test_empty_when_no_tasks(self):
        owned = tasks.list_owned_inprogress("nobody")
        assert owned == []


class TestReleaseTask:
    """release_task should reset an in_progress task to pending."""

    def test_release_by_owner_succeeds(self):
        t = tasks.create_task("release-me")
        tasks.claim_task(t.id, owner="alpha")
        res = tasks.release_task(t.id, owner="alpha")
        assert "Released" in res
        loaded = tasks.load_task(t.id)
        assert loaded.status == "pending"
        assert loaded.owner is None

    def test_release_by_non_owner_rejected(self):
        t = tasks.create_task("not-yours")
        tasks.claim_task(t.id, owner="alpha")
        res = tasks.release_task(t.id, owner="beta")
        assert "cannot release" in res.lower()
        assert "owned by alpha" in res
        # Task should still be in_progress with alpha as owner
        loaded = tasks.load_task(t.id)
        assert loaded.status == "in_progress"
        assert loaded.owner == "alpha"

    def test_release_non_inprogress_rejected(self):
        t = tasks.create_task("still-pending")
        res = tasks.release_task(t.id, owner="nobody")
        assert "cannot release" in res.lower()

    def test_release_completed_rejected(self):
        t = tasks.create_task("already-done")
        tasks.claim_task(t.id, owner="alpha")
        tasks.complete_task(t.id, owner="alpha")
        res = tasks.release_task(t.id, owner="alpha")
        assert "cannot release" in res.lower()

    def test_release_without_owner_check(self):
        """Backward compat: no owner param means no ownership check."""
        t = tasks.create_task("no-check")
        tasks.claim_task(t.id, owner="alpha")
        res = tasks.release_task(t.id)
        assert "Released" in res
        assert tasks.load_task(t.id).status == "pending"

    def test_release_makes_task_claimable_again(self):
        """After release, another agent should be able to claim the task."""
        t = tasks.create_task("recycled")
        tasks.claim_task(t.id, owner="alpha")
        tasks.release_task(t.id, owner="alpha")

        # beta should now be able to claim it
        res = tasks.claim_task(t.id, owner="beta")
        assert "Claimed" in res
        loaded = tasks.load_task(t.id)
        assert loaded.owner == "beta"
        assert loaded.status == "in_progress"


# --------------------------------------------------------------------------- #
# G - Tool-call arguments sanitization (400 BadRequest fix)
#
# Assistant messages with tool_calls whose function.arguments is not valid
# JSON cause a 400 BadRequestError from the LLM API.  This happens when a
# stream is interrupted mid-arguments-accumulation, leaving truncated JSON.
# sanitize_message must repair such arguments to "{}" as a safety net.
# --------------------------------------------------------------------------- #

class TestToolCallArgsSanitization:
    """Regression tests for the tool_call arguments validation fix."""

    def test_truncated_json_arguments_repaired(self):
        """Truncated JSON arguments must be repaired to '{}'."""
        from mcodecore.utils import sanitize_message
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "dir'},
            }],
        }
        result = sanitize_message(msg)
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_number_arguments_repaired(self):
        """A bare number string is not a JSON object and must be repaired."""
        from mcodecore.utils import sanitize_message
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_2",
                "type": "function",
                "function": {"name": "bash", "arguments": "123"},
            }],
        }
        result = sanitize_message(msg)
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_empty_arguments_repaired(self):
        """Empty string arguments must be repaired to '{}'."""
        from mcodecore.utils import sanitize_message
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_3",
                "type": "function",
                "function": {"name": "bash", "arguments": ""},
            }],
        }
        result = sanitize_message(msg)
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_none_arguments_repaired(self):
        """None arguments must be repaired to '{}'."""
        from mcodecore.utils import _validate_tool_args
        assert _validate_tool_args(None) == "{}"

    def test_plain_text_arguments_repaired(self):
        """Plain text (not JSON) must be repaired to '{}'."""
        from mcodecore.utils import sanitize_message
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_4",
                "type": "function",
                "function": {"name": "bash", "arguments": "just some text"},
            }],
        }
        result = sanitize_message(msg)
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_array_arguments_repaired(self):
        """A JSON array is not an object and must be repaired to '{}'."""
        from mcodecore.utils import sanitize_message
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_5",
                "type": "function",
                "function": {"name": "bash", "arguments": '["a", "b"]'},
            }],
        }
        result = sanitize_message(msg)
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_null_arguments_repaired(self):
        """JSON null is not an object and must be repaired to '{}'."""
        from mcodecore.utils import sanitize_message
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_6",
                "type": "function",
                "function": {"name": "bash", "arguments": "null"},
            }],
        }
        result = sanitize_message(msg)
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_valid_arguments_unchanged(self):
        """Valid JSON object arguments must pass through unchanged."""
        from mcodecore.utils import sanitize_message
        valid_args = '{"command": "dir /b"}'
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_7",
                "type": "function",
                "function": {"name": "bash", "arguments": valid_args},
            }],
        }
        result = sanitize_message(msg)
        assert result["tool_calls"][0]["function"]["arguments"] == valid_args

    def test_valid_empty_object_unchanged(self):
        """An empty JSON object '{}' is valid and must pass through."""
        from mcodecore.utils import sanitize_message
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_8",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }],
        }
        result = sanitize_message(msg)
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_multiple_tool_calls_each_repaired(self):
        """Multiple tool_calls in one message must each be repaired independently."""
        from mcodecore.utils import sanitize_message
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "bash", "arguments": '{"a":'}},
                {"id": "c2", "type": "function",
                 "function": {"name": "bash", "arguments": '{"b": 2}'}},
                {"id": "c3", "type": "function",
                 "function": {"name": "bash", "arguments": "bad"}},
            ],
        }
        result = sanitize_message(msg)
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"
        assert result["tool_calls"][1]["function"]["arguments"] == '{"b": 2}'
        assert result["tool_calls"][2]["function"]["arguments"] == "{}"

    def test_no_tool_calls_unchanged(self):
        """Messages without tool_calls must not be affected."""
        from mcodecore.utils import sanitize_message
        msg = {"role": "user", "content": "hello"}
        result = sanitize_message(msg)
        assert result == {"role": "user", "content": "hello"}

    def test_missing_content_with_tool_calls(self):
        """Missing content with tool_calls must get content='' and valid args."""
        from mcodecore.utils import sanitize_message
        msg = {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_9",
                "type": "function",
                "function": {"name": "bash", "arguments": "broken"},
            }],
        }
        result = sanitize_message(msg)
        assert result["content"] == ""
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_sanitize_messages_list(self):
        """sanitize_messages must repair all tool_calls across the list."""
        from mcodecore.utils import sanitize_messages
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c", "type": "function",
                "function": {"name": "bash", "arguments": '{"x":'}
            }]},
            {"role": "tool", "tool_call_id": "c", "content": "output"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "d", "type": "function",
                "function": {"name": "bash", "arguments": "123"}
            }]},
        ]
        result = sanitize_messages(msgs)
        assert result[1]["tool_calls"][0]["function"]["arguments"] == "{}"
        assert result[3]["tool_calls"][0]["function"]["arguments"] == "{}"
        # Other messages unaffected
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[2] == {"role": "tool", "tool_call_id": "c", "content": "output"}

    def test_all_arguments_are_valid_json_after_sanitize(self):
        """Every tool_call arguments field must be valid JSON after sanitization."""
        from mcodecore.utils import sanitize_messages
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "bash", "arguments": arg}}
                for i, arg in enumerate([
                    "", "123", "null", '{"a":', "text", '["x"]', "{}", '{"a":1}'
                ])
            ]},
        ]
        result = sanitize_messages(msgs)
        for tc in result[0]["tool_calls"]:
            parsed = json.loads(tc["function"]["arguments"])
            assert isinstance(parsed, dict), \
                f"Arguments {tc['function']['arguments']!r} is not a JSON object"


class TestStreamInterruptedDetection:
    """Tests for the interrupted-stream detection in stream_response.

    When a stream ends without finish_reason == "tool_calls" but has partial
    tool_call fragments, the partial tool_calls must be dropped to avoid
    sending truncated JSON arguments to the API on the next turn.

    This covers two cases:
    1. Connection cut off (finish_reason is None).
    2. max_tokens hit mid-arguments (finish_reason == "length").
    """

    def test_interrupted_flag_set_when_no_finish_reason_with_tool_calls(self):
        """The interrupted flag logic: no finish_reason + tool_calls parts = interrupted."""
        # Simulate the condition checked in stream_response
        finish_reason = None
        tool_calls_parts = {0: {"id": "call_x", "name": "bash", "arguments": '{"cmd":'}}
        # This mirrors: if finish_reason != "tool_calls" and tool_calls_parts:
        interrupted = (finish_reason != "tool_calls" and bool(tool_calls_parts))
        assert interrupted is True

    def test_interrupted_when_max_tokens_truncates_tool_call(self):
        """max_tokens (finish_reason='length') + partial tool_calls = interrupted."""
        finish_reason = "length"
        tool_calls_parts = {0: {"id": "call_x", "name": "bash", "arguments": '{"cmd":'}}
        interrupted = (finish_reason != "tool_calls" and bool(tool_calls_parts))
        assert interrupted is True

    def test_not_interrupted_when_finish_reason_is_tool_calls(self):
        """finish_reason == 'tool_calls' means the tool calls completed normally."""
        finish_reason = "tool_calls"
        tool_calls_parts = {0: {"id": "call_x", "name": "bash", "arguments": '{"cmd":"dir"}'}}
        interrupted = (finish_reason != "tool_calls" and bool(tool_calls_parts))
        assert interrupted is False

    def test_not_interrupted_when_no_tool_calls(self):
        """No tool_calls means nothing to be interrupted about."""
        finish_reason = None
        tool_calls_parts = {}
        interrupted = (finish_reason != "tool_calls" and bool(tool_calls_parts))
        assert interrupted is False

    def test_interrupted_when_max_tokens_with_no_tool_calls(self):
        """max_tokens on plain content (no tool_calls) is NOT interrupted — just truncated text."""
        finish_reason = "length"
        tool_calls_parts = {}
        interrupted = (finish_reason != "tool_calls" and bool(tool_calls_parts))
        assert interrupted is False

    def test_interrupted_drops_tool_calls(self):
        """When interrupted, tool_calls must not be included in msg_dict."""
        # Simulate the msg_dict construction logic
        interrupted = True
        tool_calls_parts = {0: {"id": "call_x", "name": "bash", "arguments": '{"cmd":'}}
        msg_dict = {"role": "assistant"}
        if tool_calls_parts and not interrupted:
            msg_dict["tool_calls"] = "would_be_here"
        assert "tool_calls" not in msg_dict

    def test_interrupted_adds_note(self):
        """When interrupted, a note about the interruption should be added."""
        interrupted = True
        content = ""
        msg_dict = {"role": "assistant"}
        if content:
            msg_dict["content"] = content
        if interrupted:
            msg_dict["content"] = (msg_dict.get("content") or "") + \
                "\n[stream was interrupted; partial tool call discarded]"
        assert "interrupted" in msg_dict["content"]
