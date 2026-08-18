"""Real API end-to-end test for the teamagent pipeline.

Tests (all against the real LLM):
  1. spawn_teammate -> auto result delivery (via shutdown protocol)
  2. teammate_status liveness probe while teammate is running
  3. submit_plan -> review_plan metadata preservation (request_id round-trip)
  4. mid-work send_message from teammate while still running
  5. orphan task release on teammate exit

Design note: teammates only deliver results after their main loop exits
(shutdown or 360s idle_poll timeout).  For practical test durations,
each test sends a shutdown_request after detecting the expected side
effect (file created, message received, plan submitted).  This also
exercises the shutdown protocol path.
"""
import sys
import os
import time
import json
import shutil
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcodecore.config import MAILBOX_DIR, WORKDIR, TASKS_DIR
from mcodecore.context import ctx

TEAM_HISTORY_DIR = WORKDIR / ".team_history"


def _reset_env():
    """Clean all persistent state dirs for test isolation."""
    for d in [MAILBOX_DIR, TEAM_HISTORY_DIR, TASKS_DIR]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
    ctx.active_teammates.clear()
    ctx.teammate_registry.clear()
    ctx.pending_requests.clear()


_reset_env()

from mcodecore.bus import (
    consume_lead_inbox, format_inbox_msg,
    run_review_plan, run_request_shutdown,
)
from mcodecore.teammates import spawn_teammate_thread
from mcodecore.tasks import create_task, load_task, list_tasks

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36mTEST\033[0m"

results = []


def _check(name, cond, detail=""):
    status = PASS if cond else FAIL
    results.append((name, cond))
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))


def _lead_mailbox_nonempty():
    inbox = MAILBOX_DIR / "lead.jsonl"
    return inbox.exists() and inbox.stat().st_size > 0


def _wait_teammate_done(name, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        evt = ctx.active_teammates.get(name)
        if evt is None or evt.is_set():
            return True
        time.sleep(0.5)
    return False


def _shutdown_and_wait(name, timeout=60):
    """Send shutdown_request and wait for teammate to finish."""
    run_request_shutdown(name)
    return _wait_teammate_done(name, timeout)


def _wait_file(path, timeout=120):
    """Wait for a file to appear."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.5)
    return False


def _consume_all_inbox():
    """Consume and return all lead inbox messages."""
    return consume_lead_inbox(route_protocol=False)


# ------------------------------------------------------------------ #
# TEST 1: spawn_teammate -> result delivery via shutdown
# ------------------------------------------------------------------ #
def test_spawn_auto_delivery():
    print(f"\n{INFO} TEST 1: spawn_teammate -> result delivery")
    _reset_env()

    spawn_teammate_thread(
        name="worker-1",
        role="coder",
        prompt="Write a Python function called 'add(a,b)' that returns a+b "
               "to the file _test_add.py. That is your only task."
    )

    # Wait for the file to be created (work done)
    file_ok = _wait_file("_test_add.py", timeout=120)
    _check("teammate created _test_add.py", file_ok)

    # Give the LLM a moment to finish its current turn, then shutdown
    time.sleep(3)
    done = _shutdown_and_wait("worker-1", timeout=60)
    _check("teammate finished after shutdown", done)

    # Read inbox - result should be there
    msgs = _consume_all_inbox()
    result_msg = None
    for m in msgs:
        if m.get("type") == "result":
            result_msg = m
            break

    _check("result message received", result_msg is not None)

    if result_msg:
        _check("result has content",
               bool(result_msg.get("content")),
               f"content[:80]={result_msg.get('content', '')[:80]}")
        _check("result type is 'result'",
               result_msg.get("type") == "result")

    if os.path.exists("_test_add.py"):
        os.unlink("_test_add.py")


# ------------------------------------------------------------------ #
# TEST 2: teammate_status liveness while teammate is running
# ------------------------------------------------------------------ #
def test_teammate_status():
    print(f"\n{INFO} TEST 2: teammate_status liveness probe")
    _reset_env()

    spawn_teammate_thread(
        name="worker-2",
        role="coder",
        prompt="Write 'hello world' to _test_hello.txt. That is your only task."
    )

    time.sleep(3)
    reg = ctx.teammate_registry.get("worker-2", {})
    _check("registry has worker-2", bool(reg), f"reg keys={list(reg.keys())}")
    _check("status is 'running'", reg.get("status") == "running",
           f"status={reg.get('status')}")
    _check("phase field present", "phase" in reg, f"phase={reg.get('phase')}")
    _check("last_heartbeat present", "last_heartbeat" in reg,
           f"heartbeat={reg.get('last_heartbeat')}")
    _check("turns_total present", "turns_total" in reg,
           f"turns={reg.get('turns_total')}")

    # Wait for file, then shutdown
    _wait_file("_test_hello.txt", timeout=120)
    time.sleep(3)
    done = _shutdown_and_wait("worker-2", timeout=60)
    _check("teammate finished after shutdown", done)

    reg2 = ctx.teammate_registry.get("worker-2", {})
    _check("status is 'finished' after done",
           reg2.get("status") == "finished",
           f"status={reg2.get('status')}")

    if os.path.exists("_test_hello.txt"):
        os.unlink("_test_hello.txt")


# ------------------------------------------------------------------ #
# TEST 3: submit_plan -> review_plan metadata preservation
# ------------------------------------------------------------------ #
def test_plan_metadata_roundtrip():
    print(f"\n{INFO} TEST 3: submit_plan -> review_plan metadata roundtrip")
    _reset_env()

    task = create_task(subject="Plan-test: write a file",
                       description="Write 'plan works' to _test_plan.txt")

    spawn_teammate_thread(
        name="worker-3",
        role="coder",
        prompt=f"Claim task {task.id} with claim_task, then use submit_plan "
               f"to propose: 'I will write plan works to _test_plan.txt'. "
               f"Do NOT write the file until your plan is approved."
    )

    # Wait for plan_approval_request to arrive in lead inbox
    deadline = time.time() + 120
    plan_msg = None
    while time.time() < deadline:
        if _lead_mailbox_nonempty():
            msgs = _consume_all_inbox()
            for m in msgs:
                if m.get("type") == "plan_approval_request":
                    plan_msg = m
                    break
            if plan_msg:
                break
        time.sleep(0.5)

    _check("plan_approval_request received", plan_msg is not None,
           f"type={plan_msg.get('type') if plan_msg else 'None'}")

    if plan_msg:
        meta = plan_msg.get("metadata", {})
        req_id = meta.get("request_id", "")
        meta_task_id = meta.get("task_id", "")

        _check("metadata has request_id", bool(req_id),
               f"request_id={req_id}")
        _check("metadata has task_id", bool(meta_task_id),
               f"task_id={meta_task_id}")
        _check("task_id matches created task",
               meta_task_id == task.id,
               f"expected={task.id} got={meta_task_id}")

        formatted = format_inbox_msg(plan_msg)
        _check("format_inbox_msg preserves request_id",
               f"req:{req_id}" in formatted,
               f"formatted={formatted[:120]}")
        _check("format_inbox_msg preserves task_id",
               f"task:{task.id}" in formatted,
               f"formatted={formatted[:120]}")

        # Approve the plan
        approval = run_review_plan(request_id=req_id, approve=True,
                                   feedback="Approved. Proceed.")
        _check("run_review_plan returns success",
               "approved" in approval.lower(),
               f"response={approval}")

        # Wait for file to be created after approval
        file_ok = _wait_file("_test_plan.txt", timeout=120)
        _check("_test_plan.txt created after approval", file_ok)

        # Shutdown the teammate
        time.sleep(3)
        _shutdown_and_wait("worker-3", timeout=60)

    if os.path.exists("_test_plan.txt"):
        os.unlink("_test_plan.txt")


# ------------------------------------------------------------------ #
# TEST 4: mid-work send_message from teammate while running
# ------------------------------------------------------------------ #
def test_mid_work_message():
    print(f"\n{INFO} TEST 4: mid-work send_message from teammate")
    _reset_env()

    spawn_teammate_thread(
        name="worker-4",
        role="coder",
        prompt="Use send_message to send 'HELLO_FROM_TEAMMATE' to 'lead'. "
               "Then write 'ok' to _test_mid.txt."
    )

    # Wait for the message to arrive (teammate may still be running)
    deadline = time.time() + 120
    found_msg = False
    while time.time() < deadline:
        if _lead_mailbox_nonempty():
            msgs = _consume_all_inbox()
            for m in msgs:
                content = m.get("content", "")
                if "HELLO_FROM_TEAMMATE" in content:
                    found_msg = True
                    break
            if found_msg:
                break
        time.sleep(0.5)

    _check("mid-work message received", found_msg,
           "HELLO_FROM_TEAMMATE found in inbox")

    # Wait for file, then shutdown
    _wait_file("_test_mid.txt", timeout=120)
    time.sleep(3)
    _shutdown_and_wait("worker-4", timeout=60)

    _check("_test_mid.txt created", os.path.exists("_test_mid.txt"))

    if os.path.exists("_test_mid.txt"):
        os.unlink("_test_mid.txt")


# ------------------------------------------------------------------ #
# TEST 5: orphan task release on teammate exit
# ------------------------------------------------------------------ #
def test_orphan_release_on_exit():
    print(f"\n{INFO} TEST 5: orphan task release on teammate exit")
    _reset_env()

    task = create_task(subject="Orphan-test: should be released",
                       description="This task will be orphaned")

    spawn_teammate_thread(
        name="worker-5",
        role="coder",
        prompt=f"Claim task {task.id} with claim_task. "
               f"Write 'claimed' to _test_orphan.txt. "
               f"Do NOT complete the task."
    )

    # Wait for file (work done), then shutdown
    _wait_file("_test_orphan.txt", timeout=120)
    time.sleep(3)
    done = _shutdown_and_wait("worker-5", timeout=60)
    _check("teammate finished after shutdown", done)

    # The task should be released back to pending (orphan release in finally)
    deadline = time.time() + 15
    released = False
    while time.time() < deadline:
        try:
            t = load_task(task.id)
            if t.status == "pending" or (t.owner is None):
                released = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    _check("orphaned task released back to pending",
           released,
           f"task status after teammate exit")

    if os.path.exists("_test_orphan.txt"):
        os.unlink("_test_orphan.txt")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 70)
    print("  Real API E2E Test for teamagent pipeline")
    print("=" * 70)

    tests = [
        ("TEST 1: auto delivery", test_spawn_auto_delivery),
        ("TEST 2: teammate_status", test_teammate_status),
        ("TEST 3: plan metadata", test_plan_metadata_roundtrip),
        ("TEST 4: mid-work message", test_mid_work_message),
        ("TEST 5: orphan release", test_orphan_release_on_exit),
    ]

    for label, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  [{FAIL}] {label} exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((label, False))

    print("\n" + "=" * 70)
    total = len(results)
    passed = sum(1 for _, c in results if c)
    print(f"  Results: {passed}/{total} passed")
    for name, cond in results:
        print(f"    {'OK' if cond else 'XX'} {name}")
    print("=" * 70)
    sys.exit(0 if passed == total else 1)
