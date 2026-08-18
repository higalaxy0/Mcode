"""Multi-dimension real-API message passing tests for the teamagent bus.

Complements tests/test_teamagent_real_api.py (which covers spawn/result,
status liveness, plan approval, mid-work send, orphan release) with the
following message-passing dimensions, all driven through the real LLM:

  D1  peer-to-peer: teammate-1 -> teammate-2 mailbox -> relay to lead
  D2  lead -> teammate mid-work delivery + acknowledgement
  D3  idle_poll wake-up: lead message revives an idling teammate
  D4  concurrent fan-in: 3 teammates x (message + finish result), integrity
  D5  plan rejection: review_plan(approve=False) feedback round-trip
  D6  request_plan: lead triggers a plan submission from an idle teammate
  D7  payload fidelity: CJK / special chars / ~5KB content via send_message
  D8  task dependency gating + "Unblocked" notification (mixed LLM/direct)
  D9  bus-level concurrent append integrity (3 threads x 150 msgs, no LLM)

Each test prints a "[finished] ..." marker when done so progress can be
monitored from the log file of a background run.
"""
import sys
import os
import json
import time
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
    consume_lead_inbox, run_review_plan, run_request_shutdown,
    run_request_plan, run_send_message,
)
from mcodecore.teammates import spawn_teammate_thread
from mcodecore.tasks import (create_task, load_task, claim_task,
                             complete_task, release_task)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36mTEST\033[0m"

results = []


def _check(name, cond, detail=""):
    status = PASS if cond else FAIL
    results.append((name, bool(cond)))
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))


def _wait_file(path, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.5)
    return False


def _wait_teammate_done(name, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        evt = ctx.active_teammates.get(name)
        if evt is None or evt.is_set():
            return True
        time.sleep(0.5)
    return False


def _shutdown_and_wait(name, timeout=90):
    run_request_shutdown(name)
    return _wait_teammate_done(name, timeout)


class InboxCollector:
    """Accumulates every message that ever hits the lead inbox."""

    def __init__(self):
        self.msgs = []

    def pump(self):
        self.msgs.extend(consume_lead_inbox(route_protocol=True))

    def wait(self, pred, timeout=180):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.pump()
            for m in self.msgs:
                try:
                    if pred(m):
                        return m
                except Exception:
                    continue
            time.sleep(0.5)
        return None

    def count(self, pred):
        self.pump()
        return sum(1 for m in self.msgs if pred(m))


def _rm(*paths):
    for p in paths:
        if os.path.exists(p):
            os.unlink(p)


# ------------------------------------------------------------------ #
# D1: peer-to-peer teammate messaging
# ------------------------------------------------------------------ #
def test_d1_peer_to_peer():
    print(f"\n{INFO} D1: peer-to-peer teammate messaging (t1 -> t2 -> lead)")
    _reset_env()
    MARK = "PEER_HELLO_7f3a"

    spawn_teammate_thread(
        name="peer-2",
        role="relay",
        prompt="Stand by: do not call any tools yet, just reply with the "
               "text 'waiting'. You will receive a message from teammate "
               "'peer-1'. When you receive it, use send_message to send to "
               f"'lead' the text: RELAY:{MARK} . Then call the finish tool "
               "with summary 'relayed'."
    )
    time.sleep(2)
    spawn_teammate_thread(
        name="peer-1",
        role="sender",
        prompt=f"Call send_message to send the text '{MARK}' to 'peer-2' "
               "(to='peer-2'). Then call the finish tool with summary "
               "'sent to peer'."
    )

    box = InboxCollector()
    relay = box.wait(lambda m: "content" in m and f"RELAY:{MARK}" in m.get("content", ""),
                     timeout=240)
    _check("peer-2 relayed peer-1 message to lead", relay is not None)

    sent_result = box.wait(lambda m: m.get("type") == "result"
                           and m.get("from") == "peer-1", timeout=60)
    _check("peer-1 result delivered", sent_result is not None)

    _wait_teammate_done("peer-1", timeout=60)
    _wait_teammate_done("peer-2", timeout=60)
    print("[finished] D1")


# ------------------------------------------------------------------ #
# D2: lead -> teammate mid-work message + acknowledgement
# ------------------------------------------------------------------ #
def test_d2_lead_midwork_ack():
    print(f"\n{INFO} D2: lead -> teammate mid-work delivery + ack")
    _reset_env()
    ACK = "ACK_LEAD_MSG_9c2e"

    spawn_teammate_thread(
        name="busy-1",
        role="coder",
        prompt="Do exactly 5 slow steps, one per turn. For i in 1..5: "
               "first run bash 'python -c \"import time; time.sleep(6)\"' "
               "(mandatory, do not skip), then use write_file to create "
               f"_test_b{{i}}.txt with content 'step{{i}}'. After step 5 "
               "call the finish tool with summary 'steps done'."
    )
    time.sleep(10)
    run_send_message("busy-1",
                     "Acknowledge this instruction NOW by calling "
                     f"send_message to 'lead' with content '{ACK}', "
                     "then continue your remaining steps.")

    box = InboxCollector()
    ack = box.wait(lambda m: ACK in m.get("content", ""), timeout=180)
    _check("teammate acknowledged mid-work lead message", ack is not None)

    files_ok = all(_wait_file(f"_test_b{i}.txt", timeout=120)
                   for i in range(1, 6))
    _check("all 5 step files written after ack", files_ok)

    done = _wait_teammate_done("busy-1", timeout=180)
    _check("busy-1 finished", done)
    _rm(*[f"_test_b{i}.txt" for i in range(1, 6)])
    print("[finished] D2")


# ------------------------------------------------------------------ #
# D3: idle_poll wake-up by lead message
# ------------------------------------------------------------------ #
def test_d3_idle_wakeup():
    print(f"\n{INFO} D3: idle_poll wake-up by lead message")
    _reset_env()

    spawn_teammate_thread(
        name="idle-1",
        role="coder",
        prompt="Do not call any tools now. Reply with exactly the text "
               "'standing by' and nothing else, then wait silently."
    )
    time.sleep(15)  # let the teammate reach idle_poll
    reg = ctx.teammate_registry.get("idle-1", {})
    _check("idle-1 alive before wake-up",
           reg.get("status") == "running" and not _wait_teammate_done("idle-1", timeout=1),
           f"phase={reg.get('phase')}")

    run_send_message("idle-1",
                     "New instruction: use write_file to create "
                     "_test_wake.txt with content 'woken_by_lead', then "
                     "call the finish tool with summary 'woke and done'.")

    file_ok = _wait_file("_test_wake.txt", timeout=180)
    _check("idle teammate woke and wrote _test_wake.txt", file_ok)

    box = InboxCollector()
    res = box.wait(lambda m: m.get("type") == "result"
                   and m.get("from") == "idle-1", timeout=90)
    _check("idle-1 result delivered after wake-up", res is not None,
           f"content[:60]={(res or {}).get('content', '')[:60]}")

    _wait_teammate_done("idle-1", timeout=60)
    _rm("_test_wake.txt")
    print("[finished] D3")


# ------------------------------------------------------------------ #
# D4: concurrent fan-in from 3 teammates
# ------------------------------------------------------------------ #
def test_d4_concurrent_fanin():
    print(f"\n{INFO} D4: concurrent fan-in (3 teammates)")
    _reset_env()
    names = ["fan-a", "fan-b", "fan-c"]

    for n in names:
        spawn_teammate_thread(
            name=n,
            role="coder",
            prompt=f"Two actions only: (1) call send_message to 'lead' "
                   f"with content 'FANIN_{n}_MSG'. (2) call the finish "
                   f"tool with summary 'FANIN_{n}_DONE'."
        )

    box = InboxCollector()
    deadline = time.time() + 300
    while time.time() < deadline:
        msgs_ok = box.count(lambda m: m.get("type") == "message"
                            and f"FANIN_{m.get('from', '')}_MSG" == f"FANIN_{m.get('from', '')}_MSG"
                            and m.get("content", "").startswith(f"FANIN_{m.get('from', '')}_MSG"))
        results_ok = box.count(lambda m: m.get("type") == "result")
        if msgs_ok >= 3 and results_ok >= 3:
            break
        time.sleep(1)

    _check("3 distinct fan-in messages received",
           box.count(lambda m: m.get("type") == "message"
                     and m.get("content", "").startswith("FANIN_")) >= 3,
           f"count={box.count(lambda m: m.get('content', '').startswith('FANIN_'))}")
    _check("3 finish results received",
           box.count(lambda m: m.get("type") == "result") >= 3)
    per_sender = {n: box.count(lambda m, n=n: m.get("from") == n
                               and m.get("content", "").startswith(f"FANIN_{n}_MSG"))
                  for n in names}
    _check("each teammate's message intact and attributed",
           all(v >= 1 for v in per_sender.values()), f"{per_sender}")

    for n in names:
        _wait_teammate_done(n, timeout=90)
    print("[finished] D4")


# ------------------------------------------------------------------ #
# D5: plan rejection feedback round-trip
# ------------------------------------------------------------------ #
def test_d5_plan_reject():
    print(f"\n{INFO} D5: plan rejection feedback round-trip")
    _reset_env()
    task = create_task(subject="Reject-test: conditional write",
                       description="Write file only if plan approved")

    spawn_teammate_thread(
        name="planner-1",
        role="coder",
        prompt=f"Claim task {task.id} with claim_task. Then call "
               f"submit_plan with plan 'Plan v1: write lowercase content' "
               f"and task_id '{task.id}'. Wait for the review result. "
               "If the plan is REJECTED: do NOT write any file; instead "
               "send_message to 'lead' with content starting with "
               "'REJECT_ACK:' followed by a short quote of the feedback, "
               "then call the finish tool. If approved, write "
               "_test_reject.txt with content 'v1'."
    )

    box = InboxCollector()
    plan_msg = box.wait(lambda m: m.get("type") == "plan_approval_request",
                        timeout=240)
    _check("plan_approval_request received", plan_msg is not None)

    if plan_msg:
        req_id = plan_msg.get("metadata", {}).get("request_id", "")
        resp = run_review_plan(request_id=req_id, approve=False,
                               feedback="Use UPPERCASE v2 format instead")
        _check("run_review_plan reject returns success",
               "rejected" in resp.lower(), f"resp={resp}")

        ack = box.wait(lambda m: m.get("content", "").startswith("REJECT_ACK:"),
                       timeout=180)
        _check("teammate acked rejection with feedback",
               ack is not None,
               f"content[:80]={(ack or {}).get('content', '')[:80]}")
        if ack:
            _check("rejection feedback quoted back",
                   "v2" in ack.get("content", "").lower()
                   or "uppercase" in ack.get("content", "").lower())

        time.sleep(5)
        _check("_test_reject.txt NOT created after rejection",
               not os.path.exists("_test_reject.txt"))

    _shutdown_and_wait("planner-1", timeout=90)
    _rm("_test_reject.txt")
    print("[finished] D5")


# ------------------------------------------------------------------ #
# D6: request_plan trigger from idle
# ------------------------------------------------------------------ #
def test_d6_request_plan():
    print(f"\n{INFO} D6: request_plan wakes an idle teammate")
    _reset_env()

    spawn_teammate_thread(
        name="planner-2",
        role="coder",
        prompt="Wait for the lead to ask you to submit a plan. Do not "
               "call any tools yet; reply only 'standing by for plan "
               "request'. When the lead asks, call submit_plan with "
               "plan='Plan: verify request_plan flow' and task_id='' "
               "(empty string). Then wait: if the plan is approved, "
               "write_file _test_reqplan.txt with content 'ok' and call "
               "the finish tool."
    )
    time.sleep(15)
    run_request_plan("planner-2", "organizing test files")

    box = InboxCollector()
    plan_msg = box.wait(lambda m: m.get("type") == "plan_approval_request",
                        timeout=240)
    _check("plan submitted after request_plan", plan_msg is not None)

    if plan_msg:
        req_id = plan_msg.get("metadata", {}).get("request_id", "")
        resp = run_review_plan(request_id=req_id, approve=True,
                               feedback="Approved")
        _check("plan approved", "approved" in resp.lower())
        file_ok = _wait_file("_test_reqplan.txt", timeout=180)
        _check("_test_reqplan.txt created after approval", file_ok)

    _wait_teammate_done("planner-2", timeout=90)
    _rm("_test_reqplan.txt")
    print("[finished] D6")


# ------------------------------------------------------------------ #
# D7: payload fidelity (CJK / special chars / ~5KB)
# ------------------------------------------------------------------ #
def test_d7_payload_fidelity():
    print(f"\n{INFO} D7: CJK / special-char / large payload fidelity")
    _reset_env()

    lines = ["PAYLOAD_START_UAF3921"]
    lines.append("中文消息测试-问候语：你好，世界！🎓🎉")
    lines.append("special: \"double\" 'single' back\\slash \t tab <tag> &amp;")
    for i in range(1, 101):
        lines.append(f"LINE_{i:03d}_{'x' * (i % 7)}")
    lines.append("PAYLOAD_END_ZK8845")
    payload = "\n".join(lines)
    with open("_test_payload.txt", "w", encoding="utf-8") as f:
        f.write(payload)

    spawn_teammate_thread(
        name="porter-1",
        role="coder",
        prompt="Read the file _test_payload.txt with read_file (it has "
               "many lines; read all of them). Then send its ENTIRE "
               "content EXACTLY, character for character including the "
               "Chinese characters and all markers, to 'lead' using ONE "
               "send_message call. Do not summarize or truncate. Then "
               "call the finish tool with summary 'payload sent'."
    )

    box = InboxCollector()
    got = box.wait(lambda m: "PAYLOAD_START_UAF3921" in m.get("content", ""),
                   timeout=300)
    _check("payload message received", got is not None)

    if got:
        content = got.get("content", "")
        _check("end marker present", "PAYLOAD_END_ZK8845" in content)
        _check("middle line LINE_050_ intact", "LINE_050_" in content)
        _check("CJK line intact (exact match)",
               "中文消息测试-问候语：你好，世界！🎓🎉" in content)
        _check("payload size preserved (>=70%)",
               len(content) >= 0.7 * len(payload),
               f"sent={len(payload)} got={len(content)}")

    _shutdown_and_wait("porter-1", timeout=90)
    _rm("_test_payload.txt")
    print("[finished] D7")


# ------------------------------------------------------------------ #
# D8: task dependency gating + unblocked notification
# ------------------------------------------------------------------ #
def test_d8_dependency_unblock():
    print(f"\n{INFO} D8: dependency gating + Unblocked notification")
    _reset_env()

    # Direct (no LLM): blocked claim refused, then unblock notification
    dep_a = create_task(subject="Dep-A: base work")
    dep_b = create_task(subject="Dep-B: depends on A", blockedBy=[dep_a.id])
    dep_c = create_task(subject="Dep-C: depends on B", blockedBy=[dep_b.id])

    r1 = claim_task(dep_b.id, owner="probe")
    _check("claim of blocked task refused", "Cannot start" in r1, f"r1={r1[:60]}")

    # LLM part: teammate completes A
    spawn_teammate_thread(
        name="dep-worker",
        role="coder",
        prompt=f"Claim task {dep_a.id} with claim_task, then use write_file "
               "to create _test_dep_a.txt with content 'a done', then call "
               f"complete_task on {dep_a.id}, then call the finish tool."
    )

    deadline = time.time() + 240
    a_done = False
    while time.time() < deadline:
        try:
            if load_task(dep_a.id).status == "completed":
                a_done = True
                break
        except Exception:
            pass
        time.sleep(1)
    _check("dep-worker completed task A", a_done)

    r2 = claim_task(dep_b.id, owner="probe")
    _check("task B claimable after A completed", r2.startswith("Claimed"),
           f"r2={r2[:60]}")

    r3 = complete_task(dep_b.id, owner="probe")
    _check("complete_task reports unblocked downstream",
           "Unblocked" in r3 and dep_c.subject in r3, f"r3={r3[:80]}")

    # Cleanup: leave C pending
    _wait_teammate_done("dep-worker", timeout=90)
    _rm("_test_dep_a.txt")
    print("[finished] D8")


# ------------------------------------------------------------------ #
# D9: bus-level concurrent append integrity (no LLM)
# ------------------------------------------------------------------ #
def test_d9_bus_concurrency():
    print(f"\n{INFO} D9: bus concurrent append integrity (infra-level)")
    _reset_env()

    N, SENDERS = 150, ["w1", "w2", "w3"]

    def sender(agent):
        for i in range(N):
            ctx.bus.send(agent, "lead", f"m{i}", "message", {"seq": i})

    threads = [threading.Thread(target=sender, args=(a,)) for a in SENDERS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    msgs = consume_lead_inbox(route_protocol=False)
    _check("all 450 messages delivered", len(msgs) == N * len(SENDERS),
           f"got={len(msgs)}")
    ok_order = True
    for a in SENDERS:
        seqs = [m.get("metadata", {}).get("seq") for m in msgs
                if m.get("from") == a]
        if seqs != sorted(seqs) or len(seqs) != N:
            ok_order = False
    _check("per-sender order preserved, zero loss/corruption", ok_order)
    print("[finished] D9")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 70)
    print("  Real API multi-dimension message-passing matrix")
    print("=" * 70)

    tests = [
        ("D1: peer-to-peer", test_d1_peer_to_peer),
        ("D2: mid-work ack", test_d2_lead_midwork_ack),
        ("D3: idle wake-up", test_d3_idle_wakeup),
        ("D4: concurrent fan-in", test_d4_concurrent_fanin),
        ("D5: plan reject", test_d5_plan_reject),
        ("D6: request_plan", test_d6_request_plan),
        ("D7: payload fidelity", test_d7_payload_fidelity),
        ("D8: dependency unblock", test_d8_dependency_unblock),
        ("D9: bus concurrency", test_d9_bus_concurrency),
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
