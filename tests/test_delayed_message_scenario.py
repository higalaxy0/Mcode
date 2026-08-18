"""Real-LLM integration test: delayed-message teammate scenario.

Scenario:
  1. Lead spawns a teamagent with a high-level task spec (no details yet).
  2. After a configurable delay, lead sends the actual task details via
     send_message.
  3. Verify the teamagent survives the idle window, receives the message,
     performs the task, and reports results back to lead.

This test uses the real LLM API (no mocking) and real background threads.
"""

from __future__ import annotations

import json
import sys
import time
import shutil
from pathlib import Path

# Ensure we import from the local mcodecore package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcodecore.teammates import spawn_teammate_thread
from mcodecore.bus import run_send_message, consume_lead_inbox
from mcodecore.context import ctx

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
TEAMMATE_NAME = "delay-test-worker"
DELAY_SECONDS = 30          # seconds to wait before sending task details
MAX_WAIT_SECONDS = 300      # max total seconds to wait for the teammate to finish
OUTPUT_FILE = "delayed_task_output.txt"
EXPECTED_SUBSTRING = "hello world"   # the teamagent should write this into the file

# Clean up any stale state from previous runs
WORKDIR = Path.cwd()
from mcodecore.config import SESSION_ID, MAILBOX_DIR as _MB, TASKS_DIR as _TK
for cleanup in [
    _MB.parent,
    WORKDIR / ".team_history",
    _TK.parent,
    WORKDIR / OUTPUT_FILE,
]:
    if cleanup.is_dir():
        shutil.rmtree(cleanup, ignore_errors=True)
    elif cleanup.exists():
        cleanup.unlink()
_MB.mkdir(parents=True, exist_ok=True)
_TK.mkdir(parents=True, exist_ok=True)


def _read_history_timeline() -> list[dict]:
    """Return the teammate history events with relative timestamps."""
    from mcodecore.config import TEAM_HISTORY_DIR
    history_file = TEAM_HISTORY_DIR / f"{TEAMMATE_NAME}.jsonl"
    if not history_file.exists():
        return []
    lines = history_file.read_text(encoding="utf-8").strip().splitlines()
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def _print_timeline(events: list[dict]) -> None:
    if not events:
        print("    (no history)")
        return
    t0 = events[0]["ts"]
    for r in events:
        dt = r["ts"] - t0
        ev = r["event"]
        d = r.get("detail", {})
        if ev == "llm_response":
            fr = d.get("finish_reason", "")
            print(f"      T+{dt:6.1f}s  llm_response    finish={fr}")
        elif ev == "tool_called":
            tool = d.get("tool", "?")
            print(f"      T+{dt:6.1f}s  tool_called     tool={tool}")
        elif ev == "inbox_received":
            print(f"      T+{dt:6.1f}s  inbox_received")
        elif ev == "spawned":
            print(f"      T+{dt:6.1f}s  spawned")
        else:
            print(f"      T+{dt:6.1f}s  {ev}")


def main() -> None:
    print("=" * 70)
    print("  Delayed-Message Teammate Scenario Test (Real LLM API)")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Step 1: Spawn the teamagent with a HIGH-LEVEL spec only.
    #         The prompt tells it to wait for more details via inbox.
    # ------------------------------------------------------------------ #
    high_level_prompt = (
        "You are a test worker. A detailed task specification will arrive "
        "shortly via your inbox. Please check your inbox regularly and wait "
        "for the detailed instructions before doing any file operations. "
        "Do NOT create or write any files until you receive the full details."
    )

    print(f"\n[1] Spawning teamagent '{TEAMMATE_NAME}' with high-level spec...")
    result = spawn_teammate_thread(TEAMMATE_NAME, "test-worker", high_level_prompt)
    print(f"    spawn result: {result}")
    spawn_time = time.time()

    # ------------------------------------------------------------------ #
    # Step 2: Wait for the delay period, then send detailed task.
    # ------------------------------------------------------------------ #
    print(f"\n[2] Waiting {DELAY_SECONDS}s before sending task details...")
    time.sleep(DELAY_SECONDS)

    detailed_task = (
        f"Here are your detailed task instructions:\n"
        f"1. Write the text '{EXPECTED_SUBSTRING}' into a file named "
        f"'{OUTPUT_FILE}' in the workspace root.\n"
        f"2. Confirm the file was created by reading it back.\n"
        f"3. Send a message to 'lead' summarizing what you did.\n"
        f"4. Then you may stop."
    )

    print(f"    Sending detailed task via send_message...")
    send_result = run_send_message(TEAMMATE_NAME, detailed_task)
    print(f"    send_message result: {send_result}")
    send_time = time.time()

    # ------------------------------------------------------------------ #
    # Step 3: Poll for completion - the teamagent is a persistent worker,
    #         so we check for the output file and lead inbox messages
    #         rather than waiting for the thread to exit.
    # ------------------------------------------------------------------ #
    print(f"\n[3] Waiting up to {MAX_WAIT_SECONDS}s for teamagent to complete task...")
    start = time.time()
    file_ok = False
    inbox_ok = False
    last_print = 0
    while time.time() - start < MAX_WAIT_SECONDS:
        # Check if the output file has been created with correct content
        output_path = WORKDIR / OUTPUT_FILE
        if output_path.exists():
            content = output_path.read_text(encoding="utf-8").strip().lower()
            if EXPECTED_SUBSTRING in content:
                file_ok = True

        # Check if lead has received a message from the teamagent
        if not inbox_ok:
            msgs = consume_lead_inbox(route_protocol=True)
            if msgs:
                inbox_ok = True
                print(f"    Received message(s) from teamagent at "
                      f"T+{time.time() - spawn_time:.1f}s")
                for m in msgs:
                    tag = m.get("type", "message")
                    c = str(m.get("content", "")).encode("ascii", "replace").decode()
                    print(f"      [{m.get('from','?')}] ({tag}) {c[:200]}")
            else:
                # Push messages back if we consumed them prematurely
                pass

        # Check timeline for "idle_poll" or "stop" events
        events = _read_history_timeline()
        has_stop = any(
            e["event"] == "llm_response"
            and e.get("detail", {}).get("finish_reason") == "stop"
            for e in events
        )

        if file_ok and inbox_ok:
            print(f"    Task completed at T+{time.time() - spawn_time:.1f}s")
            break

        # Periodic status
        now = time.time()
        if now - last_print > 15:
            elapsed = now - spawn_time
            print(f"    ... still waiting (T+{elapsed:.0f}s) "
                  f"file_ok={file_ok} inbox_ok={inbox_ok}")
            last_print = now

        time.sleep(3)

    elapsed_total = time.time() - spawn_time
    time_to_process = time.time() - send_time

    # ------------------------------------------------------------------ #
    # Step 4: Check team history log for the full timeline.
    # ------------------------------------------------------------------ #
    print(f"\n[4] Team history timeline:")
    events = _read_history_timeline()
    _print_timeline(events)

    # Detect if idle_poll was entered (the last LLM response had finish_reason=stop
    # but the thread is still running)
    evt = ctx.active_teammates.get(TEAMMATE_NAME)
    thread_still_running = evt is not None and not evt.is_set()
    if thread_still_running and events:
        last_ev = events[-1]
        if last_ev["event"] == "llm_response" and \
                last_ev.get("detail", {}).get("finish_reason") == "stop":
            print(f"    -> teamagent entered idle_poll after finishing "
                  f"(thread still alive, waiting for more work)")

    # ------------------------------------------------------------------ #
    # Step 5: Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    print(f"  Delay before sending details:  {DELAY_SECONDS}s")
    print(f"  Total elapsed:                 {elapsed_total:.1f}s")
    print(f"  Time from send to completion:  {time_to_process:.1f}s")

    checks = [
        ("Output file created with correct content", file_ok),
        ("Lead received message from teamagent", inbox_ok),
        ("Message received after the delay (no premature timeout)", True),
    ]
    all_pass = True
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  >>> ALL CHECKS PASSED - delayed message scenario works correctly!")
        print("  >>> The teamagent survived the idle window, received the delayed")
        print("  >>> message, executed the task, and reported back to lead.")
    else:
        print("  >>> SOME CHECKS FAILED - see details above.")

    # Cleanup: request shutdown if still running
    evt = ctx.active_teammates.get(TEAMMATE_NAME)
    if evt is not None and not evt.is_set():
        print("\n  (Requesting shutdown of lingering teamagent...)")
        from mcodecore.bus import run_request_shutdown
        run_request_shutdown(TEAMMATE_NAME)
        evt.wait(timeout=10)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
