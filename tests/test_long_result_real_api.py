"""Real-LLM test: verify long teamagent results are NOT truncated.

Scenario:
  1. Spawn a teamagent asked to produce a long detailed report (>500 chars).
  2. The teamagent does the work, then sends a long final result to lead.
  3. Lead consumes the inbox via run_check_inbox (the previously buggy path).
  4. Verify the full result is preserved -- no 200-char truncation.

This exercises the fix in bus.py run_check_inbox: m['content'][:200] -> m['content'].
"""

from __future__ import annotations

import json
import sys
import time
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcodecore.teammates import spawn_teammate_thread
from mcodecore.bus import run_check_inbox, run_request_shutdown
from mcodecore.context import ctx

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
TEAMMATE_NAME = "long-result-worker"
MAX_WAIT_SECONDS = 240
MIN_RESULT_LEN = 500       # result must be at least this long to be "long"

# Clean up stale state
WORKDIR = Path.cwd()
for cleanup in [
    WORKDIR / ".mailboxes",
    WORKDIR / ".team_history",
    WORKDIR / ".tasks",
    WORKDIR / "long_report.txt",
]:
    if cleanup.is_dir():
        shutil.rmtree(cleanup, ignore_errors=True)
    elif cleanup.exists():
        cleanup.unlink()
(WORKDIR / ".mailboxes").mkdir(exist_ok=True)
(WORKDIR / ".tasks").mkdir(exist_ok=True)


def main() -> None:
    print("=" * 70)
    print("  Long Result Truncation Test (Real LLM API)")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Step 1: Spawn teamagent with a task that produces a long result.
    # ------------------------------------------------------------------ #
    prompt = (
        "You are a code reviewer. Write a detailed code review report "
        "with at least 6 sections, each section having a heading and "
        "2-3 sentences of analysis. The report must be at least 500 "
        "characters long. Cover: architecture, error handling, naming "
        "conventions, test coverage, performance, and security. "
        "When done, send the FULL report to 'lead' via send_message, "
        "then finish. The report you send must be the complete long "
        "version -- do not summarize or shorten it."
    )

    print(f"\n[1] Spawning teamagent '{TEAMMATE_NAME}'...")
    spawn_teammate_thread(TEAMMATE_NAME, "code-reviewer", prompt)
    spawn_time = time.time()

    # ------------------------------------------------------------------ #
    # Step 2: Wait for the teamagent to finish.
    # ------------------------------------------------------------------ #
    print(f"\n[2] Waiting up to {MAX_WAIT_SECONDS}s for teamagent to finish...")
    start = time.time()
    finished = False
    last_print = 0
    while time.time() - start < MAX_WAIT_SECONDS:
        evt = ctx.active_teammates.get(TEAMMATE_NAME)
        if evt is None or evt.is_set():
            finished = True
            break
        now = time.time()
        if now - last_print > 20:
            print(f"    ... still running (T+{now - spawn_time:.0f}s)")
            last_print = now
        time.sleep(2)

    elapsed = time.time() - start
    if finished:
        print(f"    Teamagent finished after {elapsed:.1f}s")
    else:
        print(f"    [WARNING] Teamagent did not finish within {MAX_WAIT_SECONDS}s")

    # ------------------------------------------------------------------ #
    # Step 3: Consume lead's inbox via run_check_inbox (the fixed path).
    # ------------------------------------------------------------------ #
    print("\n[3] Consuming lead inbox via run_check_inbox...")
    inbox_text = run_check_inbox(include_read=False)
    inbox_len = len(inbox_text)
    print(f"    Inbox text length: {inbox_len} chars")

    if inbox_text == "(inbox empty)":
        print("    [FAIL] Inbox is empty -- no result received")
        _cleanup()
        sys.exit(1)

    # Print a preview (first and last 200 chars)
    preview_head = inbox_text[:200]
    preview_tail = inbox_text[-200:] if len(inbox_text) > 200 else ""
    print(f"    Head: {preview_head!r}")
    if preview_tail:
        print(f"    Tail: {preview_tail!r}")

    # ------------------------------------------------------------------ #
    # Step 4: Verify the result is NOT truncated.
    # ------------------------------------------------------------------ #
    print(f"\n[4] Verifying result length >= {MIN_RESULT_LEN} chars...")
    print(f"    Received: {inbox_len} chars")

    # Core assertion: the inbox text must be significantly longer than
    # the old 200-char truncation limit. If it's exactly ~200 chars,
    # the truncation bug is still present.
    checks = []

    # Check 1: total length exceeds old truncation limit
    check1 = inbox_len > 250  # 200 + some overhead for headers
    checks.append(("Inbox text > 250 chars (not truncated to 200)", check1))

    # Check 2: the content portion is long enough to be a real report
    # Subtract approximate header overhead (~30 chars per message)
    content_len = max(0, inbox_len - 50)
    check2 = content_len >= MIN_RESULT_LEN
    checks.append((f"Content >= {MIN_RESULT_LEN} chars", check2))

    # Check 3: no truncation marker -- the old bug would cut at exactly 200
    # chars of content, so if we see content well beyond 200, we're good.
    # We check that the text doesn't end abruptly mid-word at ~200 chars.
    check3 = inbox_len > 400
    checks.append(("Inbox text > 400 chars (well beyond old limit)", check3))

    # ------------------------------------------------------------------ #
    # Step 5: Also verify via raw mailbox inspection (before check_inbox
    #         consumed it). Since check_inbox already consumed the inbox,
    #         we check the team_history log for the actual result length.
    # ------------------------------------------------------------------ #
    print("\n[5] Cross-checking with team history log...")
    history_file = WORKDIR / ".team_history" / f"{TEAMMATE_NAME}.jsonl"
    history_result_len = 0
    if history_file.exists():
        lines = history_file.read_text(encoding="utf-8").strip().splitlines()
        for line in reversed(lines):
            try:
                rec = json.loads(line)
                if rec.get("event") == "finished":
                    detail = rec.get("detail", {})
                    raw_result = detail.get("result", "")
                    history_result_len = len(raw_result)
                    print(f"    History 'finished' result length: {history_result_len} chars")
                    print(f"    (note: history log truncates to 200 chars for logging only)")
                    break
            except json.JSONDecodeError:
                pass
    else:
        print("    (no history file)")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    all_pass = True
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  >>> ALL CHECKS PASSED - long result is NOT truncated!")
        print("  >>> The run_check_inbox fix (removing [:200]) works correctly")
        print("  >>> with real LLM-generated content.")
    else:
        print("  >>> SOME CHECKS FAILED - result may still be truncated.")
        print("  >>> Expected: full long report (>500 chars)")
        print(f"  >>> Got: {inbox_len} chars")

    _cleanup()
    sys.exit(0 if all_pass else 1)


def _cleanup():
    """Request shutdown if still running, then clean up test artifacts."""
    evt = ctx.active_teammates.get(TEAMMATE_NAME)
    if evt is not None and not evt.is_set():
        print("\n  (Requesting shutdown of lingering teamagent...)")
        run_request_shutdown(TEAMMATE_NAME)
        evt.wait(timeout=10)

    for cleanup in [
        WORKDIR / ".mailboxes",
        WORKDIR / ".team_history",
        WORKDIR / ".tasks",
        WORKDIR / "long_report.txt",
    ]:
        if cleanup.is_dir():
            shutil.rmtree(cleanup, ignore_errors=True)
        elif cleanup.exists():
            cleanup.unlink()
    (WORKDIR / ".mailboxes").mkdir(exist_ok=True)
    (WORKDIR / ".tasks").mkdir(exist_ok=True)


if __name__ == "__main__":
    main()
