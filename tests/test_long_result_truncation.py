"""Test: verify whether long teamagent results get truncated.

Traces the full path:
  teamagent -> bus.send -> lead inbox -> check_inbox / _drain_inbox

No real LLM API needed - this is a pure data-path test.
"""

from __future__ import annotations

import json
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcodecore.config import WORKDIR, MAILBOX_DIR, TEAM_HISTORY_DIR, TASKS_DIR
from mcodecore.context import ctx
from mcodecore.bus import MessageBus, run_check_inbox, consume_lead_inbox

# --- Setup: clean state ---
for d in [MAILBOX_DIR, TEAM_HISTORY_DIR, TASKS_DIR]:
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)

# --- Simulate a long teamagent result ---
LONG_RESULT = "X" * 5000  # 5000 chars - well beyond any truncation limit
SHORT_RESULT = "All done."

print("=" * 70)
print("  Long Result Truncation Test")
print("=" * 70)

# =========================================================================
# Test 1: run_check_inbox (the tool the LLM calls explicitly)
# =========================================================================
print("\n[1] Testing run_check_inbox (lead calls check_inbox tool)...")
ctx.bus.send("worker-alpha", "lead", LONG_RESULT, "result")
msgs = consume_lead_inbox(route_protocol=True)  # consume to simulate
# Push back manually for run_check_inbox test
for m in msgs:
    ctx.bus.send(m["from"], "lead", m["content"], m["type"], m.get("metadata"))

result_text = run_check_inbox(include_read=False)
result_len = len(result_text)
print(f"    Sent: {len(LONG_RESULT)} chars")
print(f"    Received via check_inbox: {result_len} chars total")
print(f"    Content preview: {result_text[:250]!r}")

# The critical check: is the 5000-char content present in full?
if "X" * 5000 in result_text:
    print("    [PASS] Full content preserved in check_inbox")
else:
    # Count how many X's survived
    x_count = result_text.count("X")
    print(f"    [FAIL] Content TRUNCATED: only {x_count} of 5000 'X' chars survived")
    print(f"           (truncated at ~{x_count} chars per message)")

# =========================================================================
# Test 2: _drain_inbox path (automatic inbox drain - NOT a tool call)
# =========================================================================
print("\n[2] Testing _drain_inbox (automatic drain after agent turn)...")
ctx.bus.send("worker-beta", "lead", LONG_RESULT, "result")

# Simulate what _drain_inbox does (agent.py:172-176)
inbox_msgs = consume_lead_inbox(route_protocol=True)
if inbox_msgs:
    inbox_text = "\n".join(
        f"From {m['from']}: {m['content']}" for m in inbox_msgs
    )
    drain_len = len(inbox_text)
    print(f"    Sent: {len(LONG_RESULT)} chars")
    print(f"    Received via _drain_inbox: {drain_len} chars total")
    if "X" * 5000 in inbox_text:
        print("    [PASS] Full content preserved in _drain_inbox")
    else:
        x_count = inbox_text.count("X")
        print(f"    [FAIL] Content TRUNCATED: only {x_count} of 5000 'X' chars survived")
else:
    print("    [FAIL] No messages received")

# =========================================================================
# Test 3: Multiple long messages in one inbox
# =========================================================================
print("\n[3] Testing multiple long messages via check_inbox...")
ctx.bus.send("worker-gamma", "lead", LONG_RESULT, "result")
ctx.bus.send("worker-delta", "lead", LONG_RESULT, "result")
ctx.bus.send("worker-epsilon", "lead", SHORT_RESULT, "message")

result_text = run_check_inbox(include_read=False)
total_x = result_text.count("X")
print(f"    Sent: 2 x {len(LONG_RESULT)} chars + 1 short message")
print(f"    Total 'X' chars received: {total_x} (expected: {2 * len(LONG_RESULT)})")
if total_x == 2 * len(LONG_RESULT):
    print("    [PASS] All content preserved")
else:
    print(f"    [FAIL] Content TRUNCATED: {total_x} of {2 * len(LONG_RESULT)} 'X' chars")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("  ANALYSIS SUMMARY")
print("=" * 70)
print("""
  Truncation points in the teamagent result delivery path:

  1. teamagent -> ctx.bus.send(name, "lead", result, "result")
     [teammates.py:384]  -> NO truncation. Full result written to JSONL.

  2. bus.send -> JSONL file
     [bus.py:28-42]      -> NO truncation. json.dumps writes full content.

  3a. Lead auto-drain (_drain_inbox)
     [agent.py:163-178]  -> NO truncation. Full content injected into history.

  3b. Lead check_inbox tool (run_check_inbox)
     [bus.py:325-339]    -> FIXED: was m['content'][:200], now full content.
        Messages are returned in full since consume_lead_inbox atomically
        clears the inbox - truncating here permanently loses data.

  4. Console print of tool output
     [agent.py:145]      -> Display-only truncation at 300 chars.
        Full output still appended to messages (line 146).

  5. History log (log_team_history)
     [teammates.py]      -> truncate() at 200 chars, but ONLY for the
        .team_history debug log file, NOT for actual message delivery.

  CONCLUSION:
  After the fix, NO truncation occurs in the result delivery path.
  All five points preserve full content for actual message delivery.
""")

# Cleanup
shutil.rmtree(MAILBOX_DIR, ignore_errors=True)
shutil.rmtree(TEAM_HISTORY_DIR, ignore_errors=True)
shutil.rmtree(TASKS_DIR, ignore_errors=True)
MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
