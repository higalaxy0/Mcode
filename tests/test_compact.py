"""Context compaction logic tests.

Covers the pure-function parts of ``mcodecore.compact`` (no LLM calls):
token estimation, turn grouping, orphan trimming, snip/micro compaction,
tool_result_budget, and persistence.
"""

from __future__ import annotations

from mcodecore import compact
from mcodecore.context import ctx


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #

def test_estimate_tokens_basic():
    msgs = [{"role": "user", "content": "x" * 40}]  # 40 chars ~ 10 tokens
    est = compact.estimate_tokens_messages(msgs)
    assert est >= 10  # 40/4=10 + overhead


def test_estimate_tokens_list_content():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}]
    est = compact.estimate_tokens_messages(msgs)
    assert est > 0


def test_estimate_tokens_with_tool_calls():
    msgs = [{"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "function": {"name": "bash",
                          "arguments": '{"command":"ls"}'}}]}]
    est = compact.estimate_tokens_messages(msgs)
    assert est > 0


# --------------------------------------------------------------------------- #
# ensure_valid_start
# --------------------------------------------------------------------------- #

def test_ensure_valid_start_strips_leading_tool():
    msgs = [
        {"role": "tool", "tool_call_id": "c1", "content": "orphan"},
        {"role": "user", "content": "hi"},
    ]
    out = compact.ensure_valid_start(msgs)
    assert out[0]["role"] == "user"


# --------------------------------------------------------------------------- #
# group_turns
# --------------------------------------------------------------------------- #

def _tool_call_assistant(cid="c1", name="bash"):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name, "arguments": "{}"}}]}


def test_group_turns_basic():
    msgs = [
        {"role": "user", "content": "u1"},
        _tool_call_assistant("c1"),
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "user", "content": "u2"},
        _tool_call_assistant("c2"),
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        {"role": "assistant", "content": "done"},
    ]
    turns = compact.group_turns(msgs)
    # turn boundaries occur at an assistant message with tool_calls
    assert len(turns) >= 2


def test_group_turns_single_turn_no_tool_calls():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    turns = compact.group_turns(msgs)
    assert len(turns) == 1


# --------------------------------------------------------------------------- #
# _strip_orphan_tail / _strip_orphan_head  -- orphan tool safety
# --------------------------------------------------------------------------- #

def test_strip_orphan_tail_removes_dangling_tool():
    msgs = [
        {"role": "user", "content": "u"},
        _tool_call_assistant("c1"),
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": "orphan"},  # orphan
    ]
    out = compact._strip_orphan_tail(list(msgs))
    # orphan tool message with no matching tool_call must be removed
    assert not any(m.get("tool_call_id") == "c2" for m in out)
    # legitimate tool result (c1) must be preserved
    assert any(m.get("tool_call_id") == "c1" for m in out)
    assert len(out) < len(msgs)


def test_strip_orphan_tail_replaces_dangling_tool_calls():
    # Trailing assistant with tool_calls but no result -> replaced with pure content
    msgs = [
        {"role": "user", "content": "u"},
        _tool_call_assistant("c1"),
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        _tool_call_assistant("c2"),  # no following tool result
    ]
    out = compact._strip_orphan_tail(list(msgs))
    last = out[-1]
    assert last["role"] == "assistant"
    assert "tool_calls" not in last
    assert last["content"]


def test_strip_orphan_head_removes_leading_tool():
    msgs = [
        {"role": "tool", "tool_call_id": "c1", "content": "orphan"},
        {"role": "user", "content": "u"},
    ]
    out = compact._strip_orphan_head(list(msgs))
    assert out[0]["role"] == "user"


def test_strip_orphan_head_keeps_responded_tool_calls():
    msgs = [
        _tool_call_assistant("c1"),
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "user", "content": "u"},
    ]
    out = compact._strip_orphan_head(list(msgs))
    assert out[0]["role"] == "assistant"
    assert "tool_calls" in out[0]  # kept (has a response)


# --------------------------------------------------------------------------- #
# snip_compact
# --------------------------------------------------------------------------- #

def test_snip_compact_noop_when_few_turns():
    msgs = [{"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"}]
    out = compact.snip_compact(msgs, min_keep_turns=25)
    assert out is msgs  # too few turns, returned as-is


def test_snip_compact_inserts_snippet_marker():
    # Build 30 turns (each with a tool call) to trigger snip
    msgs = []
    for i in range(30):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append(_tool_call_assistant(f"c{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
    msgs.append({"role": "assistant", "content": "final"})
    out = compact.snip_compact(msgs, min_keep_turns=5)
    markers = [
        m for m in out
        if (m.get("content") or "").startswith("[snipped") and m.get("role") == "user"
    ]
    assert len(markers) == 1
    # both head and tail are kept
    assert any(m["role"] == "assistant" and m["content"] == "final" for m in out)


def test_snip_compact_does_not_accumulate_markers():
    """Repeated snip runs must not stack stale `[snipped ...]` placeholders in head.

    Simulates the real cycle: snip once, then the conversation keeps growing so a
    second snip fires on top of an already-snipped list (which still has enough turns
    to re-trigger). The old marker must be dropped, not retained next to the new one.
    """
    msgs = []
    for i in range(40):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append(_tool_call_assistant(f"c{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
    msgs.append({"role": "assistant", "content": "mid"})

    # first snip -> 1 head + marker + 5 tail = 6 turns, re-group keeps ~6 turns.
    out = compact.snip_compact(msgs, min_keep_turns=5)
    assert sum(1 for m in out if (m.get("content") or "").startswith("[snipped")) == 1

    # grow the conversation again with many fresh turns so the second snip re-triggers.
    for i in range(40, 90):
        out.append({"role": "user", "content": f"u{i}"})
        out.append(_tool_call_assistant(f"c{i}"))
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
    out.append({"role": "assistant", "content": "final"})

    out2 = compact.snip_compact(out, min_keep_turns=5)
    markers = [
        m for m in out2
        if (m.get("content") or "").startswith("[snipped") and m.get("role") == "user"
    ]
    assert len(markers) == 1, "stale snip markers accumulated across runs"


# --------------------------------------------------------------------------- #
# snip_compact -- pin user task prompts & context block
# --------------------------------------------------------------------------- #

def test_snip_pins_all_user_task_prompts_before_tail():
    """Every genuine user prompt before the tail window must survive snip (Q1)."""
    msgs = []
    for i in range(30):
        msgs.append({"role": "user", "content": f"task-{i}"})
        msgs.append(_tool_call_assistant(f"c{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
    msgs.append({"role": "assistant", "content": "final"})

    out = compact.snip_compact(msgs, min_keep_turns=5)
    # all early task prompts (before the tail) are pinned, not snipped away
    pinned = [m["content"] for m in out
              if m.get("role") == "user" and m.get("content", "").startswith("task-")]
    # 30 turns -> tail keeps 5 -> 25 prompts live before the tail window
    assert "task-0" in pinned
    assert "task-24" in pinned
    # the 5 most-recent prompts belong to the tail window (not pinned) but still present
    assert any(m.get("content") == "task-29" for m in out)


def test_snip_keeps_new_task_prompt_when_old_task_ran_long():
    """A new task prompt must not be compressed away after a long prior task (Q2)."""
    msgs = []
    # --- old task A: many rounds, push the tail window far past task B's prompt ---
    msgs.append({"role": "user", "content": "Task A: refactor auth"})
    for i in range(40):
        msgs.append(_tool_call_assistant(f"a{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"a{i}", "content": f"ra{i}"})
    # --- new task B: the agent must remember this after snip ---
    msgs.append({"role": "user", "content": "Task B: fix login bug"})
    for i in range(30):
        msgs.append(_tool_call_assistant(f"b{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"b{i}", "content": f"rb{i}"})
    msgs.append({"role": "assistant", "content": "done B"})

    out = compact.snip_compact(msgs, min_keep_turns=5)
    contents = [m.get("content") for m in out]
    assert "Task A: refactor auth" in contents, "old task identity lost"
    assert "Task B: fix login bug" in contents, "new task prompt was snipped away"


def test_snip_placeholder_carries_context_block():
    """The snip placeholder must carry rebuilt context (files / todos / etc.) (Q3)."""
    ctx.current_todos = [
        {"content": "implement X", "status": "in_progress"},
        {"content": "write tests", "status": "pending"},
    ]
    try:
        msgs = []
        for i in range(30):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append(_tool_call_assistant(f"c{i}", name="read_file"))
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
        msgs.append({"role": "assistant", "content": "final"})

        out = compact.snip_compact(msgs, min_keep_turns=5)
        marker = next(m for m in out
                      if (m.get("content") or "").startswith("[snipped"))
        assert "implement X" in marker["content"], "todo list not injected into placeholder"
    finally:
        ctx.current_todos = []


def test_snip_excludes_synthetic_user_messages_from_pin():
    """Injected [Inbox]/[Compacted] style user messages must not be pinned."""
    msgs = []
    msgs.append({"role": "user", "content": "real task prompt"})
    for i in range(30):
        msgs.append(_tool_call_assistant(f"c{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
    # synthetic injected user messages that must NOT be pinned
    msgs.append({"role": "user", "content": "[Inbox] new message arrived"})
    for i in range(30, 35):
        msgs.append(_tool_call_assistant(f"c{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
    msgs.append({"role": "assistant", "content": "final"})

    out = compact.snip_compact(msgs, min_keep_turns=5)
    contents = [m.get("content") for m in out if m.get("role") == "user"]
    assert "real task prompt" in contents
    assert "[Inbox] new message arrived" not in contents, "synthetic message was pinned"


# --------------------------------------------------------------------------- #
# micro_compact
# --------------------------------------------------------------------------- #

def test_micro_compact_noop_when_few_tool_turns():
    msgs = [
        {"role": "user", "content": "u"},
        _tool_call_assistant("c1"),
        {"role": "tool", "tool_call_id": "c1", "content": "r" * 200},
    ]
    out = compact.micro_compact(msgs)
    # only one tool turn, not compacted
    assert "compacted" not in out[2]["content"]


def test_micro_compact_replaces_old_tool_results():
    # 30 turns with long tool results, exceeding KEEP_RECENT_LOOP_TURN
    msgs = []
    for i in range(30):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append(_tool_call_assistant(f"c{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": "X" * 200})
    out = compact.micro_compact(msgs)
    compacted = [m for m in out if m.get("role") == "tool"
                 and "compacted" in m.get("content", "")]
    assert len(compacted) > 0
    # the most recent N turns' tool results are kept (not compacted)
    recent_tool = [m for m in out if m.get("role") == "tool"]
    assert any("X" * 10 in m["content"] for m in recent_tool[-5:])


# --------------------------------------------------------------------------- #
# tool_result_budget
# --------------------------------------------------------------------------- #

def test_tool_result_budget_noop_under_limit():
    msgs = [
        {"role": "user", "content": "u"},
        _tool_call_assistant("c1"),
        {"role": "tool", "tool_call_id": "c1", "content": "small"},
    ]
    out = compact.tool_result_budget(msgs, max_bytes=200_000)
    assert out[2]["content"] == "small"


def test_tool_result_budget_persists_oversized():
    big = "A" * 40000  # exceeds PERSIST_THRESHOLD(30000)
    msgs = [
        {"role": "user", "content": "u"},
        _tool_call_assistant("c1"),
        {"role": "tool", "tool_call_id": "c1", "content": big},
    ]
    out = compact.tool_result_budget(msgs, max_bytes=100)
    assert "persisted-output" in out[2]["content"]


# --------------------------------------------------------------------------- #
# persist_large_output
# --------------------------------------------------------------------------- #

def test_persist_large_output_under_threshold_unchanged():
    out = compact.persist_large_output("c1", "short")
    assert out == "short"


def test_persist_large_output_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(compact, "PERSIST_THRESHOLD", 10)
    monkeypatch.setattr(compact, "TOOL_RESULTS_DIR", tmp_path)
    out = compact.persist_large_output("c1", "X" * 500)
    assert "persisted-output" in out
    assert (tmp_path / "c1.txt").exists()
    assert "X" * 500 in (tmp_path / "c1.txt").read_text()


# --------------------------------------------------------------------------- #
# Calibrator integration
# --------------------------------------------------------------------------- #

def test_estimate_tokens_uses_calibrator():
    # Record a sample; the calibration factor should change the estimate
    ctx.calibrator.samples.clear()
    ctx.calibrator.calibration_factor = 1.0
    est_before = compact.estimate_tokens_messages(
        [{"role": "user", "content": "x" * 8000}])
    ctx.calibrator.record(8000, 4000)  # actual/est = 0.5
    est_after = compact.estimate_tokens_messages(
        [{"role": "user", "content": "x" * 8000}])
    assert est_after < est_before  # factor 0.5 reduced the estimate
