"""Token estimation / context compaction / persistence."""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

from .calibrator import TokenCalibrator  # noqa: F401  (re-exported for compat)
from .config import (CONTEXT_LIMIT, KEEP_RECENT_LOOP_TURN, PERSIST_THRESHOLD,
                     TOOL_RESULTS_DIR, TRANSCRIPT_DIR, _MSG_OVERHEAD_TOKENS,
                     LLM_MODEL, client)
from .context import ctx
from .tasks import list_tasks


def _msg_content_len(m) -> int:
    """Estimate the character count of a single message's content."""
    total = 0
    c = m.get("content", "")
    if isinstance(c, str):
        total += len(c)
    elif isinstance(c, list):
        for b in c:
            if getattr(b, "type", None) == "text":
                total += len(getattr(b, "text", "") or "")
            elif isinstance(b, dict) and b.get("type") == "text":
                total += len(b.get("text", "") or "")
            else:
                total += len(str(b))
    for tc in m.get("tool_calls") or []:
        total += len(tc.get("function", {}).get("arguments", ""))
        total += len(tc.get("function", {}).get("name", ""))
    return total


def estimate_tokens_messages(msgs) -> int:
    """Roughly estimate the token count of a message list (4 chars ~= 1 token, then apply the calibration factor)."""
    estimated = 0
    for m in msgs:
        estimated += _msg_content_len(m) // 4 + _MSG_OVERHEAD_TOKENS
    return ctx.calibrator.calibrated(estimated)


def ensure_valid_start(messages: list) -> list:
    """Remove leading orphan tool messages."""
    while messages and messages[0].get("role") == "tool":
        messages.pop(0)
    return messages


def _has_tool_calls(msg) -> bool:
    """Whether the message is an assistant message with tool_calls."""
    return msg.get("role") == "assistant" and bool(msg.get("tool_calls"))


def group_turns(messages: list) -> list[list]:
    """Group messages into turns (the last message of each group is an assistant message with tool_calls)."""
    turns = []
    current = []
    for msg in messages:
        if _has_tool_calls(msg):
            if current:
                turns.append(current)
                current = []
            current.append(msg)
        else:
            current.append(msg)
    if current:
        turns.append(current)
    return turns


def _strip_orphan_tail(msgs: list) -> list:
    """Remove trailing orphan tool messages and dangling tool_calls."""
    while msgs and msgs[-1].get("role") == "tool":
        msgs.pop()
    while msgs and _has_tool_calls(msgs[-1]):
        last = msgs[-1]
        content = last.get("content") or "[tool calls without results were dropped]"
        msgs[-1] = {"role": "assistant", "content": content}
        while msgs and msgs[-1].get("role") == "tool":
            msgs.pop()
    return msgs


def _strip_orphan_head(msgs: list) -> list:
    """Remove leading orphan tool messages and unanswered tool_calls."""
    while msgs:
        if msgs[0].get("role") == "tool":
            msgs.pop(0)
            continue
        if _has_tool_calls(msgs[0]):
            tids = {tc["id"] for tc in msgs[0]["tool_calls"]}
            j = 1
            responded = set()
            while j < len(msgs) and msgs[j].get("role") == "tool":
                responded.add(msgs[j].get("tool_call_id"))
                j += 1
            if tids.issubset(responded):
                break
            msgs.pop(0)
            while msgs and msgs[0].get("role") == "tool":
                msgs.pop(0)
        else:
            break
    return msgs


def snip_compact(messages: list, min_keep_turns: int = 25) -> list:
    """L1 compaction: keep head 1 turn + tail N turns, replacing the middle with a placeholder."""
    turns = group_turns(messages)
    if len(turns) <= min_keep_turns + 1:
        return messages

    keep_head_turns = 1
    keep_tail_turns = min_keep_turns

    head = turns[:keep_head_turns]
    tail = turns[-keep_tail_turns:]

    head_flat = [m for t in head for m in t]
    tail_flat = [m for t in tail for m in t]

    head_flat = _strip_orphan_tail(head_flat)
    tail_flat = _strip_orphan_head(tail_flat)

    snipped_turns = len(turns) - keep_head_turns - keep_tail_turns
    print(f"snip_compact: kept head 1 + tail {keep_tail_turns}, "
          f"snipped {snipped_turns} turns (total {len(turns)})")
    result = (
        head_flat
        + [{"role": "user", "content": f"[snipped {snipped_turns} conversation turns]"}]
        + tail_flat
    )
    return ensure_valid_start(result)


def micro_compact(messages: list) -> list:
    """L2 compaction: replace old tool_result contents with placeholders, keeping the most recent N turns."""
    turns = group_turns(messages)
    turn_starts = []
    offset = 0
    for turn in turns:
        turn_starts.append(offset)
        offset += len(turn)
    tool_turns = []
    for ti, turn in enumerate(turns):
        tool_indices = [i for i, m in enumerate(turn) if m.get("role") == "tool"]
        if tool_indices:
            tool_turns.append((ti, tool_indices))
    if len(tool_turns) <= KEEP_RECENT_LOOP_TURN:
        return messages
    turns_to_compact = tool_turns[:-KEEP_RECENT_LOOP_TURN]
    compacted = 0
    for ti, tool_indices in turns_to_compact:
        base = turn_starts[ti]
        for li in tool_indices:
            gi = base + li
            msg = messages[gi]
            content = msg.get("content")
            if isinstance(content, str) and len(content) > 120:
                messages[gi] = {
                    **msg,
                    "content": "[Earlier tool result compacted. Re-run if needed. ]",
                }
                compacted += 1
    if compacted:
        print(
            f"micro_compact: compacted {compacted} tool_results across "
            f"{len(turns_to_compact)} turns (kept recent {KEEP_RECENT_LOOP_TURN} turns)"
        )
    return messages


def persist_large_output(tool_call_id: str, output: str) -> str:
    """L3 persistence: write overly long tool output to a file and return a placeholder summary."""
    if len(output) <= PERSIST_THRESHOLD:
        return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_call_id}.txt"
    if not path.exists():
        path.write_text(output)
    return (f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n"
            f"</persisted-output>")


def tool_result_budget(messages: list, max_bytes: int = 200_000) -> list:
    """Limit the total bytes of the most recent turn's tool output; persist when exceeded."""
    last_tc_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
            last_tc_idx = i
            break
    if last_tc_idx is None:
        return messages

    last_turn_tools = []
    for i in range(last_tc_idx + 1, len(messages)):
        if messages[i].get("role") == "tool":
            last_turn_tools.append(messages[i])
        elif messages[i].get("role") == "user":
            break
    if not last_turn_tools:
        return messages

    total = sum(len(msg.get("content") or "") for msg in last_turn_tools)
    if total <= max_bytes:
        return messages

    ranked = sorted(last_turn_tools, key=lambda m: len(m.get("content") or ""), reverse=True)
    print(f"tool_result_budget,last-turn bytes: {total}")
    for msg in ranked:
        total = sum(len(m.get("content") or "") for m in last_turn_tools)
        if total <= max_bytes:
            break
        content = msg.get("content") or ""
        if len(content) <= PERSIST_THRESHOLD:
            continue
        tid = msg.get("tool_call_id", "unknown")
        msg["content"] = persist_large_output(tid, content)
    return messages


def write_transcript(messages: list) -> Path:
    """Write the current message list to disk as a transcript."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    return path


def _messages_to_text(messages, max_chars: int = 80000) -> str:
    """Convert a message list to plain text (for summarization)."""
    lines = []
    total = 0
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(getattr(b, "text", "")) for b in content
                               if getattr(b, "type", None) == "text")
        if not isinstance(content, str):
            content = str(content)
        tcs = m.get("tool_calls") or []
        tc_info = ""
        if tcs:
            tc_info = " [tool_calls: " + ", ".join(
                tc.get("function", {}).get("name", "?") for tc in tcs) + "]"
        line = f"{role}:{tc_info} {content}"
        if total + len(line) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                lines.append(line[:remaining] + "...[truncated]")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def summarize_history(messages: list) -> str:
    """Call the LLM to summarize the conversation history."""
    conversation = _messages_to_text(messages, max_chars=80000)
    prompt = ("Summarize this coding-agent conversation so work can continue.\n"
              "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
              "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n"
              + conversation)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip() or "(empty summary)"
    except Exception as e:
        print(f"summarize_history error: {type(e).__name__}: {e}")
        return "(summary failed - using raw tail)"


def _build_post_compact_context(messages: list, budget_tokens: int = 6000) -> str:
    """Rebuild a context block after compaction: recent files / current plan / task board / teammates."""
    sections = []
    file_paths = []
    seen = set()
    for m in reversed(messages):
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            if fn.get("name") in ("read_file", "write_file", "edit_file"):
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                    p = args.get("path")
                    if p and p not in seen:
                        seen.add(p)
                        file_paths.append(p)
                except Exception:
                    pass
        if len(file_paths) >= 15:
            break
    if file_paths:
        sections.append("Recently accessed files:\n" + "\n".join(f"   - {p}" for p in file_paths))

    if ctx.current_todos:
        todo_lines = []
        for i, t in enumerate(ctx.current_todos):
            if t["status"] == "completed":
                todo_lines.append(f"  ✔️ #{i+1}: {t['content']}")
            else:
                todo_lines.append(f"   [{t['status']}] #{i+1}: {t['content']}")
        sections.append("Current plan:\n" + "\n".join(todo_lines))

    try:
        tasks = list_tasks()
        active = [t for t in tasks if t.status != "completed"]
        if active:
            task_lines = []
            for t in active:
                owner = f" [{t.owner}]" if t.owner else ""
                task_lines.append(f"  {t.id}: {t.subject} [{t.status}]{owner}")
            sections.append("Task board (active):\n" + "\n".join(task_lines))
    except Exception:
        pass

    teammates = []
    tm_seen = set()
    for name, info in ctx.teammate_registry.items():
        if name not in tm_seen:
            tm_seen.add(name)
            status = info.get("status", "running")
            teammates.append(f"   - {name} ({info.get('role', '')}) [{status}]")
    if not teammates:
        for m in reversed(messages):
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                if fn.get("name") == "spawn_teammate":
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                        name = args.get("name", "")
                        role = args.get("role", "")
                        if name and name not in tm_seen:
                            tm_seen.add(name)
                            teammates.append(f"  - {name} ({role})")
                    except Exception:
                        pass
    if teammates:
        sections.append("Spawned teammates:\n" + "\n".join(teammates))

    result = "\n\n".join(sections)
    estimated = estimate_tokens_messages([{"role": "user", "content": result}])
    while sections and estimated > budget_tokens:
        sections.pop()
        result = "\n\n".join(sections)
        estimated = estimate_tokens_messages([{"role": "user", "content": result}])
    return result


def compact_history(messages: list) -> list:
    """Auto compact: persist transcript + LLM summary + rebuild context."""
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    summary = summarize_history(messages)
    context_block = _build_post_compact_context(messages)
    compacted = [{"role": "user", "content": f"[Compacted]\n\n{summary}\n\n{context_block}"}]
    return compacted


def reactive_compact(messages: list) -> list:
    """Reactive compact: compaction triggered when the context exceeds the limit."""
    transcript = write_transcript(messages)
    summary = summarize_history(messages)
    context_block = _build_post_compact_context(messages)
    return [
        {"role": "user", "content": f"[Reactive compact]\n\n{summary}\n\n{context_block}"}
    ]
