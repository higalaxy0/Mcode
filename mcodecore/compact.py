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
    """Remove trailing orphan tool messages and dangling tool_calls.

    Two cases to clean (only at the *tail* boundary, which is where a cut can
    land mid-execution):

    1. *dangling tool_calls* -- an assistant message carrying ``tool_calls``
       whose results never arrived.  These are stripped (``tool_calls``
       removed, content replaced with a placeholder) so the LLM does not try
       to continue a tool call that will never get a result.

    2. *orphan tool messages* -- ``role == "tool"`` messages with no matching
       ``tool_call_id`` in any preceding assistant message.  These are
       removed.

    Crucially, a **valid** trailing ``assistant(tool_calls) -> tool(result)``
    pair is *preserved*: the tool result is the answer the agent just received
    and must not be dropped.
    """
    # 1) Strip trailing dangling tool_calls first.  A dangling tool_call is
    #    always the last message (the result has not arrived yet), so one
    #    pass suffices.  After stripping its tool_calls the assistant message
    #    becomes plain text and any orphan tool messages that followed it are
    #    exposed for step 2.
    while msgs and _has_tool_calls(msgs[-1]):
        last = msgs[-1]
        content = last.get("content") or "[tool calls without results were dropped]"
        msgs[-1] = {"role": "assistant", "content": content}

    # 2) Strip trailing *orphan* tool messages (no matching tool_call_id in
    #    the nearest preceding assistant tool_calls block).  A tool message
    #    that *does* match a preceding assistant block is a legitimate result
    #    and must be kept -- stop immediately.
    while msgs and msgs[-1].get("role") == "tool":
        tid = msgs[-1].get("tool_call_id")
        # Walk backwards to the nearest assistant-with-tool_calls message and
        # check whether it issued this tool_call_id.
        has_caller = False
        for j in range(len(msgs) - 2, -1, -1):
            if msgs[j].get("role") != "assistant":
                continue
            if _has_tool_calls(msgs[j]):
                tids = {tc["id"] for tc in msgs[j]["tool_calls"]}
                has_caller = tid in tids
            break  # nearest assistant block (tool_calls or plain text)
        if has_caller:
            break  # legitimate pair -- preserve everything from here on
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


def _is_snip_marker(msg) -> bool:
    """Whether the message is a stale snip placeholder inserted by a previous snip_compact run."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    return isinstance(content, str) and content.startswith("[snipped")


def _strip_snip_markers(messages: list) -> list:
    """Remove stale snip placeholders so they don't accumulate across repeated snip runs.

    A previous snip inserts a ``[snipped N conversation turns]`` user message as the
    boundary between head and tail. On the next snip the group logic merges this marker
    into the first turn (since it carries no tool_calls), so the old marker survives into
    the new head and stacks up one per run. Stripping them up front keeps head stable.
    Returns the original list unchanged when there is nothing to strip (preserves identity
    for the no-op fast path).
    """
    if not any(_is_snip_marker(m) for m in messages):
        return messages
    return [m for m in messages if not _is_snip_marker(m)]


def _is_task_anchor(msg) -> bool:
    """Whether *msg* is a genuine user task prompt worth pinning across snips.

    Excludes synthetic/injected user messages (system-generated placeholders that
    start with ``[``) and the interruption sentinel, so only real task inputs
    survive as pinned anchors.
    """
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        return False
    if content.startswith("["):   # [Inbox] / [snipped ...] / [Compacted] / [Reactive compact] ...
        return False
    if content == "interrupted by user":
        return False
    return True


def _build_snipped_activity_summary(messages: list) -> str:
    """Build a lightweight summary of the snipped-away region (no LLM needed).

    Used as a fallback when ``_build_post_compact_context`` returns an empty
    block (no known files / todos / tasks / teammates), so the placeholder
    still carries *what kind of work* was done in the dropped turns rather
    than being a bare ``[snipped N turns]``.

    Extracts:
      * tool-call type distribution (e.g. "25× edit_file, 3× bash")
      * last assistant text message in the region (if any)
      * truncated user prompts in the region
    """
    tool_names: dict[str, int] = {}
    last_asst_text = ""
    user_snippets: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            for tc in tcs:
                name = (tc.get("function", {}) or {}).get("name", "?")
                tool_names[name] = tool_names.get(name, 0) + 1
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                last_asst_text = content.strip()
        elif role == "user" and _is_task_anchor(m):
            text = (m.get("content") or "").strip()
            if text and not text.startswith("["):
                user_snippets.append(text[:80])
    lines = []
    if tool_names:
        stats = ", ".join(f"{c}× {n}" for n, c in tool_names.items())
        lines.append(f"Snipped activity: {stats}")
    if user_snippets:
        shown = user_snippets[:5]
        more = f" (+{len(user_snippets)-5} more)" if len(user_snippets) > 5 else ""
        lines.append("Snipped task prompts: " + " | ".join(shown) + more)
    if last_asst_text:
        lines.append(f"Last note before snip: {last_asst_text[:200]}")
    return "\n".join(lines)


def snip_compact(messages: list, min_keep_turns: int = 50) -> list:
    """L1 sliding-window compaction.

    Keeps every genuine user task prompt (pinned, in order) so the agent never
    forgets what tasks were issued; keeps the most recent ``min_keep_turns``
    turns as the active working window; and replaces everything in between with
    a single placeholder carrying a rebuilt context block (recent files /
    todolist / task board / teammates).

    Layered window relationship (OPT-4 scheme B):
        snip_compact keeps ``min_keep_turns`` (default 50) turns of *recent
        conversational context* -- including assistant reasoning, tool_calls
        and their results -- so the agent can still look back at recent
        exchanges.  However only the most recent ``KEEP_RECENT_LOOP_TURN``
        (default 25) turns retain *full tool_result contents*; the older half
        of the snip window has its bulky tool outputs replaced with
        placeholders by :func:`micro_compact` (L2).  The two constants are
        intentionally *decoupled*: ``min_keep_turns`` governs the breadth of
        the active window (how much history is structurally retained), while
        ``KEEP_RECENT_LOOP_TURN`` governs the depth of verbatim data within
        that window (how much tool output survives unmodified).
    """
    messages = _strip_snip_markers(messages)
    turns = group_turns(messages)
    if len(turns) <= min_keep_turns + 1:
        return messages

    keep_tail_turns = min_keep_turns
    tail = turns[-keep_tail_turns:]
    # index of the first message belonging to the tail window
    tail_start = sum(len(t) for t in turns[:-keep_tail_turns])

    # 1) pinned: genuine user task prompts *before* the tail window (task
    #    identity).  Capped so that long multi-task sessions don't let the
    #    pinned block grow without bound: when the cap is exceeded the oldest
    #    prompts are collapsed into a single terse summary line.
    PIN_CAP = 10
    pinned = [m for m in messages[:tail_start] if _is_task_anchor(m)]
    if len(pinned) > PIN_CAP:
        old = pinned[:-PIN_CAP]
        recent = pinned[-PIN_CAP:]
        old_summary = "; ".join((p.get("content") or "")[:60] for p in old)
        pinned = [{"role": "user",
                   "content": f"[Earlier tasks: {old_summary}]"}] + recent

    # 2) rebuilt context block (no LLM): recent files / todos / tasks /
    #    teammates.  When all three are empty (e.g. a pure bash session with
    #    no file ops / todos / tasks), fall back to a lightweight activity
    #    summary extracted from the snipped-away region so the placeholder
    #    still tells the agent *what kind of work* was done.
    snipped_turns = len(turns) - keep_tail_turns
    context_block = _build_post_compact_context(messages)
    # The activity summary (tool distribution / last note / task prompts in
    # the dropped region) is *complementary* to the context block (which only
    # carries recent-files / todos / tasks / teammates), so always generate it
    # and append rather than treating the two as mutually exclusive.
    activity = _build_snipped_activity_summary(messages[:tail_start])
    if activity:
        context_block = (context_block + "\n\n" + activity) if context_block else activity
    placeholder_content = f"[snipped {snipped_turns} conversation turns]"
    if context_block:
        placeholder_content += f"\n{context_block}"

    # 3) tail sliding window (active working area), orphan-cleaned at both
    #    ends: head (unanswered tool_calls / orphan tool msgs from the cut
    #    boundary) and tail (dangling tool_calls when the conversation was
    #    interrupted mid-execution).
    tail_flat = _strip_orphan_tail(_strip_orphan_head([m for t in tail for m in t]))

    print(f"snip_compact: pinned {len(pinned)} task prompts, kept tail "
          f"{keep_tail_turns} turns, snipped {snipped_turns} turns (total {len(turns)})")
    result = pinned + [{"role": "user", "content": placeholder_content}] + tail_flat
    return ensure_valid_start(result)


def micro_compact(messages: list) -> list:
    """L2 compaction: replace old tool_result contents with placeholders, keeping the most recent N turns.

    Layered window relationship (OPT-4 scheme B):
        Only the most recent ``KEEP_RECENT_LOOP_TURN`` (default 25) turns keep
        their full tool_result contents.  Older turns within the active window
        have tool outputs replaced with lightweight placeholders, trimming bulk
        while preserving the turn structure (assistant reasoning + tool_calls).
        This is intentionally smaller than snip_compact's ``min_keep_turns``
        (default 50): snip governs *breadth* of the retained window, micro
        governs *depth* of verbatim tool data within that window.  Together
        they form a two-tier active area: the inner 25 turns are fully
        detailed, the outer 25 turns are structurally present but data-light.
    """
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
