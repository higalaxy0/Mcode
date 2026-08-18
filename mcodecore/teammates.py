"""Threaded teammate agents."""

from __future__ import annotations

import json
import platform
import threading
import time
from pathlib import Path

from .config import (LLM_MODEL, WORKDIR, TEAM_HISTORY_DIR, client, TURN_BUDGET,
                     TURN_BUDGET_RENEWAL, TURN_BUDGET_HARD_CAP,
                     CLAIM_MIN_TURNS, CONTEXT_LIMIT, debug)
from .context import ctx
from .hooks import trigger_hooks
from .bus import idle_poll, _teammate_submit_plan, has_pending_plan_approval
from .memory import read_memory_index, _load_memories_async, _await_memories
from .streaming import classify_transient, retry_after_seconds, backoff_delay
from .tasks import list_tasks, claim_task, complete_task, list_owned_inprogress, release_task
from .utils import truncate, sanitize_message, sanitize_messages, parse_tool_args
from .compact import (estimate_tokens_messages, compact_history, reactive_compact,
                      tool_result_budget, snip_compact, micro_compact)


def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    """Spawn a background threaded teammate agent."""
    if name in ctx.active_teammates:
        return f"Teammate '{name}' already exists"
    system = (f"You are '{name}',a {role} at {WORKDIR}. "
              f"Use tools to complete tasks. "
              f"You can list and claim tasks from the board. "
              f"Check inbox for protocol messages. "
              f"Send results via send_message to 'lead'. "
              f"When ALL your tasks are done and results sent, call the finish tool to exit cleanly. "
              f"OS is {platform.system()},ensure commands and file paths compatible with {platform.system()}.")
    _mem_index = read_memory_index()
    if _mem_index:
        system += f"\n\nMemories available:\n{_mem_index}"
    _TEAM_HISTORY_ENABLED = True
    # NOTE: history dir is the module-level TEAM_HISTORY_DIR imported from
    # config (session-scoped); no local override needed.  Teammates from
    # different mcode windows in the same folder write to disjoint files.
    _team_history_lock = threading.Lock()

    def log_team_history(teammate_name: str, event: str,
                         detail: dict, history_dir: Path) -> None:
        if not _TEAM_HISTORY_ENABLED:
            return
        history_dir.mkdir(exist_ok=True)
        record = {"ts": time.time(), "event": event, "detail": detail}
        log_file = history_dir / f"{teammate_name}.jsonl"
        try:
            with _team_history_lock:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  \033[33m[history] {teammate_name} "
                  f"log failed ({event}):{e}\033[0m")

    def handle_inbox_message(name: str, msg: dict, messages: list) -> bool:
        msg_type = msg.get("type", "message")
        meta = msg.get("metadata", {})
        req_id = meta.get("request_id", "")

        if msg_type == "shutdown_request":
            ctx.bus.send(name, "lead", "Shutting down gracefully.",
                         "shutdown_response",
                         {"request_id": req_id, "approve": True})
            print(f"  \033[35m[protocol] {name} approved shutdown "
                  f"({req_id})\033[0m")
            log_team_history(name, "shutdown_request",
                             {"request_id": req_id}, TEAM_HISTORY_DIR)
            return True

        if msg_type == "plan_approval_response":
            approve = meta.get("approve", False)
            if approve:
                messages.append({"role": "user",
                                 "content": f"[Plan approved] Proceed with the task."})
            else:
                messages.append({"role": "user",
                                 "content": f"[Plan rejected] Feedback: {msg['content']}"})
            log_team_history(name, "plan_response",
                             {"request_id": req_id, "approved": approve,
                              "feedback": truncate(msg.get('content', ''))},
                             TEAM_HISTORY_DIR)
        return False
    MAX_REACTIVE_RETRIES = 3
    def run():
        try:
            idle_result = None
            messages = [{"role": "user", "content": f"{prompt}.<identity>You are '{name}', role: {role}. Continue your work.</identity>"}]
            _mem_holder = _load_memories_async(messages)
            sub_tools = [
                {"type": "function", "function": {"name": "bash", "description": f"Run a shell command.NOTE:The current OS is {platform.system()}.Ensure commands are valid for this environment.Timeout:300s default;append '# timeout=600' for longer.Background:prefix with 'bg: ' to run long tasks (monitors,servers) in background - returns PID + log path;use read_file to check output later.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
                {"type": "function", "function": {"name": "read_file", "description": "Read file contents. Returns lines with line-number prefixes. Use offset to start from a specific line and limit to control how many lines to read.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path relative to workspace root."}, "offset": {"type": "integer", "description": "Line number to start reading from (1-based). Defaults to 1."}, "limit": {"type": "integer", "description": "Maximum number of lines to read. Defaults to 2000."}}, "required": ["path"]}}},
                {"type": "function", "function": {"name": "write_file", "description": "Write content to a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
                {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
                {"type": "function", "function": {"name": "glob", "description": "Find files matching a glob pattern. Supports `**` for recursive directory matching (e.g. `**/*.py`). Returns paths relative to workspace root using forward slashes, sorted by modification time (newest first), limited to 100 results.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Glob pattern, e.g. `**/*.py` or `src/**/*.ts`."}}, "required": ["pattern"]}}},
                {"type": "function", "function": {"name": "grep", "description": "Search file contents for a regex pattern. Returns matching lines with file:line prefixes. Searches recursively under the given path (relative to workdir).", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Regular expression to search for."}, "path": {"type": "string", "description": "File or directory to search in (relative to workdir). Defaults to '.'."}, "include": {"type": "string", "description": "Glob pattern to filter filenames, e.g. '*.py'."}}, "required": ["pattern"]}}},
                {"type": "function", "function": {"name": "send_message", "description": "Send a message to another agent.", "parameters": {"type": "object", "properties": {"to": {"type": "string", "description": "The identifier or name of the target agent."}, "content": {"type": "string", "description": "The message content to send."}}, "required": ["to", "content"]}}},
                {"type": "function", "function": {"name": "submit_plan", "description": "Submit a plan for Lead approval. You must own the task (task_id must be a task you have claimed).", "parameters": {"type": "object", "properties": {"plan": {"type": "string"}, "task_id": {"type": "string", "description": "The ID of the task this plan is for. Must be a task you have claimed."}}, "required": ["plan", "task_id"]}}},
                {"type": "function", "function": {"name": "list_tasks", "description": "List all tasks with status, owner, and dependencies.", "parameters": {"type": "object", "properties": {"include_completed": {"type": "boolean", "description": "If true,include completed tasks in the listing."}}, "required": ["include_completed"]}}},
                {"type": "function", "function": {"name": "claim_task", "description": "Claim a pending task.", "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}}},
                {"type": "function", "function": {"name": "complete_task", "description": "Mark an in-progress task as completed.", "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}}},
                {"type": "function", "function": {"name": "finish", "description": "Signal that your work is complete. Call this when you have finished all your tasks and sent your results to lead. This stops the agent immediately - do NOT call this if you still have work to do.", "parameters": {"type": "object", "properties": {"summary": {"type": "string", "description": "A brief summary of what you accomplished."}}, "required": ["summary"]}}}
            ]

            def _run_list_tasks(include_completed: bool = True):
                if isinstance(include_completed, str):
                    include_completed = include_completed.strip().lower() == "true"
                include_completed = bool(include_completed)
                tasks = list_tasks()
                if not tasks:
                    return "No tasks."
                return "\n".join(
                    f"  {t.id}: {t.subject} [{t.status}]"
                    for t in tasks)

            # Turn-budget tracked loop (Fix #1A+C):
            #   - turns_used[0] counts turns in the current budget window.
            #   - turns_total[0] counts all turns (including renewals).
            #   - Mutable containers so the _run_claim_task closure can
            #     read the live turn count (Fix #1C: refuse to claim when
            #     remaining turns < CLAIM_MIN_TURNS).
            turns_used = [0]
            turns_total = [0]
            # Flag set by the finish tool handler so the main loop knows
            # to exit immediately after the current turn (Fix #1).
            _finished = [False]
            # Track whether the teammate has done any real work (called
            # at least one tool).  Used by the natural-stop heuristic
            # (Fix #1): only exit early if work was actually done.  On
            # the first turn with no tool calls, fall through to
            # idle_poll so auto-claim can pick up available tasks.
            _did_work = [False]

            def _run_finish(summary: str) -> str:
                """Handler for the finish tool: send result, request exit.

                Sends the summary as a result message to lead immediately
                and sets ``_finished[0]`` so the main loop breaks out
                without entering ``idle_poll`` (which would block up to
                360 s waiting for more work).  (Fix #1)
                """
                ctx.bus.send(name, "lead", summary, "result")
                _finished[0] = True
                return f"Finish acknowledged. Result delivered. Shutting down."

            def _run_claim_task(task_id: str):
                # Fix #1C: refuse to claim if remaining turn budget
                # is below CLAIM_MIN_TURNS - prevents claiming tasks
                # that cannot be completed before exhaustion.
                remaining = TURN_BUDGET - turns_used[0]
                if remaining < CLAIM_MIN_TURNS:
                    return (f"Cannot claim {task_id}: only {remaining} "
                            f"turns left in budget (minimum "
                            f"{CLAIM_MIN_TURNS} required). Complete "
                            f"current work first.")
                return claim_task(task_id, owner=name)

            def _run_complete_task(task_id: str):
                return complete_task(task_id, owner=name)

            log_team_history(name, "spawned",
                             {"role": role}, TEAM_HISTORY_DIR)

            def _heartbeat(phase: str = "working") -> None:
                """Update registry liveness info (thread-safe via memory_lock).

                Called after each turn, at idle_poll entry, and when
                blocked on plan approval.  Lets the lead's
                ``teammate_status`` tool distinguish "still working"
                from "dead" so it does not panic on empty inboxes.
                """
                with ctx.memory_lock:
                    if name in ctx.teammate_registry:
                        ctx.teammate_registry[name]["last_heartbeat"] = time.time()
                        ctx.teammate_registry[name]["phase"] = phase
                        ctx.teammate_registry[name]["turns_total"] = turns_total[0]

            def _run_send_message(to, content):
                ctx.bus.send(name, to, content)
                return "Sent"

            def _run_submit_plan(plan, task_id=""):
                return _teammate_submit_plan(name, plan, task_id=task_id)

            # Reference module-level fsops handlers
            from .fsops import run_bash, run_read, run_write, run_edit, run_glob, run_grep
            sub_handlers = {
                "bash": run_bash, "read_file": run_read, "write_file": run_write,
                "edit_file": run_edit, "glob": run_glob, "grep": run_grep,
                "send_message": _run_send_message,
                "submit_plan": _run_submit_plan,
                "list_tasks": _run_list_tasks,
                "claim_task": _run_claim_task,
                "complete_task": _run_complete_task,
                "finish": _run_finish,
            }

            # Inject MCP tools if any server is connected
            _mcp = ctx.mcp
            if _mcp.is_connected:
                _mcp_schemas = _mcp.list_all_tool_schemas()
                _existing_names = {t["function"]["name"] for t in sub_tools}
                _mcp_handlers = _mcp.build_handlers()
                for _schema in _mcp_schemas:
                    _mname = _schema["function"]["name"]
                    if _mname in _existing_names:
                        continue
                    sub_tools.append(_schema)
                    sub_handlers[_mname] = _mcp_handlers[_mname]

            while True:
                should_shutdown = False
                non_protocol = []
                reactive_retries = 0
                while turns_used[0] < TURN_BUDGET:
                    inbox = ctx.bus.read_inbox(name)
                    for msg in inbox:
                        stopped = handle_inbox_message(name, msg, messages)
                        if stopped:
                            should_shutdown = True
                            break
                    if should_shutdown:
                        break
                    if inbox and not should_shutdown:
                        # Surface all non-protocol messages to the LLM, not
                        # just type=="message".  Previously types like
                        # "result", "crashed", "LLM API error" were silently
                        # dropped. (Bug F-6 fix)
                        _protocol_types = {"shutdown_request",
                                           "plan_approval_response"}
                        non_protocol = [m for m in inbox
                                        if m.get("type") not in _protocol_types]
                        if non_protocol:
                            messages.append({"role": "user",
                                             "content": f"<inbox>{json.dumps(non_protocol)}</inbox>"})
                        log_team_history(name, "inbox_received",
                                         [{"from": m.get("from"), "type": m.get("type"),
                                           "content": truncate(m.get("content", ""))}
                                          for m in inbox], TEAM_HISTORY_DIR)
                    try:
                        # Progressive compaction pipeline (same 4-tier
                        # architecture as the lead agent):
                        #   L3: tool_result_budget - persist oversized tool
                        #       outputs from the last turn to disk;
                        #   L1: snip_compact - sliding-window, pin task prompts,
                        #       keep recent 50 turns;
                        #   L2: micro_compact - replace old tool_result contents
                        #       with placeholders (keep recent 25 turns full);
                        #   auto: compact_history - LLM summary when token
                        #         estimate exceeds CONTEXT_LIMIT.
                        # Without L1-L3, long-running teamagents accumulate
                        # every tool call's full output and overflow the context
                        # window far sooner than necessary.
                        import copy as _copy
                        api_messages = _copy.deepcopy(messages)
                        api_messages[:] = tool_result_budget(api_messages)
                        api_messages[:] = snip_compact(api_messages)
                        api_messages[:] = micro_compact(api_messages)
                        if estimate_tokens_messages(api_messages) > CONTEXT_LIMIT:
                            debug(f"[teammate {name} auto compact, "
                                  f"context size = "
                                  f"{estimate_tokens_messages(api_messages)}]")
                            api_messages[:] = compact_history(api_messages)
                        messages[:] = _copy.deepcopy(api_messages)
                        request_messages = list(api_messages)
                        # Inject the system prompt at position 0 *after*
                        # compaction, matching the lead agent pattern.
                        # Previously system was in messages and got swept
                        # away by compact_history / reactive_compact.
                        request_messages.insert(0, {"role": "system", "content": system})
                        # Inject memories into the last user message (same as
                        # the lead agent).  Done *after* compaction so the
                        # memory content is not lost when compaction rebuilds
                        # the message list.
                        memories_content = _await_memories(_mem_holder)
                        if memories_content:
                            for mi in range(len(request_messages) - 1, -1, -1):
                                if (request_messages[mi].get("role") == "user"
                                        and isinstance(request_messages[mi].get("content"), str)):
                                    request_messages[mi] = {
                                        **request_messages[mi],
                                        "content": memories_content + "\n\n" + request_messages[mi]["content"],
                                    }
                                    break
                        response = client.chat.completions.create(
                            model=LLM_MODEL,
                            messages=sanitize_messages(request_messages),
                            tools=sub_tools,
                            tool_choice="auto",
                            temperature=0.7,
                            max_tokens=16384,
                            timeout=200,
                            extra_body={
                                "chat_template_kwargs": {
                                    "enable_thinking": False}
                            }
                        )
                        reactive_retries = 0
                    except Exception as e:
                        _e_str = str(e).lower()
                        if classify_transient(e):
                            if reactive_retries < MAX_REACTIVE_RETRIES:
                                delay = retry_after_seconds(e)
                                if delay is None:
                                    delay = backoff_delay(reactive_retries)
                                debug(f"[teammate {name} retry {reactive_retries + 1}/{MAX_REACTIVE_RETRIES} "
                                      f"after {delay:.1f}s: {type(e).__name__}: {str(e)[:200]}]")
                                time.sleep(delay)
                                reactive_retries += 1
                                continue
                        # prompt_too_long: reactive compact and retry
                        if (("prompt_too_long" in _e_str
                             or "too many tokens" in _e_str)
                                and reactive_retries < MAX_REACTIVE_RETRIES):
                            debug(f"[teammate {name} reactive compact]")
                            messages[:] = reactive_compact(messages)
                            reactive_retries += 1
                            continue
                        log_team_history(name, "llm_failed",
                                         {"error_type": type(e).__name__,
                                          "error": truncate(str(e))}, TEAM_HISTORY_DIR)
                        ctx.bus.send(name, "lead", f"[teammate] {name} LLM call failed: "f"{type(e).__name__}: {e}", "LLM API error")
                        return
                    # Guard against empty choices (content-filter, rate-limit soft-fail)
                    if not response.choices:
                        messages.append({"role": "assistant", "content": ""})
                        break
                    assistant_message = response.choices[0].message
                    log_team_history(name, "llm_response",
                                     {
                                         "content": truncate(assistant_message.content or ""),
                                         "tool_calls": truncate(json.dumps(
                                             [tc.model_dump(exclude_none=True)
                                              for tc in (assistant_message.tool_calls or [])],
                                             ensure_ascii=False)),
                                         "finish_reason": response.choices[0].finish_reason,
                                     }, TEAM_HISTORY_DIR)
                    messages.append(sanitize_message(assistant_message.model_dump(exclude_none=True)))
                    if response.choices[0].finish_reason != "tool_calls":
                        break
                    for tool_call in (assistant_message.tool_calls or []):
                        if response.choices[0].finish_reason == "tool_calls":
                            # PreToolUse hook: wrap in try/except so a buggy
                            # hook does not kill the teammate (graceful skip).
                            try:
                                blocked = trigger_hooks("PreToolUse", tool_call.function)
                            except Exception as _hook_err:
                                debug(f"[teammate] {name} PreToolUse hook error: {_hook_err}")
                                blocked = None
                            if blocked:
                                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(blocked)})
                                continue
                            handler = sub_handlers.get(tool_call.function.name)
                            args = parse_tool_args(tool_call.function.arguments)
                            # Mirror the lead agent's error handling: any
                            # tool exception is surfaced to the LLM as an
                            # error string, NOT propagated to the outer
                            # except (which would report "crashed" and
                            # kill the teammate). (Fix #5)
                            try:
                                output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
                            except Exception as _tool_err:
                                output = f"Error: {_tool_err}"
                            # PostToolUse hook: wrap in try/except (tool already ran,
                            # result is kept even if hook crashes).
                            try:
                                trigger_hooks("PostToolUse", tool_call.function, output)
                            except Exception as _hook_err:
                                debug(f"[teammate] {name} PostToolUse hook error: {_hook_err}")
                            log_team_history(name, "tool_called",
                                             {"tool": tool_call.function.name,
                                              "args": truncate(json.dumps(args, ensure_ascii=False)),
                                              "output": truncate(output)}, TEAM_HISTORY_DIR)
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
                    # One LLM call consumed one turn (Fix #1A)
                    turns_used[0] += 1
                    turns_total[0] += 1
                    _heartbeat("working")

                    # Fix #1: Mark that the teammate has done real work
                    # (at least one tool call this turn).  Used by the
                    # natural-stop heuristic to distinguish "started and
                    # finished" from "hasn't started yet".
                    if assistant_message.tool_calls:
                        _did_work[0] = True

                if should_shutdown:
                    break

                # Fix #1: If the teammate called the finish tool, exit
                # immediately without entering idle_poll.  The result was
                # already sent by _run_finish.
                if _finished[0]:
                    break

                # Fix #1: If the LLM stopped naturally (no tool_calls,
                # finish_reason != "tool_calls") AND has already done
                # some work this session, AND there are no owned
                # in_progress tasks, the work is likely done.  Entering
                # idle_poll here would block up to 360 s for nothing.
                # Skip idle_poll and deliver the result immediately.
                # Exception 1: if there's a pending plan approval (the
                # teammate called submit_plan and is waiting), we must
                # stay alive in idle_poll to receive the response.
                # Exception 2: if the teammate hasn't done any work yet
                # (first turn, no tool calls), fall through to idle_poll
                # so auto-claim can pick up available tasks.
                _natural_stop = (turns_used[0] < TURN_BUDGET)
                if _natural_stop and _did_work[0]:
                    owned = list_owned_inprogress(name)
                    if not owned:
                        # Check for pending plan approval from this teammate
                        if not has_pending_plan_approval(name):
                            # Race guard: a protocol response (e.g. plan
                            # approval) may already be sitting in our
                            # mailbox even though pending_requests no
                            # longer marks it pending (the lead flipped
                            # the state when it reviewed).  Exiting now
                            # would lose that message forever.  Peek
                            # (non-destructive) and stay alive instead.
                            if not ctx.bus.peek_inbox(name):
                                debug(f"[teammate] {name} natural stop, "
                                      f"no owned tasks, exiting")
                                break

                # Turn-budget renewal (Fix #1A): if we exhausted the soft
                # cap but still own in_progress tasks, grant more turns
                # (up to the hard cap).  This prevents task abandonment
                # when turns were wasted polling for approvals/deps.
                if turns_total[0] < TURN_BUDGET_HARD_CAP:
                    owned = list_owned_inprogress(name)
                    if owned:
                        renewal = min(TURN_BUDGET_RENEWAL,
                                      TURN_BUDGET_HARD_CAP - turns_total[0])
                        turns_used[0] = max(0, TURN_BUDGET - renewal)
                        log_team_history(name, "turn_renewal",
                                         {"turns_total": turns_total[0],
                                          "renewed": renewal,
                                          "owned_tasks": len(owned)},
                                         TEAM_HISTORY_DIR)
                        continue  # re-enter inner while with renewed budget

                _heartbeat("idle_poll")
                idle_result = idle_poll(name, messages, name, role)
                if idle_result == "shutdown":
                    break
                if idle_result == "timeout":
                    break
                # Fix #9: idle_poll returned "work" (a lead instruction or
                # dependency-ready notification arrived).  The inner loop
                # exited because turns_used was exhausted.  If we don't
                # reset the budget here, the renewed budget check above
                # only triggers for *owned* tasks - a lead instruction
                # with no owned tasks would be silently lost (loop back,
                # turns_used still >= TURN_BUDGET, idle_poll on empty
                # inbox -> timeout).  Reset the soft budget (capped by
                # the hard cap) so the teammate can act on the new work.
                if idle_result == "work" and turns_total[0] < TURN_BUDGET_HARD_CAP:
                    renewal = min(TURN_BUDGET_RENEWAL,
                                  TURN_BUDGET_HARD_CAP - turns_total[0])
                    turns_used[0] = max(0, TURN_BUDGET - renewal)
                    log_team_history(name, "idle_work_renewal",
                                     {"turns_total": turns_total[0],
                                      "renewed": renewal},
                                     TEAM_HISTORY_DIR)
                # Guard against infinite loop when the hard cap is reached
                # but idle_poll still returns "work" (e.g. the worker owns
                # in_progress tasks).  Without renewal budget left, looping
                # back would just call idle_poll again forever.  Force-exit
                # with a clear message so the lead is notified. (Fix #4)
                if turns_total[0] >= TURN_BUDGET_HARD_CAP:
                    messages.append({"role": "user",
                                     "content": "[turn budget exhausted] "
                                                "Hard cap reached; stopping."})
                    break

            # Extract the final result: search backwards for the last
            # assistant message with non-empty content.  We must NOT use
            # messages[-1] blindly because it could be a tool message
            # (role:"tool") whose content is raw tool output, not the
            # agent's summary. (Bug D fix)
            result = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    result = msg["content"]
                    break
            if not result:
                result = (f"teamagent stopped after {turns_total[0]} turns "
                          f"without final answer.")
            ctx.bus.send(name, "lead", result, "result")
        except Exception as exc:
            # Outer safety net: any uncaught exception (tool crash,
            # parse failure, idle_poll I/O error, etc.) sends a
            # 'crashed' notification so the lead is never left in
            # the dark.  The finally block still sets the event.
            import traceback as _tb
            _err_type = type(exc).__name__
            _err_msg = str(exc)
            _detail = f"{_err_type}: {_err_msg}"
            try:
                ctx.bus.send(name, "lead", _detail, "crashed")
            except Exception:
                pass  # bus itself broken; event channel still works
            print(f"  \033[31m[teammate] {name} crashed: "
                  f"{_detail}\033[0m")
            _final_crash = _detail
        finally:
            # CRITICAL: set event + update registry BEFORE logging.
            # A logging failure must never prevent event notification.
            evt = ctx.active_teammates.get(name)
            if evt is not None:
                evt.set()
            if name in ctx.teammate_registry:
                ctx.teammate_registry[name]["status"] = "finished"
            # Release any in_progress tasks this worker still owns so
            # they become available for other workers to claim.  Without
            # this, tasks are orphaned when a worker exits (timeout,
            # crash, or shutdown). (Fix #3)
            try:
                orphans = list_owned_inprogress(name)
                for t in orphans:
                    release_task(t["id"], owner=name)
                    if orphans:
                        ctx.bus.send(name, "lead",
                                     f"Released orphaned task "
                                     f"{t['id']}: {t['subject']}",
                                     "message")
            except Exception as e:
                print(f"  \033[33m[teammate] {name} release error: "
                      f"{e}\033[0m")
            # Now attempt logging (already wrapped internally, but
            # guard again for safety).
            try:
                try:
                    _final
                except NameError:
                    try:
                        _final_crash
                    except NameError:
                        _final_crash = ""
                    _final = _final_crash
                log_team_history(name, "finished",
                                 {"result": truncate(_final),
                                  "idle_result": idle_result},
                                 TEAM_HISTORY_DIR)
            except Exception:
                pass
            print(f"  \033[32m[teammate] {name} finished\033[0m")

    ctx.active_teammates[name] = threading.Event()
    ctx.teammate_registry[name] = {
        "role": role,
        "spawned_at": time.time(),
        "status": "running",
        "phase": "starting",
        "last_heartbeat": time.time(),
        "turns_total": 0,
    }
    threading.Thread(target=run, daemon=True).start()
    print(f"  \033[36m[teammate] {name} spawned as {role}\033[0m")
    return (f"Teammate '{name}' spawned as {role}. "
            f"The teamagent runs in the background. "
            f"You do not need to poll - its results will be delivered to you "
            f"automatically when it finishes or needs your input. "
            f"Use teammate_status to check its liveness/progress.")
