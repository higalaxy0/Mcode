"""Threaded teammate agents."""

from __future__ import annotations

import json
import platform
import threading
import time
from pathlib import Path

from .config import LLM_MODEL, WORKDIR, client
from .context import ctx
from .hooks import trigger_hooks
from .bus import idle_poll, _teammate_submit_plan
from .memory import read_memory_index, _load_memories_async, _await_memories
from .tasks import list_tasks, claim_task, complete_task
from .utils import truncate, sanitize_message, sanitize_messages, parse_tool_args


def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    """Spawn a background threaded teammate agent."""
    if name in ctx.active_teammates:
        return f"Teammate '{name}' already exists"
    system = (f"You are '{name}',a {role} at {WORKDIR}. "
              f"Use tools to complete tasks. "
              f"You can list and claim tasks from the board. "
              f"Check inbox for protocol messages. "
              f"Send results via send_message to 'lead'. "
              f"OS is {platform.system()},ensure commands and file paths compatible with {platform.system()}.")
    _mem_index = read_memory_index()
    if _mem_index:
        system += f"\n\nMemories available:\n{_mem_index}"
    _TEAM_HISTORY_ENABLED = True
    TEAM_HISTORY_DIR = WORKDIR / ".team_history"
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
            messages.append({"role": "system", "content": system})
            _mem_holder = _load_memories_async(messages)
            sub_tools = [
                {"type": "function", "function": {"name": "bash", "description": f"Run a shell command.NOTE:The current OS is {platform.system()}.Ensure commands are valid for this environment.Timeout:300s default;append '# timeout=600' for longer.Background:prefix with 'bg: ' to run long tasks (monitors,servers) in background - returns PID + log path;use read_file to check output later.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
                {"type": "function", "function": {"name": "read_file", "description": "Read file contents. Returns lines with line-number prefixes. Use offset to start from a specific line and limit to control how many lines to read.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path relative to workspace root."}, "offset": {"type": "integer", "description": "Line number to start reading from (1-based). Defaults to 1."}, "limit": {"type": "integer", "description": "Maximum number of lines to read. Defaults to 2000."}}, "required": ["path"]}}},
                {"type": "function", "function": {"name": "write_file", "description": "Write content to a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
                {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
                {"type": "function", "function": {"name": "glob", "description": "Find files matching a glob pattern. Supports `**` for recursive directory matching (e.g. `**/*.py`). Returns paths relative to workspace root using forward slashes, sorted by modification time (newest first), limited to 100 results.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Glob pattern, e.g. `**/*.py` or `src/**/*.ts`."}}, "required": ["pattern"]}}},
                {"type": "function", "function": {"name": "grep", "description": "Search file contents for a regex pattern. Returns matching lines with file:line prefixes. Searches recursively under the given path (relative to workdir).", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Regular expression to search for."}, "path": {"type": "string", "description": "File or directory to search in (relative to workdir). Defaults to '.'."}, "include": {"type": "string", "description": "Glob pattern to filter filenames, e.g. '*.py'."}}, "required": ["pattern"]}}},
                {"type": "function", "function": {"name": "send_message", "description": "Send a message to another agent.", "parameters": {"type": "object", "properties": {"to": {"type": "string", "description": "The identifier or name of the target agent."}, "content": {"type": "string", "description": "The message content to send."}}, "required": ["to", "content"]}}},
                {"type": "function", "function": {"name": "submit_plan", "description": "Submit a plan for Lead approval.", "parameters": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}}},
                {"type": "function", "function": {"name": "list_tasks", "description": "List all tasks with status, owner, and dependencies.", "parameters": {"type": "object", "properties": {"include_completed": {"type": "boolean", "description": "If true,include completed tasks in the listing."}}, "required": ["include_completed"]}}},
                {"type": "function", "function": {"name": "claim_task", "description": "Claim a pending task.", "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}}},
                {"type": "function", "function": {"name": "complete_task", "description": "Mark an in-progress task as completed.", "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}}}
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

            def _run_claim_task(task_id: str):
                return claim_task(task_id, owner=name)

            def _run_complete_task(task_id: str):
                return complete_task(task_id)

            log_team_history(name, "spawned",
                             {"role": role}, TEAM_HISTORY_DIR)

            def _run_send_message(to, content):
                ctx.bus.send(name, to, content)
                return "Sent"

            def _run_submit_plan(plan):
                return _teammate_submit_plan(name, plan)

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
                for _ in range(50):
                    inbox = ctx.bus.read_inbox(name)
                    for msg in inbox:
                        stopped = handle_inbox_message(name, msg, messages)
                        if stopped:
                            should_shutdown = True
                            break
                    if should_shutdown:
                        break
                    if inbox and not should_shutdown:
                        non_protocol = [m for m in inbox
                                        if m.get("type") == "message"]
                        if non_protocol:
                            messages.append({"role": "user",
                                             "content": f"<inbox>{json.dumps(non_protocol)}</inbox>"})
                        log_team_history(name, "inbox_received",
                                         [{"from": m.get("from"), "type": m.get("type"),
                                           "content": truncate(m.get("content", ""))}
                                          for m in inbox], TEAM_HISTORY_DIR)
                    try:
                        request_messages = list(messages)
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
                        _is_timeout = ("timeout" in type(e).__name__.lower() or "timeout" in str(e).lower() or "timed out" in str(e).lower())
                        if _is_timeout:
                            if reactive_retries < MAX_REACTIVE_RETRIES:
                                reactive_retries += 1
                                continue
                        log_team_history(name, "llm_failed",
                                         {"error_type": type(e).__name__,
                                          "error": truncate(str(e))}, TEAM_HISTORY_DIR)
                        ctx.bus.send(name, "lead", f"[teammate] {name} LLM call failed: "f"{type(e).__name__}: {e}", "LLM API error")
                        return
                    # Guard against empty ``choices`` (content-filter / rate-limit
                    # soft-fail): some providers return HTTP 200 with ``choices=[]``.
                    # Without this guard ``choices[0]`` raises ``IndexError`` which
                    # would escape the outer try as a silent crash (P5c).
                    if not response.choices:
                        log_team_history(name, "llm_response",
                                         {"content": "",
                                          "tool_calls": "",
                                          "finish_reason": "empty_choices"},
                                         TEAM_HISTORY_DIR)
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
                            # PreToolUse hook runs user-supplied code; wrap it so
                            # a buggy/malicious hook crashes the hook (gracefully
                            # degrades to "not blocked") instead of the teammate.
                            blocked = None
                            try:
                                blocked = trigger_hooks("PreToolUse", tool_call.function)
                            except Exception as _hook_err:
                                print(f"  \033[33m[teammate] {name} PreToolUse hook "
                                      f"error ({_hook_err}), skipping hook\033[0m")
                                log_team_history(name, "hook_error",
                                                 {"phase": "PreToolUse",
                                                  "tool": tool_call.function.name,
                                                  "error": truncate(str(_hook_err))},
                                                 TEAM_HISTORY_DIR)
                            if blocked:
                                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(blocked)})
                                continue
                            handler = sub_handlers.get(tool_call.function.name)
                            args = parse_tool_args(tool_call.function.arguments)
                            output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
                            # PostToolUse hook likewise wrapped: the tool already
                            # ran successfully, a hook crash must not discard the
                            # result or kill the teammate.
                            try:
                                trigger_hooks("PostToolUse", tool_call.function, output)
                            except Exception as _hook_err:
                                print(f"  \033[33m[teammate] {name} PostToolUse hook "
                                      f"error ({_hook_err}), skipping hook\033[0m")
                                log_team_history(name, "hook_error",
                                                 {"phase": "PostToolUse",
                                                  "tool": tool_call.function.name,
                                                  "error": truncate(str(_hook_err))},
                                                 TEAM_HISTORY_DIR)
                            log_team_history(name, "tool_called",
                                             {"tool": tool_call.function.name,
                                              "args": truncate(json.dumps(args, ensure_ascii=False)),
                                              "output": truncate(output)}, TEAM_HISTORY_DIR)
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

                if should_shutdown:
                    break
                idle_result = idle_poll(name, messages, name, role)
                if idle_result == "shutdown":
                    break
                if idle_result == "timeout":
                    break

            result = messages[-1]["content"]
            if not result:
                for msg in reversed(messages):
                    if msg["role"] == "assistant" and "content" in msg:
                        result = msg["content"]
                        if result:
                            break
                if not result:
                    result = "teamagent stopped after 50 turns without final answer."
            ctx.bus.send(name, "lead", result, "result")
        except Exception as e:
            # Catch-all for the "protection blind spot": any exception that
            # escapes the inner LLM try/except (tool handler crash, parse
            # failure, idle_poll I/O error, etc.) lands here.  Without this
            # the teammate would exit silently with an empty lead mailbox.
            # We send a ``crashed`` notification so the lead always learns.
            crash_msg = (f"[teammate] {name} crashed: "
                         f"{type(e).__name__}: {e}")
            try:
                ctx.bus.send(name, "lead", crash_msg, "crashed")
            except Exception:
                pass  # best-effort; finally still sets the event
            log_team_history(name, "crashed",
                             {"error_type": type(e).__name__,
                              "error": truncate(str(e))}, TEAM_HISTORY_DIR)
            print(f"  \033[31m[teammate] {name} crashed: "
                  f"{type(e).__name__}: {e}\033[0m")
        finally:
            # Fix 1: Set the event and mark registry status FIRST, before
            # any logging that might itself raise.  Previously
            # log_team_history() ran before evt.set(), so a logging failure
            # (e.g. json.dumps on non-serializable data) would leave the
            # event UNSET and status "running" forever - lead loses BOTH
            # notification channels.  Now the event/status are guaranteed.
            try:
                _final = result
            except (NameError, UnboundLocalError):
                _final = ""
            evt = ctx.active_teammates.get(name)
            if evt is not None:
                evt.set()
            if name in ctx.teammate_registry:
                ctx.teammate_registry[name]["status"] = "finished"
            # Logging is best-effort: never let it prevent the print or
            # mask the real exit.  Wrapped so a serialisation error here
            # cannot resurrect a "looks alive" teammate.
            try:
                log_team_history(name, "finished",
                                 {"result": truncate(_final), "idle_result": idle_result}, TEAM_HISTORY_DIR)
            except Exception as _log_err:
                print(f"  \033[33m[teammate] {name} history log failed: "
                      f"{_log_err}\033[0m")
            print(f"  \033[32m[teammate] {name} finished\033[0m")

    ctx.active_teammates[name] = threading.Event()
    ctx.teammate_registry[name] = {
        "role": role,
        "spawned_at": time.time(),
        "status": "running",
    }
    threading.Thread(target=run, daemon=True).start()
    print(f"  \033[36m[teammate] {name} spawned as {role}\033[0m")
    return (f"Teammate '{name}' spawned as {role}. "
            f"The teamagent runs in the background - you do not need to wait for it to finish. "
            f"The teamagent will notify you via events (check check_inbox for messages) "
            f"when it needs your input (e.g. plan approval) or when it has completed its work.")
