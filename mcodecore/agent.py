"""Lead Agent main loop + REPL entry."""

from __future__ import annotations

import copy
import json
import queue
import threading
import time

from .config import (LLM_MODEL, MAX_REACTIVE_RETRIES, CONTEXT_LIMIT, client, _enable_ansi, debug)
from .context import ctx
from .exceptions import AgentInterrupt
from .hooks import trigger_hooks
from .streaming import stream_response, classify_transient, retry_after_seconds, backoff_delay
from .compact import (estimate_tokens_messages, tool_result_budget, snip_compact,
                      micro_compact, compact_history, reactive_compact)
from .memory import _load_memories_async, _await_memories, _post_turn_memory
from .bus import consume_lead_inbox, format_inbox_msg, get_pending_plan_approvals
from .tools import build_system, TOOLS, TOOL_HANDLERS
from .utils import sanitize_messages, sanitize_message, parse_tool_args

# -- Event-driven REPL timing constants ----------------------------------------
# Teammate watcher poll interval (seconds).  Trades latency for I/O cost;
# 0.5s means a teammate result is surfaced within ~0.5s of completion.
_WATCHER_INTERVAL = 0.5
# Main-loop ``queue.get`` timeout (seconds).  Must be short enough that
# Ctrl+C on Windows is reliably caught (unbounded ``get`` can swallow it)
# but long enough to avoid busy-spinning.  0.5s is a good balance.
_REPL_POLL_INTERVAL = 0.5


def agent_loop(messages: list) -> None:
    """Lead Agent main loop.

    Each iteration:
    - rebuild the system prompt;
    - sanitize the tail message (ensure assistant messages carry content);
    - three-tier compaction + auto compact;
    - stream the response;
    - timeout retry / reactive compact;
    - tool execution + hooks;
    - on non-tool-call finish, trigger memory extraction + Stop hook.
    """
    reactive_retries = 0
    _mem_holder = _load_memories_async(messages)
    memory_turn = len(messages) - 1 if messages and isinstance(messages[-1].get("content"), str) else None

    # Fix #2: Stall detector.  Track consecutive turns where the ONLY
    # tools called are polling tools (check_inbox, teammate_status).
    # After STALL_MAX consecutive polling-only turns, force a natural
    # stop so the lead doesn't busy-loop forever while waiting for
    # teammates that may have already finished or crashed.
    STALL_MAX = 3
    _polling_only_streak = 0
    _POLLING_TOOLS = {"check_inbox", "teammate_status"}

    while True:
        # Fix #3: Re-inject pending plan approvals as reminders.  If a
        # teammate submitted a plan that the lead hasn't acted on (e.g.
        # after compaction or a long tool sequence), the original
        # plan_approval_request may have been consumed from the inbox
        # and lost.  Re-inject a reminder so the lead can never forget.
        _pending_plans = get_pending_plan_approvals()
        if _pending_plans:
            reminders = "\n".join(
                f"  - Request {r['request_id']} from {r['sender']}: "
                f"{r['summary']}" for r in _pending_plans)
            messages.append({"role": "user",
                             "content": f"[Reminder] You have "
                             f"{len(_pending_plans)} pending plan approval(s) "
                             f"awaiting your review. Use review_plan to "
                             f"approve or reject:\n{reminders}"})
        SYSTEM = build_system()
        pre_compress = copy.deepcopy(messages)
        api_messages = copy.deepcopy(messages)
        api_messages[:] = tool_result_budget(api_messages)
        api_messages[:] = snip_compact(api_messages)
        api_messages[:] = micro_compact(api_messages)
        if estimate_tokens_messages(api_messages) > CONTEXT_LIMIT:
            debug(f"[auto compact, context size = {estimate_tokens_messages(api_messages)}]")
            api_messages[:] = compact_history(api_messages)
        messages[:] = copy.deepcopy(api_messages)
        debug("agent: thinking!")
        try:
            request_messages = api_messages.copy()
            request_messages.insert(0, {"role": "system", "content": SYSTEM})
            memories_content = _await_memories(_mem_holder)
            memory_turn = None
            for mi in range(len(request_messages) - 1, -1, -1):
                if request_messages[mi].get("role") == "user" and isinstance(request_messages[mi].get("content"), str):
                    memory_turn = mi
                    break
            if memories_content and memory_turn is not None:
                request_messages[memory_turn] = {
                    **request_messages[memory_turn],
                    "content": memories_content + "\n\n" + request_messages[memory_turn]["content"],
                }
            # Ensure every message carries a `content` key; the backend rejects
            # messages missing content (e.g. pure tool-call assistant turns).
            request_messages = sanitize_messages(request_messages)
            response = stream_response(
                model=LLM_MODEL,
                messages=request_messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=16384,
                timeout=300,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False}
                }
            )
            reactive_retries = 0
        except AgentInterrupt:
            print("\033[33m[interrupted by user]\033[0m")
            return
        except Exception as e:
            _e_name = type(e).__name__
            _e_str = str(e).lower()
            # Transient errors: 429 / 5xx / connection errors / timeouts.
            # Retry with exponential backoff, respecting Retry-After header.
            if classify_transient(e):
                if reactive_retries < MAX_REACTIVE_RETRIES:
                    delay = retry_after_seconds(e)
                    if delay is None:
                        delay = backoff_delay(reactive_retries)
                    debug(f"[agent retry {reactive_retries + 1}/{MAX_REACTIVE_RETRIES} "
                          f"after {delay:.1f}s: {_e_name}: {str(e)[:200]}")
                    import time as _time
                    _time.sleep(delay)
                    reactive_retries += 1
                    continue
            if ("prompt_too_long" in _e_str or "too many tokens" in _e_str) and reactive_retries < MAX_REACTIVE_RETRIES:
                debug("[reactive compact]")
                messages[:] = reactive_compact(messages)
                reactive_retries += 1
                continue
            debug(f"[API error, {_e_name}: {str(e)[:200]}]")
            raise
        if getattr(response, "usage", None) and getattr(response.usage, "prompt_tokens", None):
            _estimated_req = estimate_tokens_messages(request_messages)
            ctx.calibrator.record(_estimated_req, response.usage.prompt_tokens)
        assistant_message = response.choices[0].message
        messages.append(sanitize_message(assistant_message.model_dump(exclude_none=True)))
        if response.choices[0].finish_reason == "interrupted":
            return
        if response.choices[0].finish_reason != "tool_calls":
            threading.Thread(
                target=_post_turn_memory,
                args=(pre_compress,),
                daemon=True,
                name="memory-extract"
            ).start()
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return
        for tool_call in (assistant_message.tool_calls or []):
            if response.choices[0].finish_reason != "tool_calls":
                continue
            print(f"\033[36m> {tool_call.function.name}:{tool_call.function.arguments[:80]}\033[0m")
            blocked = trigger_hooks("PreToolUse", tool_call.function)
            if blocked:
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(blocked)})
                continue
            handler = TOOL_HANDLERS.get(tool_call.function.name)
            args = parse_tool_args(tool_call.function.arguments)
            try:
                output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
            except AgentInterrupt:
                print("\033[33m[tool interrupted by user]\033[0m")
                output = "[interrupted by user]"
                messages.append({"role": "user", "content": "interrupted by user"})
                return
            except Exception as e:
                output = f"Error: {e}"
            trigger_hooks("PostToolUse", tool_call.function, output)
            print(f"{tool_call.function.name}:{str(output)[:300]}")
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

        # Fix #2: Stall detector.  Check whether ALL tools called this
        # turn were polling-only (check_inbox, teammate_status).  If so,
        # increment the streak; otherwise reset.  When the streak reaches
        # STALL_MAX, inject a force-stop message.  If the LLM still
        # polls after being told to stop (streak > STALL_MAX), force a
        # hard return so the REPL regains control.
        _tools_this_turn = {tc.function.name for tc in (assistant_message.tool_calls or [])}
        if _tools_this_turn and _tools_this_turn.issubset(_POLLING_TOOLS):
            _polling_only_streak += 1
        else:
            _polling_only_streak = 0
        if _polling_only_streak == STALL_MAX:
            debug(f"[stall detector] {STALL_MAX} consecutive polling-only "
                  f"turns, injecting stop guidance")
            messages.append({"role": "user",
                             "content": "[System] You have been polling "
                             "repeatedly without taking action. If you are "
                             "waiting for teammates, they may have finished "
                             "or encountered issues. Stop and summarize "
                             "the current state for the user."})
        elif _polling_only_streak > STALL_MAX:
            debug(f"[stall detector] LLM ignored stop guidance, forcing return")
            return


def _run_agent_turn(history: list) -> bool:
    """Run one iteration of agent_loop; return whether the REPL should exit."""
    try:
        agent_loop(history)
    except AgentInterrupt:
        print("\033[33m[interrupted]\033[0m")
    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt]")
        return True
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
    return False


def _drain_inbox(history: list) -> bool:
    """Check finished teammates, consume lead's inbox, and inject it into history."""
    finished = [name for name, evt in list(ctx.active_teammates.items())
                if evt.is_set()]
    for name in finished:
        ctx.active_teammates.pop(name, None)
        if name in ctx.teammate_registry:
            ctx.teammate_registry[name]["status"] = "finished"

    inbox_msgs = consume_lead_inbox(route_protocol=True)
    if not inbox_msgs:
        return False
    inbox_text = "\n".join(format_inbox_msg(m) for m in inbox_msgs)
    history.append({"role": "user", "content": f"[Inbox]\n{inbox_text}"})
    debug(f"\n[Inbox: {len(inbox_msgs)} messages injected]")
    return _run_agent_turn(history)


# -- Event-driven REPL infrastructure ------------------------------------------

# REPL prompt string (cyan).  Rendered without a trailing newline.
_PROMPT = "\033[36mMcode >> \033[0m"


def _print_prompt() -> None:
    """Render the REPL prompt at the current cursor position (no newline).

    Extracted as a helper so tests can intercept prompt rendering and
    assert its ordering relative to agent turns.
    """
    print(_PROMPT, end="", flush=True)

def _lead_mailbox_has_messages() -> bool:
    """Non-destructive check: is there anything in lead's mailbox?"""
    from .config import MAILBOX_DIR
    inbox = MAILBOX_DIR / "lead.jsonl"
    try:
        return inbox.exists() and inbox.stat().st_size > 0
    except OSError:
        return False


def _teammate_watcher(event_q: queue.Queue, stop: threading.Event) -> None:
    """Background thread: detect teammate completion / pending inbox messages.

    Polls every ``_WATCHER_INTERVAL`` seconds.  Two independent signals
    trigger a ``("teammate", None)`` event:

    1. Any entry in ``ctx.active_teammates`` whose ``Event`` is set
       (teammate finished its run).
    2. Lead's mailbox JSONL is non-empty (a teammate sent a result,
       crashed message, or plan_approval_request mid-work).

    Signal (2) is checked independently of (1) because a teammate may
    still be running when it sends a plan_approval_request, and because
    the ``finally`` block sends orphan-release messages *after*
    ``evt.set()`` — by which time ``_drain_inbox`` may have already
    popped the teammate from ``active_teammates``.
    """
    while not stop.is_set():
        time.sleep(_WATCHER_INTERVAL)
        if stop.is_set():
            break
        # Signal 1: finished teammates
        for name, evt in list(ctx.active_teammates.items()):
            if evt.is_set():
                event_q.put(("teammate", None))
                break
        # Signal 2: pending inbox messages (independent of signal 1)
        if _lead_mailbox_has_messages():
            event_q.put(("teammate", None))


def _input_reader(event_q: queue.Queue, stop: threading.Event) -> None:
    """Background thread: read stdin lines and push to the event queue.

    ``input()`` blocks the calling thread; running it here lets the
    main loop stay responsive to teammate events via ``event_q.get``.

    The prompt is NOT printed here (plain ``input()``, no argument).
    The main loop owns prompt rendering (via ``_print_prompt``): it
    prints the prompt only when the REPL is idle.  If this thread
    printed the prompt itself, it would re-print it immediately after
    each Enter - BEFORE the agent's output - causing two known
    display bugs:

    1. A spurious/blank prompt line appearing right after Enter
       (the pre-mature prompt gets buried under agent output);
    2. No prompt visible after the agent finishes a task (the buried
       prompt is the one blocking ``input()``, so nothing fresh is
       ever printed again).
    """
    while not stop.is_set():
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            event_q.put(("quit", None))
            return
        event_q.put(("input", line))


def main() -> None:
    """REPL main entry (event-driven)."""
    _enable_ansi()
    from .mcp import init_mcp
    init_mcp()
    from .tools import _inject_mcp_tools
    _inject_mcp_tools()
    print("Enter a question, press Enter to send. Type q to quit.\n")

    # Fix #8: Startup orphan sweep.  Release in_progress tasks left by
    # teammates from a previous session (daemon threads are killed on
    # exit without running their finally block).  Without this, such
    # tasks remain stuck.  Task board is session-scoped
    # (``.tasks/<SESSION_ID>``), so this only ever touches THIS session's
    # tasks and can no longer release another window's in-progress work.
    from .tasks import release_orphaned_tasks
    _released = release_orphaned_tasks(set(ctx.active_teammates))
    if _released:
        print(f"  \033[33m[orphan-sweep] released {_released} orphaned "
              f"task(s) from previous session\033[0m")

    # Multi-window fix: quarantine leftover flat mailbox files from
    # pre-session-scoped versions so this session can never consume (steal)
    # another window's undelivered mail.
    from .config import quarantine_legacy_mailboxes
    _moved = quarantine_legacy_mailboxes()
    if _moved:
        print(f"  \033[33m[mail-quarantine] moved {_moved} legacy mailbox "
              f"file(s) to .mailboxes/orphan/\033[0m")

    history: list = []
    event_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    watcher = threading.Thread(
        target=_teammate_watcher, args=(event_q, stop),
        daemon=True, name="teammate-watcher")
    reader = threading.Thread(
        target=_input_reader, args=(event_q, stop),
        daemon=True, name="input-reader")
    watcher.start()
    reader.start()

    _prompt_shown = False
    while True:
        if not _prompt_shown:
            _print_prompt()
            _prompt_shown = True
        try:
            # Bounded timeout so Ctrl+C (KeyboardInterrupt) is reliably
            # caught on Windows where an unbounded ``queue.get`` can
            # swallow the signal.
            kind, data = event_q.get(timeout=_REPL_POLL_INTERVAL)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break

        if kind == "quit":
            break

        if kind == "input":
            query = data
            if query.strip().lower() in ("q", "exit", "quit"):
                break
            if not query.strip():
                # Empty line: still check inbox (preserves old behaviour).
                # The consumed Enter moved the cursor to a fresh line, so
                # the prompt must be re-rendered.
                _prompt_shown = False
                if _drain_inbox(history):
                    break
                continue
            trigger_hooks("UserPromptSubmit", query)
            history.append({"role": "user", "content": query})
            # Agent output follows on fresh lines; re-render the prompt
            # once the turn finishes and the queue goes idle.
            _prompt_shown = False
            if _run_agent_turn(history):
                break
            if _drain_inbox(history):
                break

        elif kind == "teammate":
            # Re-render the prompt only if the drain actually consumed
            # mailbox messages (which implies agent output on fresh
            # lines).  A teammate-finished signal with an empty mailbox
            # produces no output; re-printing then would stack prompts
            # on one line ("Mcode >> Mcode >> ").
            _dirty = _lead_mailbox_has_messages()
            if _drain_inbox(history):
                break
            if _dirty:
                _prompt_shown = False

    stop.set()
    from .mcp import shutdown_mcp
    shutdown_mcp()


if __name__ == "__main__":
    main()
