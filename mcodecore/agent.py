"""Lead Agent main loop + REPL entry."""

from __future__ import annotations

import copy
import json
import threading

from .config import (LLM_MODEL, MAX_REACTIVE_RETRIES, CONTEXT_LIMIT, client, _enable_ansi, debug)
from .context import ctx
from .exceptions import AgentInterrupt
from .hooks import trigger_hooks
from .streaming import stream_response, classify_transient, retry_after_seconds, backoff_delay
from .compact import (estimate_tokens_messages, tool_result_budget, snip_compact,
                      micro_compact, compact_history, reactive_compact)
from .memory import _load_memories_async, _await_memories, _post_turn_memory
from .bus import consume_lead_inbox
from .tools import build_system, TOOLS, TOOL_HANDLERS
from .utils import sanitize_messages, sanitize_message, parse_tool_args


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
    while True:
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
        print("\033[1;32m\nagent: thinking!\033[0m")
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
    inbox_text = "\n".join(f"From {m['from']}: {m['content']}" for m in inbox_msgs)
    history.append({"role": "user", "content": f"[Inbox]\n{inbox_text}"})
    debug(f"\n[Inbox: {len(inbox_msgs)} messages injected]")
    return _run_agent_turn(history)


def main() -> None:
    """REPL main entry."""
    _enable_ansi()
    # Initialize MCP (connect to all configured servers)
    from .mcp import init_mcp
    init_mcp()
    from .tools import _inject_mcp_tools
    _inject_mcp_tools()
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []
    while True:
        try:
            query = input("\033[36mMcode >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query is None:
            if _drain_inbox(history):
                break
            continue
        if query.strip().lower() in ("q", "exit", "quit"):
            break
        if not query.strip():
            continue
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        if _run_agent_turn(history):
            break
        if _drain_inbox(history):
            break
    # Shut down MCP sessions on exit
    from .mcp import shutdown_mcp
    shutdown_mcp()


if __name__ == "__main__":
    main()
