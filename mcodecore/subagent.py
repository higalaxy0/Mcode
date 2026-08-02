"""Synchronous sub-agent."""

from __future__ import annotations

from .config import LLM_MODEL, client, CONTEXT_LIMIT, debug
from .context import ctx
from .hooks import trigger_hooks
from .streaming import classify_transient, retry_after_seconds, backoff_delay
from .tools import SUB_HANDLERS, SUB_TOOLS, SUB_SYSTEM
from .utils import sanitize_message, sanitize_messages, parse_tool_args
from .compact import (estimate_tokens_messages, compact_history, reactive_compact,
                      tool_result_budget, snip_compact, micro_compact)
from .memory import _load_memories_async, _await_memories

MAX_REACTIVE_RETRIES = 3


def spawn_subagent(description: str) -> str:
    """Spawn a synchronous sub-agent for a complex subtask; return the final summary.

    Behavior:
    - Up to 50 turns;
    - Uses the 6 base tools + SUB_HANDLERS;
    - PreToolUse / PostToolUse hooks;
    - Loads relevant memories into the first user message;
    - Three-tier compaction + auto compact before each LLM call;
    - Ends when finish_reason is not tool_calls.
    """
    debug("[Subagent spawned]")
    sub_messages = [{"role": "user", "content": description}]
    # Load memories asynchronously so they are ready by the first LLM call.
    _mem_holder = _load_memories_async(sub_messages)
    reactive_retries = 0
    for _ in range(50):
        try:
            # Progressive compaction pipeline (same 4-tier architecture as
            # the lead agent):
            #   L3: tool_result_budget - persist oversized tool outputs;
            #   L1: snip_compact - sliding-window, pin task prompts;
            #   L2: micro_compact - replace old tool_result contents;
            #   auto: compact_history - LLM summary when over CONTEXT_LIMIT.
            import copy as _copy
            api_messages = _copy.deepcopy(sub_messages)
            api_messages[:] = tool_result_budget(api_messages)
            api_messages[:] = snip_compact(api_messages)
            api_messages[:] = micro_compact(api_messages)
            if estimate_tokens_messages(api_messages) > CONTEXT_LIMIT:
                debug(f"[subagent auto compact, "
                      f"context size = "
                      f"{estimate_tokens_messages(api_messages)}]")
                api_messages[:] = compact_history(api_messages)
            sub_messages[:] = _copy.deepcopy(api_messages)
            request_messages = list(api_messages)
            # Inject the system prompt at position 0 *after* compaction,
            # matching the lead agent pattern.  Previously system was in
            # sub_messages and got swept away by compact_history /
            # reactive_compact.
            request_messages.insert(0, {"role": "system", "content": SUB_SYSTEM})
            # Inject memories into the last user message (same as the lead
            # agent).  This is done *after* compaction so the memory content
            # is not lost when compaction rebuilds the message list.
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
                tools=SUB_TOOLS,
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
                    debug(f"[subagent retry {reactive_retries + 1}/{MAX_REACTIVE_RETRIES} "
                          f"after {delay:.1f}s: {type(e).__name__}: {str(e)[:200]}]")
                    import time as _time
                    _time.sleep(delay)
                    reactive_retries += 1
                    continue
            # prompt_too_long: reactive compact and retry
            if (("prompt_too_long" in _e_str
                 or "too many tokens" in _e_str)
                    and reactive_retries < MAX_REACTIVE_RETRIES):
                debug("[subagent reactive compact]")
                sub_messages[:] = reactive_compact(sub_messages)
                reactive_retries += 1
                continue
            return f"subagent API error: {type(e).__name__}: {str(e)[:200]}"
        if not response.choices:
            sub_messages.append({"role": "assistant", "content": ""})
            break
        assistant_message = response.choices[0].message
        debug(f"subagent: {assistant_message.content}")
        sub_messages.append(sanitize_message(assistant_message.model_dump(exclude_none=True)))
        if response.choices[0].finish_reason != "tool_calls":
            break
        for tool_call in (assistant_message.tool_calls or []):
            if response.choices[0].finish_reason == "tool_calls":
                blocked = trigger_hooks("PreToolUse", tool_call.function)
                if blocked:
                    sub_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(blocked)})
                    continue
                handler = SUB_HANDLERS.get(tool_call.function.name)
                args = parse_tool_args(tool_call.function.arguments)
                output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
                trigger_hooks("PostToolUse", tool_call.function, output)
                debug(f"> subagent:\n{tool_call.function.name}:{str(output)[:100]}")
                sub_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
    result = sub_messages[-1]["content"]
    if not result:
        for msg in reversed(sub_messages):
            if msg["role"] == "assistant" and "content" in msg:
                result = msg["content"]
                if result:
                    break
        if not result:
            result = "Subagent stopped after 50 turns without final answer."
    debug("[Subagent done]")
    return result
