"""Synchronous sub-agent."""

from __future__ import annotations

from .config import LLM_MODEL, client
from .context import ctx
from .hooks import trigger_hooks
from .tools import SUB_HANDLERS, SUB_TOOLS, SUB_SYSTEM
from .utils import sanitize_message, sanitize_messages, parse_tool_args

MAX_REACTIVE_RETRIES = 3
def spawn_subagent(description: str) -> str:
    """Spawn a synchronous sub-agent for a complex subtask; return the final summary.

    Behavior:
    - Up to 50 turns;
    - Uses the 6 base tools + SUB_HANDLERS;
    - PreToolUse / PostToolUse hooks;
    - Ends when finish_reason is not tool_calls.
    """
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    sub_messages = [{"role": "system", "content": SUB_SYSTEM}]
    sub_messages.append({"role": "user", "content": description})
    reactive_retries = 0
    for _ in range(50):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=sanitize_messages(sub_messages),
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
            _is_timeout = ("timeout" in type(e).__name__.lower() or "timeout" in str(e).lower() or "timed out" in str(e).lower())
            if _is_timeout:
                if reactive_retries < MAX_REACTIVE_RETRIES:
                    print(f"subagent API error: retrying! {type(e).__name__}:{str(e)[:200]}")
                    reactive_retries += 1
                    continue
            return f"subagent API error: {type(e).__name__}"
        assistant_message = response.choices[0].message
        print(f"subagent: {assistant_message.content}")
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
                print(f"> subagent:\n{tool_call.function.name}:{str(output)[:100]}")
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
    print(f"\033[35m[Subagent done]\033[0m")
    return result
