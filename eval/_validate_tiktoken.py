"""Validate tiktoken (cl100k_base) accuracy vs the real glm-5.2 API usage."""
from __future__ import annotations
import json
import sys
sys.path.insert(0, ".")

import tiktoken
from mcodecore.config import client, LLM_MODEL
from mcodecore.tools import TOOLS


def actual_prompt(messages, tools=None):
    kwargs = dict(model=LLM_MODEL, messages=messages, max_tokens=1, temperature=0.0,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    if tools:
        kwargs["tools"] = tools
    r = client.chat.completions.create(**kwargs)
    return r.usage.prompt_tokens


def _s(m):
    c = m.get("content", "")
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


enc = tiktoken.get_encoding("cl100k_base")

# Pre-compute the tool-schema token overhead ONCE (it is constant per TOOLS set).
tools_json = json.dumps(TOOLS, ensure_ascii=False)
tools_token_est = len(enc.encode(tools_json))
print(f"tiktoken tool-schema est: {tools_token_est} tokens")

# Probe actual tool overhead once
msgs_hi = [{"role": "user", "content": "hi"}]
act_no = actual_prompt(msgs_hi)
act_yes = actual_prompt(msgs_hi, tools=TOOLS)
tool_overhead_actual = act_yes - act_no
print(f"actual tool overhead: {tool_overhead_actual}")
print(f"  => tiktoken tool error: {tools_token_est - tool_overhead_actual}")

# Test various content profiles WITHOUT tools
cases = [
    ("EN short", [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello world!"}]),
    ("EN long", [{"role": "user", "content": "hello world " * 200}]),
    ("CJK short", [{"role": "user", "content": "你好世界测试"}]),
    ("CJK long", [{"role": "user", "content": "你好世界测试" * 50}]),
    ("CJK repeated char", [{"role": "user", "content": "编" * 200}]),
    ("mixed", [{"role": "user", "content": "请用 English 解释 what is recursion, 然后 give a code example in Python."}]),
    ("code", [{"role": "user", "content": "def fib(n):\n    if n<2: return n\n    return fib(n-1)+fib(n-2)\n" * 10}]),
    ("json", [{"role": "user", "content": json.dumps({"a": [1,2,3], "b": {"c": "d"}, "e": "你好"}, ensure_ascii=False) * 10}]),
]

print(f"\n--- WITHOUT tools ---")
print(f"{'case':<16} | {'tiktoken':>8} | {'actual':>8} | {'err%':>7}")
print("-" * 50)
for name, msgs in cases:
    tt = sum(len(enc.encode(_s(m))) + 4 for m in msgs)  # 4 = per-msg overhead
    act = actual_prompt(msgs)
    err = (tt - act) / act * 100 if act else 0
    print(f"{name:<16} | {tt:>8} | {act:>8} | {err:>+7.1f}%")

# Now with tools added (the realistic agent scenario)
print(f"\n--- WITH tools (+{tools_token_est} tiktoken est) ---")
print(f"{'case':<16} | {'tt+tools':>8} | {'actual':>8} | {'err%':>7}")
print("-" * 50)
for name, msgs in cases:
    tt = sum(len(enc.encode(_s(m))) + 4 for m in msgs) + tools_token_est
    act = actual_prompt(msgs, tools=TOOLS)
    err = (tt - act) / act * 100 if act else 0
    print(f"{name:<16} | {tt:>8} | {act:>8} | {err:>+7.1f}%")
