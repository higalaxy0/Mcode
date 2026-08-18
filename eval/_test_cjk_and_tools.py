"""Focused diagnostics:
  1. Pure CJK string: confirm 4-chars/token heuristic undercounts Chinese.
  2. tools= schema inflation: estimate ignores tool schemas entirely.
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

from mcodecore.config import client, LLM_MODEL
from mcodecore.compact import estimate_tokens_messages, _msg_content_len as _mcl


def _msg_content_len(msgs):
    return sum(_mcl(m) for m in msgs)

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]


def actual_prompt(messages, tools=None):
    kwargs = dict(model=LLM_MODEL, messages=messages, max_tokens=1, temperature=0.0,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    if tools:
        kwargs["tools"] = tools
    r = client.chat.completions.create(**kwargs)
    return r.usage.prompt_tokens


print("=" * 70)
print("1. PURE CJK CONTENT  (heuristic: chars/4)")
print("=" * 70)
# 200 Chinese characters. BPE: ~1 token/char => ~200 tokens.
zh = "编" * 200
msgs = [{"role": "user", "content": zh}]
est = estimate_tokens_messages(msgs)
act = actual_prompt(msgs)
raw = _msg_content_len(msgs) // 4 + 4
print(f"  200 CJK chars | raw(chars/4)={raw}  est(calibrated)={est}  actual={act}")
print(f"  => heuristic UNDER-counts by {act/raw:.2f}x  (actual/raw)")

print()
zh2 = "你好世界测试" * 50  # 300 chars, common CJK
msgs2 = [{"role": "user", "content": zh2}]
est2 = estimate_tokens_messages(msgs2)
act2 = actual_prompt(msgs2)
raw = _msg_content_len(msgs2) // 4 + 4
raw2 = _msg_content_len(msgs2) // 4 + 4
print(f"  300 CJK chars | raw(chars/4)={raw2}  est(calibrated)={est2}  actual={act2}")
print(f"  => heuristic UNDER-counts by {act2/raw2:.2f}x")

print()
print("=" * 70)
print("2. ENGLISH CONTENT (heuristic should be ~right)")
print("=" * 70)
en = "hello world " * 100  # 1200 chars
msgs = [{"role": "user", "content": en}]
est = estimate_tokens_messages(msgs)
act = actual_prompt(msgs)
raw = _msg_content_len(msgs) // 4 + 4
print(f"  1200 EN chars | raw(chars/4)={raw}  est(calibrated)={est}  actual={act}")
print(f"  => ratio actual/raw={act/raw:.2f}x")

print()
print("=" * 70)
print("3. tools= SCHEMA INFLATION (estimate ignores tool schemas)")
print("=" * 70)
msgs = [{"role": "user", "content": "hi"}]
est = estimate_tokens_messages(msgs)
act_no_tools = actual_prompt(msgs)
act_with_tools = actual_prompt(msgs, tools=TOOLS)
print(f"  msg='hi' | est={est}  actual(no tools)={act_no_tools}  actual(2 tools)={act_with_tools}")
print(f"  => tool schemas add {act_with_tools - act_no_tools} tokens that estimate NEVER counts")

# Now with the FULL mcode TOOLS set
from mcodecore.tools import TOOLS as FULL_TOOLS
act_full_tools = actual_prompt(msgs, tools=FULL_TOOLS)
print(f"  msg='hi' | actual(FULL {len(FULL_TOOLS)} tools)={act_full_tools}")
print(f"  => full toolset adds {act_full_tools - act_no_tools} tokens vs est={est}")
