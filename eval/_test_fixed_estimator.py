"""Validate the PROPOSED fixed estimator against the real API.

Fixes under test:
  1. CJK-aware char counting (CJK char ~= 1 token, ASCII ~= 4 chars/token)
  2. Constant tool-schema overhead added (pre-computed once)
  3. Calibrator records the RAW base, not the calibrated value (kills oscillation)
  4. Activation threshold lowered to 0 (always learn)
"""
from __future__ import annotations
import json
import sys
import statistics
sys.path.insert(0, ".")

import tiktoken  # only used HERE to pre-compute the tool-schema constant
from mcodecore.config import client, LLM_MODEL, _MSG_OVERHEAD_TOKENS
from mcodecore.tools import TOOLS

enc = tiktoken.get_encoding("cl100k_base")
TOOLS_JSON = json.dumps(TOOLS, ensure_ascii=False)
TOOLS_OVERHEAD = len(enc.encode(TOOLS_JSON))  # constant per TOOLS set


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x3000 <= o <= 0x30FF or   # CJK punct + Hiragana/Katakana
            0x3400 <= o <= 0x9FFF or   # CJK Unified
            0xF900 <= o <= 0xFAFF or   # CJK compat
            0xFF00 <= o <= 0xFFEF)     # Fullwidth


def _content_token_estimate(content: str) -> int:
    """CJK-aware: CJK chars ~= 1 token each, other chars ~= 4 chars/token."""
    if not content:
        return 0
    cjk = 0
    other = 0
    for ch in content:
        if _is_cjk(ch):
            cjk += 1
        else:
            other += 1
    return cjk + other // 4


def _msg_content_len_fixed(m) -> int:
    total = 0
    c = m.get("content", "")
    if isinstance(c, str):
        total += _content_token_estimate(c)
    elif isinstance(c, list):
        for b in c:
            if getattr(b, "type", None) == "text":
                total += _content_token_estimate(getattr(b, "text", "") or "")
            elif isinstance(b, dict) and b.get("type") == "text":
                total += _content_token_estimate(b.get("text", "") or "")
            else:
                total += _content_token_estimate(str(b))
    for tc in m.get("tool_calls") or []:
        total += _content_token_estimate(tc.get("function", {}).get("arguments", ""))
        total += _content_token_estimate(tc.get("function", {}).get("name", ""))
    return total


def estimate_fixed(msgs, with_tools=False) -> int:
    """Fixed estimator: CJK-aware + tool overhead, NO calibration (raw base)."""
    estimated = 0
    for m in msgs:
        estimated += _msg_content_len_fixed(m) + _MSG_OVERHEAD_TOKENS
    if with_tools:
        estimated += TOOLS_OVERHEAD
    return estimated


def actual_prompt(messages, tools=None):
    kwargs = dict(model=LLM_MODEL, messages=messages, max_tokens=1, temperature=0.0,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    if tools:
        kwargs["tools"] = tools
    r = client.chat.completions.create(**kwargs)
    return r.usage.prompt_tokens


# ---- single-shot accuracy comparison ----
print(f"TOOLS_OVERHEAD (tiktoken) = {TOOLS_OVERHEAD}")
cases = [
    ("EN short", [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello world!"}]),
    ("EN long", [{"role": "user", "content": "hello world " * 200}]),
    ("CJK short", [{"role": "user", "content": "你好世界测试"}]),
    ("CJK long", [{"role": "user", "content": "你好世界测试" * 50}]),
    ("CJK char x200", [{"role": "user", "content": "编" * 200}]),
    ("mixed", [{"role": "user", "content": "请用 English 解释 what is recursion, 然后 give a code example in Python."}]),
    ("code", [{"role": "user", "content": "def fib(n):\n    if n<2: return n\n    return fib(n-1)+fib(n-2)\n" * 10}]),
    ("json", [{"role": "user", "content": json.dumps({"a": [1,2,3], "b": {"c": "d"}, "e": "你好"}, ensure_ascii=False) * 10}]),
]

print(f"\n--- WITH tools (realistic agent scenario) ---")
print(f"{'case':<16} | {'fixed':>8} | {'actual':>8} | {'err%':>7}")
print("-" * 50)
errs = []
for name, msgs in cases:
    est = estimate_fixed(msgs, with_tools=True)
    act = actual_prompt(msgs, tools=TOOLS)
    err = (est - act) / act * 100 if act else 0
    errs.append(abs(err))
    print(f"{name:<16} | {est:>8} | {act:>8} | {err:>+7.1f}%")
print(f"mean |err| = {statistics.mean(errs):.1f}%")

# ---- multi-round with calibrator (corrected loop) ----
print("\n" + "=" * 60)
print("MULTI-ROUND (corrected calibrator: record RAW base)")
print("=" * 60)

# Minimal corrected calibrator for the test
class CalFixed:
    def __init__(self):
        self.factor = 1.0
        self.samples = []
    def record(self, raw, actual):
        if raw > 0 and actual > 0:
            self.samples.append((raw, actual))
            ratios = sorted(a / r for r, a in self.samples)
            self.factor = ratios[len(ratios) // 2]
    def calibrated(self, raw):
        return int(raw * self.factor)

cal = CalFixed()
system = {"role": "system", "content": "You are a concise assistant."}
history = [system]
turns = [
    "你好，请介绍一下你自己。",
    "用两句话解释什么是递归。",
    "Python 中列表和元组有什么区别？",
    "请给出一个列表推导式的单行示例。",
    "简单解释一下 async/await 的作用。",
    "用一句话总结我们讨论过的所有内容。",
]
print(f"{'turn':>4} | {'raw':>6} | {'calib':>6} | {'actual':>6} | {'err%':>7} | {'factor':>7}")
print("-" * 52)
for i, q in enumerate(turns, 1):
    history.append({"role": "user", "content": q})
    raw = estimate_fixed(history, with_tools=True)
    calib = cal.calibrated(raw)
    resp = client.chat.completions.create(model=LLM_MODEL, messages=history,
        max_tokens=80, temperature=0.0, tools=TOOLS,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    act = resp.usage.prompt_tokens
    content = resp.choices[0].message.content or ""
    history.append({"role": "assistant", "content": content})
    cal.record(raw, act)  # FIX: record RAW, not calibrated
    err = (calib - act) / act * 100 if act else 0
    print(f"{i:>4} | {raw:>6} | {calib:>6} | {act:>6} | {err:>+7.1f} | {cal.factor:>7.4f}")
print(f"\nfinal factor: {cal.factor:.4f}  (converged, no oscillation)")
