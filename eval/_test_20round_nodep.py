"""Verify the ZERO-dependency fix (no tiktoken) over 20 rounds.

Uses len(TOOLS_JSON)//4 for the tool-schema overhead instead of tiktoken.
Everything else identical to _test_20round_headtohead.py.
"""
from __future__ import annotations
import json
import sys
import statistics
sys.path.insert(0, ".")

from mcodecore.config import client, LLM_MODEL, _MSG_OVERHEAD_TOKENS
from mcodecore.tools import TOOLS

# ---- tool overhead: chars/4, NO tiktoken ----
_TOOLS_JSON = json.dumps(TOOLS, ensure_ascii=False)
TOOLS_OVERHEAD = len(_TOOLS_JSON) // 4
print(f"TOOLS_JSON chars={len(_TOOLS_JSON)}  overhead(chars/4)={TOOLS_OVERHEAD}")
print(f"(for reference: tiktoken gave 1736, actual API overhead ~1606)")


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x3000 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF
            or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF)


def _content_token_estimate(content: str) -> int:
    if not content:
        return 0
    cjk = sum(1 for ch in content if _is_cjk(ch))
    return cjk + (len(content) - cjk) // 4


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


def estimate_fixed_raw(msgs) -> int:
    est = sum(_msg_content_len_fixed(m) + _MSG_OVERHEAD_TOKENS for m in msgs)
    est += TOOLS_OVERHEAD
    return est


class CalFixed:
    def __init__(self):
        self.factor = 1.0
        self.samples = []
    def calibrated(self, raw):
        return int(raw * self.factor)
    def record(self, raw, actual):
        if raw > 0 and actual > 0:
            self.samples.append((raw, actual))
            if len(self.samples) > 50:
                self.samples.pop(0)
            ratios = sorted(a / r for r, a in self.samples)
            if ratios:
                self.factor = ratios[len(ratios) // 2]


SYSTEM = {"role": "system", "content": "You are a concise coding assistant. Reply briefly."}
TURNS = [
    "你好，请介绍一下你自己。",
    "用两句话解释什么是递归。",
    "Python 中列表和元组有什么区别？",
    "请给出一个列表推导式的单行示例。",
    "简单解释一下 async/await 的作用。",
    "What is the difference between a thread and a process?",
    "用 Python 写一个简单的线程池示例，用中文注释。",
    "解释一下 GIL 对多线程的影响。",
    "How do I create a REST API with FastAPI? Show a minimal example.",
    "把刚才的 FastAPI 示例改成支持查询参数 name 和 age。",
    "什么是装饰器？给一个计时装饰器的例子。",
    "解释一下 Python 的上下文管理器 and the with statement.",
    "用 dataclass 重写一个 Person 类，包含 name 和 age 字段。",
    "How does pytest fixture work? Give a short example.",
    "写一个 pytest fixture 用 fixture scope='module' 的示例。",
    "What is the difference between asyncio.gather and asyncio.wait?",
    "用 asyncio 写一个并发请求 3 个 URL 的示例，中文注释。",
    "解释一下 Python 的 type hints，包括 generic types like list[int]。",
    "How do I handle exceptions in async code? Show try/except with await.",
    "用一句话总结我们这 20 轮讨论过的所有主题。",
]


def call_api(messages, max_tokens=120):
    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, max_tokens=max_tokens,
        temperature=0.0, tools=TOOLS,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return (resp.choices[0].message.content or ""), resp.usage.prompt_tokens


def main():
    cal = CalFixed()
    history = [SYSTEM]
    hdr = f"{'turn':>4} | {'raw':>6} | {'calib':>6} | {'actual':>6} | {'err%':>7} | {'factor':>7}"
    print(hdr)
    print("-" * len(hdr))
    errs = []
    for i, q in enumerate(TURNS, 1):
        history.append({"role": "user", "content": q})
        raw = estimate_fixed_raw(history)
        calib = cal.calibrated(raw)
        content, actual = call_api(history)
        history.append({"role": "assistant", "content": content})
        err = (calib - actual) / actual * 100 if actual else 0
        errs.append(abs(err))
        cal.record(raw, actual)
        print(f"{i:>4} | {raw:>6} | {calib:>6} | {actual:>6} | {err:>+7.1f}% | {cal.factor:>7.4f}")
    print(f"\nSUMMARY (20 rounds, NO tiktoken): "
          f"mean|err|={statistics.mean(errs):.1f}%  max|err|={max(errs):.1f}%  "
          f"factor={cal.factor:.4f}")


if __name__ == "__main__":
    main()
