"""20-round head-to-head test: CURRENT estimator vs FIXED estimator.

Both run over the SAME 20-turn conversation against the real glm-5.2 API
(with the full 20-tool TOOLS set, exactly like the agent).

We compare:
  - CURRENT: 4-chars/token, calibrated value fed back to record() (oscillating),
             tool schemas ignored.
  - FIXED:   CJK-aware char counting, constant tool overhead (tiktoken once),
             raw base fed to record() (no oscillation), threshold removed.

To keep the API cost fair and the comparison apples-to-apples we send each
turn ONCE and read its usage, then evaluate BOTH estimators against that same
actual value (so we make 20 API calls, not 40).
"""
from __future__ import annotations

import json
import sys
import statistics

sys.path.insert(0, ".")

import tiktoken  # used once to measure tool-schema constant
from mcodecore.config import client, LLM_MODEL, _MSG_OVERHEAD_TOKENS
from mcodecore.tools import TOOLS

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("cl100k_base")
_TOOLS_JSON = json.dumps(TOOLS, ensure_ascii=False)
TOOLS_OVERHEAD = len(enc.encode(_TOOLS_JSON))


# ---------------------------------------------------------------------------
# current estimator (verbatim logic from compact.py)
# ---------------------------------------------------------------------------
def _msg_content_len_current(m) -> int:
    total = 0
    c = m.get("content", "")
    if isinstance(c, str):
        total += len(c)
    elif isinstance(c, list):
        for b in c:
            if getattr(b, "type", None) == "text":
                total += len(getattr(b, "text", "") or "")
            elif isinstance(b, dict) and b.get("type") == "text":
                total += len(b.get("text", "") or "")
            else:
                total += len(str(b))
    for tc in m.get("tool_calls") or []:
        total += len(tc.get("function", {}).get("arguments", ""))
        total += len(tc.get("function", {}).get("name", ""))
    return total


def estimate_current_raw(msgs) -> int:
    """Raw base of the current estimator (chars/4 + overhead), NO tools."""
    return sum(_msg_content_len_current(m) // 4 + _MSG_OVERHEAD_TOKENS for m in msgs)


# ---------------------------------------------------------------------------
# fixed estimator (CJK-aware + tool overhead)
# ---------------------------------------------------------------------------
def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x3000 <= o <= 0x30FF or
            0x3400 <= o <= 0x9FFF or
            0xF900 <= o <= 0xFAFF or
            0xFF00 <= o <= 0xFFEF)


def _content_token_estimate(content: str) -> int:
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


def estimate_fixed_raw(msgs) -> int:
    """Raw base of the fixed estimator (CJK-aware + overhead + tools)."""
    est = sum(_msg_content_len_fixed(m) + _MSG_OVERHEAD_TOKENS for m in msgs)
    est += TOOLS_OVERHEAD
    return est


# ---------------------------------------------------------------------------
# two calibrators: current (records calibrated) vs fixed (records raw)
# ---------------------------------------------------------------------------
class CalCurrent:
    """Mimics TokenCalibrator BUT records the CALIBRATED value (the bug)."""
    def __init__(self):
        self.factor = 1.0
        self.samples = []
    def calibrated(self, raw):
        return int(raw * self.factor)
    def record(self, calibrated_val, actual):
        # bug: stores actual / calibrated_value (not actual / raw)
        if calibrated_val > 1000 and actual > 0:
            self.samples.append((calibrated_val, actual))
            if len(self.samples) > 50:
                self.samples.pop(0)
            ratios = sorted(a / e for e, a in self.samples if e > 1000)
            if ratios:
                self.factor = ratios[len(ratios) // 2]


class CalFixed:
    """Corrected: records the RAW base."""
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


# ---------------------------------------------------------------------------
# the 20-turn conversation (mixed CJK + Latin + code, realistic)
# ---------------------------------------------------------------------------
SYSTEM = {"role": "system",
          "content": "You are a concise coding assistant. Reply briefly."}

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
    "解释一下 Python 的上下文管理器 and the with statement。",
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
        model=LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        tools=TOOLS,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content or ""
    return content, resp.usage.prompt_tokens


def main():
    print(f"model={LLM_MODEL}  TOOLS_OVERHEAD={TOOLS_OVERHEAD}  turns={len(TURNS)}")
    print()

    cal_cur = CalCurrent()
    cal_fix = CalFixed()

    history = [SYSTEM]

    hdr = (f"{'turn':>4} | {'cur_raw':>7} {'cur_cal':>7} {'cur_err':>8} | "
           f"{'fix_raw':>7} {'fix_cal':>7} {'fix_err':>8} | "
           f"{'actual':>7} | {'cur_fac':>7} {'fix_fac':>7}")
    print(hdr)
    print("-" * len(hdr))

    cur_errs = []
    fix_errs = []
    cur_abs = []
    fix_abs = []

    for i, user_text in enumerate(TURNS, 1):
        history.append({"role": "user", "content": user_text})

        # raw bases
        cur_raw = estimate_current_raw(history)          # no tool overhead (bug 2)
        fix_raw = estimate_fixed_raw(history)            # with tool overhead + CJK-aware

        # calibrated display values (what the agent would "see")
        cur_cal = cal_cur.calibrated(cur_raw)
        fix_cal = cal_fix.calibrated(fix_raw)

        # ONE real API call; both estimators compared against it
        content, actual = call_api(history, max_tokens=120)
        history.append({"role": "assistant", "content": content})

        # errors
        cur_err = (cur_cal - actual) / actual * 100 if actual else 0
        fix_err = (fix_cal - actual) / actual * 100 if actual else 0
        cur_errs.append(cur_err)
        fix_errs.append(fix_err)
        cur_abs.append(abs(cur_err))
        fix_abs.append(abs(fix_err))

        # record (current feeds calibrated -> oscillation; fixed feeds raw)
        cal_cur.record(cur_cal, actual)
        cal_fix.record(fix_raw, actual)

        print(f"{i:>4} | {cur_raw:>7} {cur_cal:>7} {cur_err:>+8.1f}% | "
              f"{fix_raw:>7} {fix_cal:>7} {fix_err:>+8.1f}% | "
              f"{actual:>7} | {cal_cur.factor:>7.4f} {cal_fix.factor:>7.4f}")

    print("\n" + "=" * len(hdr))
    print("SUMMARY (20 rounds)")
    print("=" * len(hdr))
    print(f"  CURRENT  mean_err={statistics.mean(cur_errs):+.1f}%  "
          f"mean|err|={statistics.mean(cur_abs):.1f}%  "
          f"max|err|={max(cur_abs):.1f}%  "
          f"final_factor={cal_cur.factor:.4f}")
    print(f"  FIXED    mean_err={statistics.mean(fix_errs):+.1f}%  "
          f"mean|err|={statistics.mean(fix_abs):.1f}%  "
          f"max|err|={max(fix_abs):.1f}%  "
          f"final_factor={cal_fix.factor:.4f}")
    print(f"\n  improvement: |err| {statistics.mean(cur_abs):.1f}% -> "
          f"{statistics.mean(fix_abs):.1f}%  "
          f"({(1 - statistics.mean(fix_abs)/statistics.mean(cur_abs))*100:.0f}% reduction)")


if __name__ == "__main__":
    main()
