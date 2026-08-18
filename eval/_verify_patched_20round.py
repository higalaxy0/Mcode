"""End-to-end verification: exercise the PATCHED mcodecore estimator +
calibrator over 20 real-API rounds.

This imports the real compact.py / calibrator.py (NOT inline copies) so it
validates the actual code that was edited.
"""
from __future__ import annotations
import sys
import statistics
sys.path.insert(0, ".")

from mcodecore.config import client, LLM_MODEL
from mcodecore.tools import TOOLS
from mcodecore.compact import (
    estimate_tokens_messages, estimate_tokens_messages_raw,
)
from mcodecore.context import ctx  # AppContext singleton

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
    history = [SYSTEM]
    hdr = (f"{'turn':>4} | {'raw':>6} | {'calib':>6} | {'actual':>6} | "
           f"{'err%':>7} | {'factor':>7}")
    print(hdr)
    print("-" * len(hdr))
    errs = []
    for i, q in enumerate(TURNS, 1):
        history.append({"role": "user", "content": q})
        raw = estimate_tokens_messages_raw(history)          # patched raw estimator
        calib = estimate_tokens_messages(history)            # patched calibrated
        content, actual = call_api(history)
        history.append({"role": "assistant", "content": content})
        err = (calib - actual) / actual * 100 if actual else 0
        errs.append(abs(err))
        # patched calibrator: record RAW base (exactly what agent.py now does)
        ctx.calibrator.record(raw, actual)
        print(f"{i:>4} | {raw:>6} | {calib:>6} | {actual:>6} | {err:>+7.1f}% | "
              f"{ctx.calibrator.calibration_factor:>7.4f}")
    print(f"\nSUMMARY (20 rounds, patched mcodecore): "
          f"mean|err|={statistics.mean(errs):.1f}%  max|err|={max(errs):.1f}%  "
          f"factor={ctx.calibrator.calibration_factor:.4f}")


if __name__ == "__main__":
    main()
