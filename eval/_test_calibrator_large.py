"""Test the calibrator activation path: large context (>1000 tokens).

The TokenCalibrator only records samples when estimated > 1000 tokens.
This script builds a large context and runs several rounds to see whether
the sliding-window median calibration actually converges and reduces error.
"""
from __future__ import annotations

import sys
import statistics

sys.path.insert(0, ".")

from mcodecore.config import client, LLM_MODEL
from mcodecore.compact import estimate_tokens_messages
from mcodecore.context import ctx


def call_api(messages, max_tokens=40, temperature=0.0):
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content or ""
    return content, resp.usage.prompt_tokens


def main():
    print(f"model={LLM_MODEL}")
    print(f"calibration_factor(start)={ctx.calibrator.calibration_factor}")

    # Seed a large context with mixed content (English + Chinese + code)
    # so the very first estimate already exceeds 1000 tokens.
    seed_blocks = [
        ("system", "You are a concise coding assistant. Reply in one short sentence."),
        ("user", "Here is a long passage about Python. " * 40),
        ("assistant", "Understood. Python is a high-level, interpreted language."),
        ("user", "这是一段很长的中文说明，用来测试中文 token 的估算准确性。" * 30),
        ("assistant", "好的，我已了解中文部分的内容。"),
        ("user",
         "def process(items):\n"
         "    results = []\n"
         "    for i, item in enumerate(items):\n"
         "        if item is not None and item.get('active'):\n"
         "            results.append({'index': i, 'value': item['value'] * 2})\n"
         "    return results\n" * 8),
        ("assistant", "I see the process function. It filters active items and doubles values."),
    ]
    history = [{"role": r, "content": c} for r, c in seed_blocks]

    est0 = estimate_tokens_messages(history)
    print(f"\nseed estimate (pre-call): {est0} tokens")

    followups = [
        "Summarize the Python passage above.",
        "用一句话总结上面的中文内容。",
        "Does the process function have any bug? Answer briefly.",
        "Rewrite process() using a list comprehension, one line.",
        "把你的列表推导式版本翻译成中文注释。",
        "Finally, list the three main topics we covered.",
    ]

    print(f"\n{'turn':>4} | {'est':>6} | {'actual':>6} | {'ratio':>6} | {'err%':>7} | "
          f"{'factor':>7} | {'nsamp':>6}")
    print("-" * 72)

    for i, q in enumerate(followups, 1):
        history.append({"role": "user", "content": q})
        estimated = estimate_tokens_messages(history)
        content, actual = call_api(history, max_tokens=40)
        history.append({"role": "assistant", "content": content})
        ratio = actual / estimated if estimated else float("inf")
        err = (actual - estimated) / actual * 100 if actual else 0
        # Mirror agent.py: record sample into the calibrator
        ctx.calibrator.record(estimated, actual)
        print(f"{i:>4} | {estimated:>6} | {actual:>6} | {ratio:>6.3f} | {err:>+7.1f} | "
              f"{ctx.calibrator.calibration_factor:>7.4f} | "
              f"{len(ctx.calibrator.samples):>6}")

    print("\n--- final calibrator state ---")
    print(f"  samples: {len(ctx.calibrator.samples)}")
    print(f"  factor : {ctx.calibrator.calibration_factor:.4f}")
    if ctx.calibrator.samples:
        rs = [a / e for e, a in ctx.calibrator.samples]
        print(f"  sample ratios: min={min(rs):.3f} median={statistics.median(rs):.3f} "
              f"max={max(rs):.3f}")
        print(f"  raw samples: {ctx.calibrator.samples}")


if __name__ == "__main__":
    main()
