"""Multi-round token estimation accuracy test against the real LLM API.

For each turn we:
  1. Build the request messages (system + accumulated history).
  2. Compute estimate_tokens_messages(request_messages) -- the function under test.
  3. Call the API (non-streaming) and read response.usage.prompt_tokens.
  4. Record (estimated, actual) and the ratio.
  5. Append the assistant reply to history and continue.

We exercise several content profiles because the 4-chars-per-token heuristic
behaves very differently for English, Chinese, and code:
  - English prose
  - Chinese prose  (each CJK char is ~1 token, NOT 4 chars/token)
  - Mixed CJK + Latin
  - Code / JSON (dense punctuation)
  - A realistic mixed multi-turn coding conversation
"""
from __future__ import annotations

import json
import sys
import statistics

sys.path.insert(0, ".")

from mcodecore.config import client, LLM_MODEL
from mcodecore.compact import estimate_tokens_messages
from mcodecore.context import ctx


def call_api(messages, max_tokens=80, temperature=0.0):
    """Non-streaming call; return (content, prompt_tokens, completion_tokens)."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content or ""
    return (content,
            resp.usage.prompt_tokens,
            getattr(resp.usage, "completion_tokens", None))


def run_scenario(name: str, turns: list[str], max_tokens=80):
    """Run a fixed set of user turns, measuring estimate vs actual each round.

    ``turns`` is the list of user message strings to send one per round.
    """
    print("\n" + "=" * 78)
    print(f"SCENARIO: {name}   ({len(turns)} turns)")
    print("=" * 78)

    system = {"role": "system", "content": "You are a concise assistant. Reply briefly."}
    history: list[dict] = [system]

    rows = []
    for i, user_text in enumerate(turns, 1):
        history.append({"role": "user", "content": user_text})
        estimated = estimate_tokens_messages(history)
        try:
            content, actual, comp = call_api(history, max_tokens=max_tokens)
        except Exception as e:
            print(f"  turn {i}: API error {e}")
            history.append({"role": "assistant", "content": "[error]"})
            continue
        history.append({"role": "assistant", "content": content})
        ratio = actual / estimated if estimated else float("inf")
        err_pct = (actual - estimated) / actual * 100 if actual else 0
        rows.append((i, estimated, actual, ratio, err_pct))
        print(f"  turn {i:2d} | est={estimated:5d}  actual={actual:5d}  "
              f"ratio={ratio:.3f}  err={err_pct:+6.1f}%  "
              f"reply={len(content)}ch")
        # Feed the sample to the real calibrator too, mirroring agent.py
        ctx.calibrator.record(estimated, actual)

    if rows:
        ratios = [r[3] for r in rows]
        errs = [r[4] for r in rows]
        print(f"  -- ratio  mean={statistics.mean(ratios):.3f}  "
              f"median={statistics.median(ratios):.3f}  "
              f"min={min(ratios):.3f}  max={max(ratios):.3f}")
        print(f"  -- err%   mean={statistics.mean(errs):+.1f}  "
              f"median={statistics.median(errs):+.1f}  "
              f"min={min(errs):+.1f}  max={max(errs):+.1f}")
    return rows


def char_profile(messages):
    """Return (total_chars, n_msgs) for diagnostics."""
    n = 0
    c = 0
    for m in messages:
        n += 1
        content = m.get("content", "")
        if isinstance(content, str):
            c += len(content)
    return c, n


def main():
    print(f"model={LLM_MODEL}  overhead/msg={4}  calibration_factor(start)="
          f"{ctx.calibrator.calibration_factor}")

    # ---- Scenario A: English prose, growing conversation ---------------
    en_turns = [
        "Hello! Who are you?",
        "Explain recursion in two sentences.",
        "What is the difference between a list and a tuple in Python?",
        "Give me a one-line example of a list comprehension.",
        "Now explain async/await briefly.",
        "Summarize everything we discussed so far in one sentence.",
    ]
    run_scenario("A. English prose", en_turns)

    # ---- Scenario B: Chinese prose (worst case for 4-chars/token) ------
    zh_turns = [
        "你好，请介绍一下你自己。",
        "用两句话解释什么是递归。",
        "Python 中列表和元组有什么区别？",
        "请给出一个列表推导式的单行示例。",
        "简单解释一下 async/await 的作用。",
        "用一句话总结我们目前讨论过的所有内容。",
    ]
    run_scenario("B. Chinese prose", zh_turns)

    # ---- Scenario C: Mixed CJK + Latin --------------------------------
    mixed_turns = [
        "请用 English 回答：what is a decorator in Python?",
        "再给一个用 Flask 写 hello world 的示例代码。",
        "解释一下刚才代码里 @app.route 这个 decorator 的作用。",
        "如果我想加一个 query parameter name，应该怎么改？",
        "把这个改成 async 的版本，用 quart 代替 flask。",
    ]
    run_scenario("C. Mixed CJK + Latin", mixed_turns)

    # ---- Scenario D: Code / JSON dense content ------------------------
    code_turns = [
        "Review this function for bugs:\n\n"
        "def fib(n):\n    if n < 2: return n\n    return fib(n-1)+fib(n-2)\n",
        "Now here is a JSON config, is it valid?\n"
        '{"server": {"host": "0.0.0.0", "port": 8080}, "debug": true, "items": [1,2,3]}\n',
        "Convert that JSON into a Python dict literal with type hints.",
        "Add error handling that retries 3 times on ConnectionError.",
    ]
    run_scenario("D. Code / JSON", code_turns, max_tokens=120)

    # ---- Scenario E: realistic mixed coding chat (longer) -------------
    real_turns = [
        "I'm building a CLI todo app in Python. Where should I start?",
        "How do I persist todos to a JSON file?",
        "Write a function load_todos(path) that returns a list of dicts.",
        "现在加一个 mark_complete(todo_id) 的功能，用中文注释。",
        "Add unit tests using pytest for load_todos and mark_complete.",
        "Refactor: split into models.py, storage.py, cli.py modules.",
    ]
    run_scenario("E. Realistic mixed coding chat", real_turns, max_tokens=150)

    # ---- Report calibrator state --------------------------------------
    print("\n" + "=" * 78)
    print("CALIBRATOR STATE AFTER ALL SCENARIOS")
    print("=" * 78)
    print(f"  samples collected: {len(ctx.calibrator.samples)}")
    print(f"  calibration_factor: {ctx.calibrator.calibration_factor:.4f}")
    if ctx.calibrator.samples:
        ests = [s[0] for s in ctx.calibrator.samples]
        acts = [s[1] for s in ctx.calibrator.samples]
        ratios = [a / e for e, a in ctx.calibrator.samples]
        print(f"  est  range: {min(ests)} .. {max(ests)}")
        print(f"  actual range: {min(acts)} .. {max(acts)}")
        print(f"  ratio range: {min(ratios):.3f} .. {max(ratios):.3f}  "
              f"(median {statistics.median(ratios):.3f})")


if __name__ == "__main__":
    main()
