"""Measure tool-schema overhead and validate the oscillation hypothesis."""
from __future__ import annotations
import json
import sys
sys.path.insert(0, ".")

from mcodecore.tools import TOOLS
from mcodecore.config import client, LLM_MODEL
from mcodecore.compact import estimate_tokens_messages

# 1. Tool schema JSON length vs actual overhead
tools_json = json.dumps(TOOLS, ensure_ascii=False)
print(f"full TOOLS: {len(TOOLS)} tools")
print(f"json len: {len(tools_json)} chars")
print(f"json/4 = {len(tools_json)//4}  (rough token guess)")

# Actual overhead
msgs = [{"role": "user", "content": "hi"}]
r_no = client.chat.completions.create(model=LLM_MODEL, messages=msgs, max_tokens=1,
    temperature=0.0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
r_yes = client.chat.completions.create(model=LLM_MODEL, messages=msgs, max_tokens=1,
    temperature=0.0, tools=TOOLS,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
print(f"actual no-tools prompt_tokens: {r_no.usage.prompt_tokens}")
print(f"actual with-tools prompt_tokens: {r_yes.usage.prompt_tokens}")
print(f"actual tool overhead: {r_yes.usage.prompt_tokens - r_no.usage.prompt_tokens}")
print(f"  => json/4 estimate error: {(len(tools_json)//4) - (r_yes.usage.prompt_tokens - r_no.usage.prompt_tokens)}")

# 2. Oscillation: simulate the calibrator feedback loop with a fixed real ratio
print("\n--- calibrator oscillation simulation ---")
# Suppose true ratio actual/raw = 1.3 for all samples (CJK-heavy).
# agent.py records (calibrated, actual) where calibrated = raw * factor.
from mcodecore.calibrator import TokenCalibrator
cal = TokenCalibrator()
raw_base = 5000  # a representative raw base estimate
true_ratio = 1.3
for step in range(8):
    calibrated = cal.calibrated(raw_base)      # what agent.py computes for display
    actual = int(raw_base * true_ratio)         # real API prompt_tokens
    cal.record(calibrated, actual)              # BUG: records calibrated, not raw
    print(f"  step {step}: factor={cal.calibration_factor:.4f}  "
          f"calibrated={calibrated}  actual={actual}  "
          f"ratio_stored={actual/calibrated:.3f}")

print("\n--- correct loop: record RAW base ---")
cal2 = TokenCalibrator()
for step in range(8):
    actual = int(raw_base * true_ratio)
    cal2.record(raw_base, actual)               # FIX: record raw base
    print(f"  step {step}: factor={cal2.calibration_factor:.4f}  "
          f"calibrated={cal2.calibrated(raw_base)}  actual={actual}")
