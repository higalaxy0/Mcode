"""Quick connectivity check: does the real LLM API respond with usage info?"""
import sys
sys.path.insert(0, ".")

from mcodecore.config import client, LLM_MODEL

resp = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ],
    max_tokens=100,
    temperature=0.0,
)
print("model:", LLM_MODEL)
print("finish:", resp.choices[0].finish_reason)
print("content:", repr(resp.choices[0].message.content))
print("usage:", resp.usage)
print("  prompt_tokens:", getattr(resp.usage, "prompt_tokens", None))
print("  completion_tokens:", getattr(resp.usage, "completion_tokens", None))
print("  total_tokens:", getattr(resp.usage, "total_tokens", None))
