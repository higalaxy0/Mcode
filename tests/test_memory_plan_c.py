"""Plan C tests: type priority + body matching + feedback always-inject.

Verifies:
- feedback-type memories are always injected even with no keyword match
- feedback memories are injected even when recent text is empty
- keyword fallback matches against body (not just name+description)
- type priority: when LLM returns a list, feedback memories are merged in
"""

from __future__ import annotations

from mcodecore import memory


# --------------------------------------------------------------------------- #
# Feedback always-inject (keyword fallback path)
# --------------------------------------------------------------------------- #

def test_feedback_always_injected_keyword_fallback(monkeypatch):
    """feedback memory is injected even if the keyword doesn't match it."""
    memory.write_memory_file("fb-pref", "feedback", "user prefers tabs", "always use tabs not spaces")
    memory.write_memory_file("proj-fact", "project", "uses postgres", "database is postgresql on port 5432")

    # Force LLM failure -> keyword fallback.
    def _fail(*a, **kw):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(memory.client, "chat_completions_create", _fail, raising=False)
    monkeypatch.setattr(memory.client.chat.completions, "create", _fail)

    selected = memory.select_relevant_memories(
        [{"role": "user", "content": "tell me about the database"}])

    # "database" keyword matches proj-fact via description; fb-pref has no match
    # but should still be included because type=feedback.
    assert "fb-pref.md" in selected
    assert "proj-fact.md" in selected


def test_feedback_injected_with_empty_recent():
    """Even with no recent user text, feedback memories are injected."""
    memory.write_memory_file("fb-empty", "feedback", "global constraint", "never delete files")
    selected = memory.select_relevant_memories([])
    assert "fb-empty.md" in selected


def test_feedback_injected_no_keyword_match(monkeypatch):
    """Feedback memory injected even when no keywords match at all."""
    memory.write_memory_file("fb-unrelated", "feedback", "always respond in Chinese", "respond in chinese always")
    memory.write_memory_file("ref-lib", "reference", "a library doc", "some library documentation")

    # Force keyword fallback.
    def _fail(*a, **kw):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(memory.client.chat.completions, "create", _fail)

    selected = memory.select_relevant_memories(
        [{"role": "user", "content": "tell me about apples"}])

    # fb-unrelated should be included (feedback always-inject); ref-lib should not.
    assert "fb-unrelated.md" in selected
    assert "ref-lib.md" not in selected


# --------------------------------------------------------------------------- #
# Body participates in keyword matching
# --------------------------------------------------------------------------- #

def test_body_keyword_matching(monkeypatch):
    """Keyword that only appears in body (not name/description) still matches."""
    memory.write_memory_file(
        "body-match", "project", "deployment info",
        "We deploy using kubernetes clusters on AWS")

    # Force keyword fallback.
    def _fail(*a, **kw):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(memory.client.chat.completions, "create", _fail)

    selected = memory.select_relevant_memories(
        [{"role": "user", "content": "how do we handle kubernetes"}])

    assert "body-match.md" in selected


def test_body_not_matching_excludes(monkeypatch):
    """Memory whose name/description/body don't contain keywords is excluded."""
    memory.write_memory_file("no-match", "project", "unrelated thing", "completely different topic here")

    def _fail(*a, **kw):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(memory.client.chat.completions, "create", _fail)

    selected = memory.select_relevant_memories(
        [{"role": "user", "content": "tell me about kubernetes deployment"}])

    assert "no-match.md" not in selected


# --------------------------------------------------------------------------- #
# Type priority in catalog (LLM path)
# --------------------------------------------------------------------------- #

def test_catalog_includes_type_tag(monkeypatch):
    """The LLM catalog must include [type] tags for priority awareness."""
    memory.write_memory_file("cat-user", "user", "user pref", "body")
    memory.write_memory_file("cat-feedback", "feedback", "feedback item", "body")
    memory.write_memory_file("cat-ref", "reference", "a reference", "body")

    captured = {}

    class FakeMsg:
        content = "[]"
    class FakeChoice:
        message = FakeMsg()
    class FakeResp:
        choices = [FakeChoice()]

    def fake_create(*a, **kw):
        captured["prompt"] = kw.get("messages", [{}])[0].get("content", "")
        return FakeResp()

    monkeypatch.setattr(memory.client.chat.completions, "create", fake_create)

    memory.select_relevant_memories([{"role": "user", "content": "hello"}])
    prompt = captured.get("prompt", "")
    assert "[user]" in prompt
    assert "[feedback]" in prompt
    assert "[reference]" in prompt
    assert "Priority" in prompt or "priority" in prompt


def test_feedback_merged_on_llm_select(monkeypatch):
    """When LLM returns a selection, feedback memories are merged in."""
    memory.write_memory_file("llm-selected", "project", "a project", "body")
    memory.write_memory_file("llm-feedback", "feedback", "a feedback", "body")

    # Find the index of llm-selected dynamically (other test files may exist).
    files = memory.list_memory_files()
    target_idx = None
    feedback_name = None
    for i, f in enumerate(files):
        if f["filename"] == "llm-selected.md":
            target_idx = i
        if f["filename"] == "llm-feedback.md":
            feedback_name = f["filename"]

    class FakeMsg:
        content = f"[{target_idx}]"
    class FakeChoice:
        message = FakeMsg()
    class FakeResp:
        choices = [FakeChoice()]

    monkeypatch.setattr(memory.client.chat.completions, "create", lambda *a, **kw: FakeResp())

    selected = memory.select_relevant_memories(
        [{"role": "user", "content": "show me the project"}])

    # LLM selected llm-selected; feedback should be merged in.
    assert "llm-selected.md" in selected
    assert "llm-feedback.md" in selected


# --------------------------------------------------------------------------- #
# max_items respected with feedback
# --------------------------------------------------------------------------- #

def test_max_items_respected_with_feedback(monkeypatch):
    """feedback memories respect max_items limit."""
    for i in range(5):
        memory.write_memory_file(f"max-fb-{i}", "feedback", f"fb {i}", f"body {i}")

    def _fail(*a, **kw):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(memory.client.chat.completions, "create", _fail)

    selected = memory.select_relevant_memories(
        [{"role": "user", "content": "something"}], max_items=3)
    assert len(selected) <= 3
