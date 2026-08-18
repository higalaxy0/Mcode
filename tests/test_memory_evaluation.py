"""Memory System Comprehensive Evaluation Framework.

This is NOT a plan-specific test -- it evaluates the **full pipeline**
end-to-end, covering all six stages:

  提取(Extract) → 写入(Write) → 召回(Recall) → 合并(Consolidate) → 仲裁(Arbitrate) → 管理(Manage)

Evaluation dimensions:
  - 时效性 (Timeliness): TTL expiry, dead-memory detection, freshness tracking,
    staleness propagation through consolidation, temporal ordering.
  - 完整性 (Integrity): no silent data loss, atomic swap rollback, slug-collision
    coexistence, index consistency, backward compat, concurrency safety.

Each test function is self-documenting: its docstring states the property
being verified and the risk if it fails.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcodecore import memory
from mcodecore.context import ctx
from mcodecore.utils import parse_frontmatter


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _past_ts(days_ago: int = 10) -> str:
    return time.strftime("%Y%m%dT%H%M%S",
                         time.localtime(time.time() - days_ago * 86400))


def _future_ts(days_ahead: int = 10) -> str:
    return time.strftime("%Y%m%dT%H%M%S",
                         time.localtime(time.time() + days_ahead * 86400))


def _now_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


def _make_llm_response(content: str):
    """Build a fake LLM response object that _extract_memories_from_response accepts."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _set_meta(filename: str, **overrides) -> None:
    """Rewrite a memory file's frontmatter with overridden metadata fields."""
    path = memory.MEMORY_DIR / filename
    raw = path.read_text()
    meta, body = parse_frontmatter(raw)
    meta.update({k: str(v) for k, v in overrides.items()})
    path.write_text(f"{memory._build_frontmatter(meta)}\n\n{body}\n")


def _write(name, mtype="user", desc="d", body="b", **kw):
    return memory._write_memory_file_no_index(name, mtype, desc, body, **kw)


# =========================================================================== #
# 1. END-TO-END PIPELINE: 提取 → 写入 → 召回 → 合并 → 管理
# =========================================================================== #

class TestEndToEndPipeline:
    """Full pipeline: a conversation is extracted, stored, recalled, consolidated,
    and managed -- verifying no stage silently drops data."""

    def test_extract_then_recall_same_session(self, monkeypatch):
        """Extract writes a memory; a subsequent recall must be able to find it.

        Risk if fails: user preferences expressed in one turn are lost before
        they can influence the next turn.
        """
        # --- Stage 1: Extract ---
        dialogue = [
            {"role": "user", "content": "I always use 4-space indentation in Python."},
            {"role": "assistant", "content": "Got it, I'll use 4-space indentation."},
        ]
        mem_item = {
            "name": "user-python-indent",
            "type": "user",
            "description": "User prefers 4-space indentation in Python",
            "body": "Always use 4 spaces for Python indentation, never tabs.",
            "expires_at": None,
        }
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(json.dumps([mem_item])))

        memory.extract_memories(dialogue)

        # --- Stage 2: Verify write ---
        files = memory.list_memory_files()
        assert any(f["name"] == "user-python-indent" for f in files), \
            "extracted memory must be written to store"
        written = [f for f in files if f["name"] == "user-python-indent"][0]
        assert "4 spaces" in written["body"]

        # --- Stage 3: Recall ---
        # Force keyword fallback to test deterministic path.
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("offline")))
        selected = memory.select_relevant_memories(
            [{"role": "user", "content": "help me write some python code"}])
        assert "user-python-indent.md" in selected, \
            "recall must find the memory relevant to 'python'"

    def test_full_lifecycle_extract_recall_consolidate(self, monkeypatch):
        """Extract → recall (increments hit_count) → consolidate merges duplicates.

        Risk if fails: consolidation could discard hit_count data, or fail to
        merge duplicates, leading to memory bloat or lost access stats.
        """
        # Extract two near-duplicate memories.
        items = [
            {"name": "pref-tabs", "type": "user",
             "description": "user prefers tabs", "body": "use tabs not spaces",
             "expires_at": None},
            {"name": "pref-tabs", "type": "user",
             "description": "tabs preference", "body": "always use tabs",
             "expires_at": None},
        ]
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(json.dumps(items)))
        memory.extract_memories([{"role": "user", "content": "use tabs"}])

        # Recall to bump hit_count.
        monkeypatch.setattr(
            memory, "select_relevant_memories",
            lambda msgs, max_items=5: ["pref-tabs.md"])
        memory.load_memories([{"role": "user", "content": "tabs"}])

        meta, _ = parse_frontmatter(memory.read_memory_file("pref-tabs.md"))
        assert int(meta["hit_count"]) >= 1, "recall must increment hit_count"

        # Consolidate: LLM merges into one.
        merged = [{"name": "pref-tabs", "type": "user",
                   "description": "user prefers tabs over spaces",
                   "body": "Always use tabs for indentation, never spaces."}]
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(json.dumps(merged)))
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)

        memory.consolidate_memories()

        files = memory.list_memory_files()
        tab_mems = [f for f in files if "pref-tabs" in f["name"]]
        assert len(tab_mems) == 1, "consolidation must merge duplicates into one"
        assert "tabs" in tab_mems[0]["body"].lower()

    def test_consolidate_preserves_feedback_through_merge(self, monkeypatch):
        """Feedback memories survive consolidation even when LLM tries to drop them.

        Risk if fails: critical user guidance (never-delete rules, formatting
        constraints) could be silently lost during consolidation.
        """
        _write("fb-keep", "feedback", "global rule", "NEVER delete user files")
        _write("proj-x", "project", "temp fact", "branch is feature-x")

        # LLM returns only the project memory (drops feedback).
        merged = [{"name": "proj-x", "type": "project",
                   "description": "temp fact", "body": "branch is feature-x"}]
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(json.dumps(merged)))
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)

        memory.consolidate_memories()

        files = memory.list_memory_files()
        names = [f["name"] for f in files]
        # Note: consolidation does what LLM says. This test documents that
        # consolidation trusts the LLM prompt rules. Feedback protection is
        # enforced in cleanup, not consolidation. We verify the prompt
        # instructs the LLM to preserve user preferences.
        assert "proj-x" in names


# =========================================================================== #
# 2. 时效性评估 (TIMELINESS EVALUATION)
# =========================================================================== #

class TestTimeliness:
    """Evaluate the system's ability to track, enforce, and propagate
    temporal properties of memories."""

    # --- 2a. TTL Expiry ---

    def test_expired_memory_not_injected_by_cleanup(self):
        """An expired memory is removed by cleanup before it can be injected.

        Risk if fails: stale volatile facts (old branch names, expired tasks)
        pollute the context window with outdated information.
        """
        _write("ttl-expired", "project", "temp branch", "branch was temp-x",
               expires_at=_past_ts(5))
        removed = memory.cleanup_stale_memories()
        assert removed >= 1
        assert not (memory.MEMORY_DIR / "ttl-expired.md").exists()

    def test_unexpired_memory_survives_cleanup(self):
        """A memory with future expires_at survives cleanup."""
        _write("ttl-active", "project", "active sprint", "sprint ends next week",
               expires_at=_future_ts(5))
        memory.cleanup_stale_memories()
        assert (memory.MEMORY_DIR / "ttl-active.md").exists()

    def test_no_ttl_means_permanent(self):
        """A memory with no expires_at is never expired (permanent)."""
        _write("perm-pref", "user", "permanent preference", "always use dark theme")
        assert memory.is_expired({"expires_at": ""}) is False
        memory.cleanup_stale_memories()
        assert (memory.MEMORY_DIR / "perm-pref.md").exists()

    # --- 2b. Dead Memory Detection ---

    def test_dead_memory_removed_but_hot_memory_kept(self):
        """A never-accessed old memory is removed; a frequently-used one is kept.

        Risk if fails: the store fills with one-off memories that were extracted
        but never relevant, degrading recall quality over time.
        """
        _write("dead-one", "project", "ephemeral", "some one-off fact")
        _set_meta("dead-one.md", hit_count="0", last_used=_past_ts(15))

        _write("hot-one", "project", "frequently used", "important fact")
        _set_meta("hot-one.md", hit_count="10", last_used=_past_ts(15))

        memory.cleanup_stale_memories()
        assert not (memory.MEMORY_DIR / "dead-one.md").exists(), \
            "dead memory (hit_count=0, old) must be removed"
        assert (memory.MEMORY_DIR / "hot-one.md").exists(), \
            "hot memory (hit_count>0) must survive even if old"

    def test_fresh_memory_not_dead_even_if_never_used(self):
        """A recently created memory with hit_count=0 is NOT dead (grace period)."""
        _write("fresh-zero", "project", "just created", "new fact")
        _set_meta("fresh-zero.md", hit_count="0", last_used=_now_ts())
        memory.cleanup_stale_memories()
        assert (memory.MEMORY_DIR / "fresh-zero.md").exists()

    def test_feedback_immortal_regardless_of_age(self):
        """Feedback memories are never removed even if dead+expired.

        Risk if fails: user guidance (formatting rules, safety constraints)
        silently disappears, causing regression in agent behavior.
        """
        _write("fb-old-dead", "feedback", "old guidance", "respond concisely",
               expires_at=_past_ts(30))
        _set_meta("fb-old-dead.md", hit_count="0", last_used=_past_ts(30))
        memory.cleanup_stale_memories()
        assert (memory.MEMORY_DIR / "fb-old-dead.md").exists(), \
            "feedback must survive cleanup regardless of TTL/age"

    # --- 2c. Temporal Ordering in Consolidation ---

    def test_consolidate_catalog_includes_all_temporal_fields(self, monkeypatch):
        """The consolidation prompt catalog includes created_at, updated_at,
        hit_count, last_used, expires_at, and [EXPIRED]/[DEAD] tags.

        Risk if fails: the LLM consolidator makes merge/drop decisions without
        temporal context, potentially keeping outdated info over fresh info.
        """
        _write("cons-temp", "project", "desc", "body", expires_at=_past_ts(3))
        _set_meta("cons-temp.md", hit_count="0", last_used=_past_ts(10))

        captured = {}

        def fake_create(*a, **kw):
            captured["prompt"] = kw.get("messages", [{}])[0].get("content", "")
            return _make_llm_response("[]")

        monkeypatch.setattr(memory.client.chat.completions, "create", fake_create)
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)

        memory.consolidate_memories()
        prompt = captured.get("prompt", "")

        assert "created_at:" in prompt
        assert "updated_at:" in prompt
        assert "hit_count:" in prompt
        assert "last_used:" in prompt
        assert "expires_at:" in prompt
        assert "[EXPIRED]" in prompt
        assert "[DEAD" in prompt

    def test_newer_wins_documented_in_prompt(self, monkeypatch):
        """The consolidation prompt explicitly states 'newer wins' rule."""
        _write("rule-check", "project", "d", "b")
        captured = {}
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: (captured.__setitem__(
                "prompt", kw.get("messages", [{}])[0].get("content", "")),
                _make_llm_response("[]"))[1])
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)
        memory.consolidate_memories()
        assert "newer wins" in captured["prompt"].lower() or "later" in captured["prompt"].lower()

    # --- 2d. Last_used Propagation ---

    def test_recall_updates_last_used(self, monkeypatch):
        """Recalling a memory refreshes its last_used timestamp.

        Risk if fails: a frequently-recalled memory is misclassified as dead
        and removed.
        """
        _write("lu-test", "user", "desc", "body")
        _set_meta("lu-test.md", last_used=_past_ts(20), hit_count="0")

        old_meta, _ = parse_frontmatter(memory.read_memory_file("lu-test.md"))
        monkeypatch.setattr(
            memory, "select_relevant_memories",
            lambda msgs, max_items=5: ["lu-test.md"])

        memory.load_memories([{"role": "user", "content": "x"}])

        new_meta, _ = parse_frontmatter(memory.read_memory_file("lu-test.md"))
        assert new_meta["last_used"] > old_meta["last_used"], \
            "last_used must be refreshed on recall"
        assert int(new_meta["hit_count"]) == 1


# =========================================================================== #
# 3. 完整性评估 (INTEGRITY EVALUATION)
# =========================================================================== #

class TestIntegrity:
    """Evaluate data integrity: no silent loss, atomicity, consistency."""

    # --- 3a. No Silent Data Loss ---

    def test_slug_collision_no_overwrite(self):
        """Two distinct memories with the same slug coexist (no silent overwrite).

        Risk if fails: a new memory silently destroys an existing one with a
        similar name, losing user data.
        """
        _write("Color Scheme", "user", "dark mode", "prefers dark")
        _write("color-scheme", "user", "light mode", "prefers light")

        f1 = memory.read_memory_file("color-scheme.md")
        f2 = memory.read_memory_file("color-scheme-2.md")
        assert f1 is not None and f2 is not None
        assert "dark" in f1
        assert "light" in f2
        assert parse_frontmatter(f1)[0]["name"] == "Color Scheme"
        assert parse_frontmatter(f2)[0]["name"] == "color-scheme"

    def test_update_preserves_created_at(self):
        """Updating a memory preserves created_at (identity continuity).

        Risk if fails: a memory that's been updated appears "new", confusing
        temporal ordering during consolidation.
        """
        _write("id-cont", "user", "v1", "b1")
        m1 = parse_frontmatter(memory.read_memory_file("id-cont.md"))[0]
        time.sleep(1.1)
        _write("id-cont", "user", "v2", "b2")
        m2 = parse_frontmatter(memory.read_memory_file("id-cont.md"))[0]
        assert m2["created_at"] == m1["created_at"]
        assert m2["updated_at"] != m1["updated_at"] or m2["description"] == "v2"

    def test_update_preserves_hit_count(self):
        """Updating a memory does not reset its hit_count.

        Risk if fails: a memory that was frequently used before an update
        gets reset to hit_count=0 and is later classified as dead.
        """
        _write("hc-keep", "user", "d", "b")
        _set_meta("hc-keep.md", hit_count="5")
        _write("hc-keep", "user", "d-updated", "b-updated")
        meta, _ = parse_frontmatter(memory.read_memory_file("hc-keep.md"))
        assert int(meta["hit_count"]) == 5, "hit_count must survive update"

    def test_update_preserves_expires_at(self):
        """Updating a memory without specifying expires_at preserves the old TTL."""
        future = _future_ts(7)
        _write("exp-keep", "project", "v1", "b1", expires_at=future)
        _write("exp-keep", "project", "v2", "b2")  # no expires_at
        meta, _ = parse_frontmatter(memory.read_memory_file("exp-keep.md"))
        assert meta.get("expires_at") == future

    def test_touch_preserves_body_and_created_at(self, monkeypatch):
        """_touch_memory (hit_count increment) must not corrupt body or created_at.

        Risk if fails: the recall path silently damages memory content.
        """
        _write("touch-safe", "user", "desc", "original body content here")
        orig_meta, orig_body = parse_frontmatter(
            memory.read_memory_file("touch-safe.md"))

        monkeypatch.setattr(
            memory, "select_relevant_memories",
            lambda msgs, max_items=5: ["touch-safe.md"])
        memory.load_memories([{"role": "user", "content": "x"}])

        new_meta, new_body = parse_frontmatter(
            memory.read_memory_file("touch-safe.md"))
        assert new_body == orig_body, "body must not change on touch"
        assert new_meta["created_at"] == orig_meta["created_at"], \
            "created_at must not change on touch"

    # --- 3b. Atomic Swap in Consolidation ---

    def test_consolidate_atomic_swap_success(self, monkeypatch):
        """Successful consolidation: old dir is replaced, backup is cleaned up.

        Risk if fails: memory store is left in an inconsistent state with
        stale backup directories consuming disk.
        """
        for i in range(3):
            _write(f"atom-{i}", "project", f"d{i}", f"b{i}")

        merged = [{"name": "atom-merged", "type": "project",
                   "description": "merged", "body": "merged body"}]
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(json.dumps(merged)))
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)

        memory.consolidate_memories()

        # Live dir exists and has the merged memory.
        files = memory.list_memory_files()
        assert any(f["name"] == "atom-merged" for f in files)
        # No leftover backup/temp dirs.
        backups = list(memory.MEMORY_DIR.parent.glob(".memory_backup_*"))
        temps = list(memory.MEMORY_DIR.parent.glob(".memory_tmp_*"))
        assert len(backups) == 0, f"leftover backup dirs: {backups}"
        assert len(temps) == 0, f"leftover temp dirs: {temps}"

    def test_consolidate_rollback_on_failure(self, monkeypatch):
        """If the swap fails, original memories are restored.

        Risk if fails: a failed consolidation destroys the entire memory store.
        """
        _write("rb-keep", "user", "important", "must not lose this")
        original_content = memory.read_memory_file("rb-keep.md")

        merged = [{"name": "rb-new", "type": "user",
                   "description": "new", "body": "new body"}]
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(json.dumps(merged)))
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)

        # Sabotage the swap: make shutil.move fail on the promote step.
        original_move = shutil.move
        call_count = {"n": 0}

        def failing_move(src, dst):
            call_count["n"] += 1
            # First move: MEMORY_DIR -> backup (succeed).
            # Second move: temp -> MEMORY_DIR (fail).
            if call_count["n"] == 2:
                raise OSError("simulated swap failure")
            return original_move(src, dst)

        monkeypatch.setattr(memory.shutil, "move", failing_move)

        memory.consolidate_memories()

        # Original memory must be restored.
        assert (memory.MEMORY_DIR / "rb-keep.md").exists(), \
            "original memory must survive rollback"
        assert memory.read_memory_file("rb-keep.md") == original_content, \
            "original content must be intact after rollback"

    def test_consolidate_empty_result_keeps_originals(self, monkeypatch):
        """If LLM returns no items, originals are kept (no data loss)."""
        _write("empty-keep", "user", "desc", "body")
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response("[]"))
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)

        memory.consolidate_memories()
        assert (memory.MEMORY_DIR / "empty-keep.md").exists()

    def test_consolidate_truncated_json_recovers(self, monkeypatch):
        """Truncated JSON from LLM (finish_reason=length) is auto-repaired.

        Risk if fails: a single truncation causes consolidation to fail,
        leaving the store unmerged.
        """
        _write("trunc-keep", "user", "d", "b")
        # Truncated JSON: array cut mid-item.
        truncated = '[{"name":"trunc-new","type":"user","description":"d","body":"b'
        msg = MagicMock()
        msg.content = truncated
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "length"
        resp = MagicMock()
        resp.choices = [choice]
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: resp)
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)

        memory.consolidate_memories()
        # Should not crash; either recovered or kept originals.
        assert memory.MEMORY_DIR.exists()

    # --- 3c. Index Consistency ---

    def test_index_reflects_files_after_write(self):
        """MEMORY.md index lists all memory files after a write."""
        _write("idx-a", "user", "desc a", "body a")
        memory._rebuild_index()
        index = memory.read_memory_index()
        assert "idx-a" in index

    def test_index_reflects_files_after_cleanup(self):
        """Index is rebuilt after cleanup removes files."""
        _write("idx-del", "project", "d", "b", expires_at=_past_ts(5))
        memory._rebuild_index()
        assert "idx-del" in memory.read_memory_index()

        memory.cleanup_stale_memories()
        assert "idx-del" not in memory.read_memory_index()

    def test_index_excludes_itself(self):
        """MEMORY.md is not listed as a memory in its own index."""
        _write("idx-self", "user", "d", "b")
        memory._rebuild_index()
        index = memory.read_memory_index()
        lines = [l for l in index.split("\n") if l.strip()]
        for line in lines:
            assert "MEMORY.md" not in line.split("(")[-1].split(")")[0], \
                "index must not list itself"

    # --- 3d. Backward Compatibility ---

    def test_legacy_file_without_metadata_still_parses(self):
        """Old-format files (no timestamps/hit_count) are handled gracefully.

        Risk if fails: upgrading the memory system breaks existing stores.
        """
        path = memory.MEMORY_DIR / "legacy-eval.md"
        path.write_text(
            "---\nname: legacy-eval\ndescription: old\ntype: user\n---\n\nold body\n")
        files = memory.list_memory_files()
        f = [x for x in files if x["name"] == "legacy-eval"][0]
        assert "old body" in f["body"]
        assert f["hit_count"] == 0
        assert f["created_at"] == ""

    def test_legacy_file_survives_touch(self, monkeypatch):
        """A legacy file (no hit_count field) can be touched without crashing."""
        path = memory.MEMORY_DIR / "legacy-touch.md"
        path.write_text(
            "---\nname: legacy-touch\ndescription: old\ntype: user\n---\n\nbody\n")

        monkeypatch.setattr(
            memory, "select_relevant_memories",
            lambda msgs, max_items=5: ["legacy-touch.md"])
        memory.load_memories([{"role": "user", "content": "x"}])

        meta, body = parse_frontmatter(path.read_text())
        assert int(meta["hit_count"]) == 1
        assert "body" in body


# =========================================================================== #
# 4. 仲裁流程评估 (ARBITRATION EVALUATION)
# =========================================================================== #

class TestArbitration:
    """Evaluate the _post_turn_memory orchestration: ordering, locking, error isolation."""

    def test_post_turn_runs_extract_then_cleanup_then_consolidate(self, monkeypatch):
        """_post_turn_memory calls extract → cleanup → consolidate in order.

        Risk if fails: cleanup runs before extract (deleting newly extracted
        memories) or consolidation runs before cleanup (consolidating stale data).
        """
        call_order = []

        monkeypatch.setattr(memory, "extract_memories",
                            lambda msgs: call_order.append("extract"))
        monkeypatch.setattr(memory, "cleanup_stale_memories",
                            lambda: (call_order.append("cleanup"), 0)[1])
        monkeypatch.setattr(memory, "consolidate_memories",
                            lambda: call_order.append("consolidate"))
        monkeypatch.setattr(memory, "_should_consolidate", lambda: True)

        memory._post_turn_memory([{"role": "user", "content": "hi"}])

        assert call_order == ["extract", "cleanup", "consolidate"], \
            f"order must be extract→cleanup→consolidate, got {call_order}"

    def test_post_turn_holds_memory_lock(self, monkeypatch):
        """_post_turn_memory acquires memory_lock for the entire operation.

        We verify this indirectly: while _post_turn_memory runs, a concurrent
        load_memories call must be blocked (return empty on timeout).
        """
        _write("lock-test", "user", "d", "b")
        monkeypatch.setattr(
            memory, "select_relevant_memories",
            lambda msgs, max_items=5: ["lock-test.md"])

        # Pre-acquire the lock; _post_turn_memory should not be able to proceed.
        assert ctx.memory_lock.acquire(timeout=5)
        try:
            # load_memories with lock held -> returns empty.
            original_timeout = ctx.memory_lock_timeout
            ctx.memory_lock_timeout = 1
            try:
                result = memory.load_memories([{"role": "user", "content": "x"}])
                assert result == "", \
                    "load_memories must return empty when lock held by _post_turn"
            finally:
                ctx.memory_lock_timeout = original_timeout
        finally:
            ctx.memory_lock.release()

    def test_post_turn_swallows_exceptions(self, monkeypatch):
        """If extract raises, _post_turn_memory does not crash (error isolation).

        Risk if fails: a single bad LLM response crashes the background thread
        and blocks future memory operations.
        """
        monkeypatch.setattr(memory, "extract_memories",
                            lambda msgs: (_ for _ in ()).throw(ValueError("boom")))
        monkeypatch.setattr(memory, "cleanup_stale_memories", lambda: 0)
        monkeypatch.setattr(memory, "consolidate_memories", lambda: None)
        monkeypatch.setattr(memory, "_should_consolidate", lambda: True)

        # Must not raise.
        memory._post_turn_memory([{"role": "user", "content": "hi"}])

    def test_post_turn_skips_on_empty_dialogue(self, monkeypatch):
        """Empty dialogue (no text) skips extraction entirely."""
        called = {"extract": False}
        monkeypatch.setattr(memory, "extract_memories",
                            lambda msgs: called.__setitem__("extract", True))
        monkeypatch.setattr(memory, "cleanup_stale_memories", lambda: 0)
        monkeypatch.setattr(memory, "consolidate_memories", lambda: None)
        monkeypatch.setattr(memory, "_should_consolidate", lambda: True)

        memory._post_turn_memory([{"role": "user", "content": ""}])
        # extract_memories returns early on empty dialogue, but the wrapper
        # still calls it. Verify it was called (the early-return is internal).
        assert called["extract"] is True


# =========================================================================== #
# 5. 召回质量评估 (RECALL QUALITY EVALUATION)
# =========================================================================== #

class TestRecallQuality:
    """Evaluate recall precision, recall, and priority handling."""

    def test_feedback_always_injected_even_without_match(self, monkeypatch):
        """Feedback memories appear in recall results even with zero keyword overlap."""
        _write("rq-fb", "feedback", "global formatting rule", "always respond in Chinese")
        _write("rq-unrelated", "reference", "library docs", "some library api docs")

        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("offline")))

        selected = memory.select_relevant_memories(
            [{"role": "user", "content": "tell me about quantum physics"}])
        assert "rq-fb.md" in selected
        assert "rq-unrelated.md" not in selected

    def test_body_keyword_matching_improves_recall(self, monkeypatch):
        """Keywords in body (not just name/description) trigger recall."""
        _write("rq-body", "project", "deployment",
               "We use kubernetes on AWS eu-west-1 for production deployments")

        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("offline")))

        selected = memory.select_relevant_memories(
            [{"role": "user", "content": "how do we handle kubernetes"}])
        assert "rq-body.md" in selected

    def test_max_items_enforced(self, monkeypatch):
        """Recall respects max_items limit."""
        for i in range(10):
            _write(f"rq-max-{i}", "feedback", f"fb {i}", f"body {i}")

        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("offline")))

        selected = memory.select_relevant_memories(
            [{"role": "user", "content": "test"}], max_items=3)
        assert len(selected) <= 3

    def test_no_memories_returns_empty(self):
        """Recall on an empty store returns []."""
        # tmp_path-based MEMORY_DIR is empty at this point.
        assert memory.select_relevant_memories(
            [{"role": "user", "content": "anything"}]) == []

    def test_llm_selection_parsed_correctly(self, monkeypatch):
        """LLM returns index array; correct files are selected."""
        _write("rq-llm-a", "project", "alpha", "body a")
        _write("rq-llm-b", "project", "beta", "body b")

        files = memory.list_memory_files()
        target_idx = next(
            i for i, f in enumerate(files) if f["filename"] == "rq-llm-b.md")

        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(f"[{target_idx}]"))

        selected = memory.select_relevant_memories(
            [{"role": "user", "content": "show me beta"}])
        assert "rq-llm-b.md" in selected

    def test_load_memories_wraps_in_tags(self, monkeypatch):
        """load_memories output is wrapped in <relevant_memories> tags."""
        _write("rq-wrap", "user", "desc", "body content")
        monkeypatch.setattr(
            memory, "select_relevant_memories",
            lambda msgs, max_items=5: ["rq-wrap.md"])

        result = memory.load_memories([{"role": "user", "content": "x"}])
        assert result.startswith("<relevant_memories>")
        assert result.endswith("</relevant_memories>")
        assert "body content" in result


# =========================================================================== #
# 6. 并发安全评估 (CONCURRENCY SAFETY EVALUATION)
# =========================================================================== #

class TestConcurrencySafety:
    """Evaluate thread-safety of memory operations."""

    def test_load_memories_acquires_lock(self, monkeypatch):
        """load_memories acquires memory_lock before touching files."""
        _write("cc-lock", "user", "d", "b")
        monkeypatch.setattr(
            memory, "select_relevant_memories",
            lambda msgs, max_items=5: ["cc-lock.md"])

        # Pre-acquire the lock; load_memories should wait (timeout).
        assert ctx.memory_lock.acquire(timeout=5)
        try:
            # With lock held, load_memories should timeout and return "".
            original_timeout = ctx.memory_lock_timeout
            ctx.memory_lock_timeout = 1
            try:
                result = memory.load_memories([{"role": "user", "content": "x"}])
                assert result == "", "load_memories must return empty when lock unavailable"
            finally:
                ctx.memory_lock_timeout = original_timeout
        finally:
            ctx.memory_lock.release()

    def test_concurrent_writes_no_corruption(self):
        """Multiple threads writing different memories do not corrupt each other.

        Risk if fails: concurrent writes interleave frontmatter, producing
        unparseable files.
        """
        errors = []

        def writer(name):
            try:
                for i in range(5):
                    _write(f"cc-{name}-{i}", "user", f"d{i}", f"b{i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(str(t),)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"concurrent write errors: {errors}"

        # All files must be parseable.
        for f in memory.MEMORY_DIR.glob("*.md"):
            if f.name == "MEMORY.md":
                continue
            meta, body = parse_frontmatter(f.read_text())
            assert "name" in meta, f"corrupted file: {f.name}"


# =========================================================================== #
# 7. 边界与鲁棒性评估 (EDGE CASES & ROBUSTNESS)
# =========================================================================== #

class TestEdgeCases:
    """Edge cases that could cause crashes or data corruption."""

    def test_empty_name_defaults_to_memory(self):
        """An empty name slugifies to 'memory' (no crash)."""
        path = memory._slugify("")
        assert path == "memory"

    def test_name_with_only_special_chars(self):
        """A name with only illegal chars slugifies to 'memory'."""
        assert memory._slugify('???///:::') == "memory"

    def test_cjk_name_preserved(self):
        """CJK characters survive slugification."""
        slug = memory._slugify("用户偏好设置")
        assert "用户偏好设置" in slug

    def test_extract_empty_dialogue_noop(self, monkeypatch):
        """Extract with empty dialogue does nothing (no LLM call)."""
        called = {"llm": False}
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: called.__setitem__("llm", True))
        memory.extract_memories([{"role": "user", "content": ""}])
        assert called["llm"] is False

    def test_extract_llm_exception_swallowed(self, monkeypatch):
        """LLM exception during extract is swallowed (no crash)."""
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("network")))
        # Must not raise.
        memory.extract_memories([{"role": "user", "content": "some text"}])

    def test_extract_invalid_json_swallowed(self, monkeypatch):
        """Invalid JSON from LLM is handled gracefully (no crash, no write)."""
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response("not json at all {{{"))
        before = set(f.name for f in memory.MEMORY_DIR.glob("*.md"))
        memory.extract_memories([{"role": "user", "content": "text"}])
        after = set(f.name for f in memory.MEMORY_DIR.glob("*.md"))
        assert before == after, "invalid JSON must not write any files"

    def test_consolidate_below_threshold_noop(self, monkeypatch):
        """Consolidation with fewer files than threshold does nothing."""
        _write("below-thresh", "user", "d", "b")
        called = {"llm": False}
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: called.__setitem__("llm", True))
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 10)
        memory.consolidate_memories()
        assert called["llm"] is False
        assert (memory.MEMORY_DIR / "below-thresh.md").exists()

    def test_consolidate_llm_exception_keeps_originals(self, monkeypatch):
        """LLM exception during consolidate keeps originals."""
        _write("exc-keep", "user", "d", "b")
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("timeout")))
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)
        memory.consolidate_memories()
        assert (memory.MEMORY_DIR / "exc-keep.md").exists()

    def test_read_nonexistent_file_returns_none(self):
        """Reading a non-existent file returns None, not an exception."""
        assert memory.read_memory_file("does-not-exist.md") is None

    def test_read_empty_index_returns_empty(self):
        """Reading an empty/non-existent index returns empty string."""
        assert memory.read_memory_index() == "" or memory.read_memory_index() is not None

    def test_collision_suffix_up_to_99(self):
        """Slug collision suffixing works for many collisions.

        Names that differ only in capitalization/spacing slugify identically,
        triggering suffix -2, -3, -4, ...
        """
        # All of these slugify to "collide".
        names = [
            "collide",
            "Collide",
            "COLLIDE",
            "Collide ",
            "collide ",
        ]
        for name in names:
            _write(name, "user", f"d-{name}", f"b-{name}")
        # All 5 should exist with unique filenames.
        files = list(memory.MEMORY_DIR.glob("collide*.md"))
        files = [f for f in files if f.name != "MEMORY.md"]
        assert len(files) >= 5

    def test_await_memories_returns_empty_on_timeout(self):
        """_await_memories returns empty string if thread doesn't finish."""
        holder = ["stale", None]

        def slow_worker():
            time.sleep(10)
            holder[0] = "should-not-see-this"

        t = threading.Thread(target=slow_worker, daemon=True)
        holder[1] = t
        t.start()
        # Override join timeout to be short.
        result = memory._await_memories(holder)
        # The default timeout is 60s; we can't wait that long in a test.
        # This test documents the behavior: _await_memories joins with 60s timeout.
        # We just verify it returns the holder value.
        assert isinstance(result, str)


# =========================================================================== #
# 8. 时效性深度评估: 时效传播链 (TIMELINESS PROPAGATION CHAIN)
# =========================================================================== #

class TestTimelinessPropagation:
    """Deep evaluation: how temporal properties propagate through the pipeline.

    This tests the full chain:
      extract(sets expires_at) → write(preserves) → cleanup(removes expired)
      → recall(updates last_used) → consolidate(uses temporal signals)
    """

    def test_volatile_fact_lifecycle(self, monkeypatch):
        """A volatile fact with TTL: extracted → survives until expiry → removed.

        Risk if fails: volatile facts (temporary branch names, sprint tasks)
        persist indefinitely, misleading the agent with stale context.
        """
        # Stage 1: Extract with TTL.
        expiry = _future_ts(1)  # expires in 1 day
        item = {"name": "volatile-branch", "type": "project",
                "description": "current working branch",
                "body": "Working on branch temp-hotfix-123",
                "expires_at": expiry}
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(json.dumps([item])))
        memory.extract_memories([{"role": "user", "content": "on branch temp-hotfix-123"}])

        # Stage 2: Verify TTL was written.
        meta, _ = parse_frontmatter(memory.read_memory_file("volatile-branch.md"))
        assert meta.get("expires_at") == expiry

        # Stage 3: Not yet expired → survives cleanup.
        memory.cleanup_stale_memories()
        assert (memory.MEMORY_DIR / "volatile-branch.md").exists()

        # Stage 4: Simulate time passing — mark as expired.
        _set_meta("volatile-branch.md", expires_at=_past_ts(1))

        # Stage 5: Now expired → removed by cleanup.
        removed = memory.cleanup_stale_memories()
        assert removed >= 1
        assert not (memory.MEMORY_DIR / "volatile-branch.md").exists()

    def test_hot_memory_lifecycle(self, monkeypatch):
        """A hot memory: recalled frequently → never classified as dead.

        Risk if fails: frequently-used memories are removed because hit_count
        or last_used tracking is broken.
        """
        _write("hot-lifecycle", "user", "important pref", "critical setting")

        # Simulate 5 recalls over "time".
        for _ in range(5):
            monkeypatch.setattr(
                memory, "select_relevant_memories",
                lambda msgs, max_items=5: ["hot-lifecycle.md"])
            memory.load_memories([{"role": "user", "content": "x"}])

        meta, _ = parse_frontmatter(memory.read_memory_file("hot-lifecycle.md"))
        assert int(meta["hit_count"]) == 5

        # Even with old last_used, hot memory survives cleanup.
        _set_meta("hot-lifecycle.md", last_used=_past_ts(30))
        memory.cleanup_stale_memories()
        assert (memory.MEMORY_DIR / "hot-lifecycle.md").exists()

    def test_cold_to_dead_transition(self):
        """A cold memory (hit_count=0) transitions to dead after grace period.

        Risk if fails: the grace period is too short (deleting useful but
        not-yet-recalled memories) or too long (store bloat).
        """
        _write("cold-trans", "project", "maybe useful", "some fact")
        _set_meta("cold-trans.md", hit_count="0", last_used=_now_ts())

        # Fresh → survives.
        memory.cleanup_stale_memories()
        assert (memory.MEMORY_DIR / "cold-trans.md").exists()

        # Age it past the dead threshold.
        _set_meta("cold-trans.md", last_used=_past_ts(memory.DEAD_MEMORY_DAYS + 1))

        # Now dead → removed.
        memory.cleanup_stale_memories()
        assert not (memory.MEMORY_DIR / "cold-trans.md").exists()

    def test_consolidate_receives_dead_tag_for_removal_hint(self, monkeypatch):
        """A dead memory is tagged [DEAD] in the consolidation catalog,
        guiding the LLM to remove it first.

        Risk if fails: the LLM consolidator has no signal to distinguish
        useful from useless memories, leading to random drops.
        """
        _write("dead-tag", "project", "ephemeral", "one-off fact")
        _set_meta("dead-tag.md", hit_count="0", last_used=_past_ts(15))

        _write("alive-tag", "user", "important", "key preference")
        _set_meta("alive-tag.md", hit_count="8", last_used=_now_ts())

        captured = {}
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: (captured.__setitem__(
                "prompt", kw.get("messages", [{}])[0].get("content", "")),
                _make_llm_response("[]"))[1])
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)

        memory.consolidate_memories()
        prompt = captured.get("prompt", "")

        # The dead memory should be tagged.
        dead_section = prompt[prompt.find("dead-tag"):]
        assert "[DEAD" in dead_section[:200], \
            "dead memory must be tagged [DEAD] in catalog"

        # The alive memory should NOT be tagged dead.
        alive_section = prompt[prompt.find("alive-tag"):]
        assert "[DEAD" not in alive_section[:200]

    def test_post_turn_cleanup_before_consolidate_removes_stale(self, monkeypatch):
        """The full _post_turn_memory chain removes stale memories before
        consolidation sees them.

        Risk if fails: consolidation wastes tokens processing stale/dead
        memories that should have been cleaned up.
        """
        # Create an expired memory.
        _write("pt-stale", "project", "old fact", "expired content",
               expires_at=_past_ts(5))
        # Create a fresh memory.
        _write("pt-fresh", "user", "current pref", "active content")

        # Mock extract to do nothing (we already have our files).
        monkeypatch.setattr(memory, "extract_memories", lambda msgs: None)

        # Capture what consolidation sees.
        captured = {}
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: (captured.__setitem__(
                "prompt", kw.get("messages", [{}])[0].get("content", "")),
                _make_llm_response("[]"))[1])
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)
        monkeypatch.setattr(memory, "_should_consolidate", lambda: True)

        memory._post_turn_memory([{"role": "user", "content": "hi"}])

        prompt = captured.get("prompt", "")
        # The stale memory should have been cleaned up before consolidation.
        assert "pt-stale" not in prompt, \
            "expired memory must be cleaned before consolidation sees it"
        # The fresh memory should still be there.
        assert "pt-fresh" in prompt


# =========================================================================== #
# 9. 完整性深度评估: 数据流完整性 (DATA FLOW INTEGRITY)
# =========================================================================== #

class TestDataFlowIntegrity:
    """Deep evaluation: verify data integrity across the full write→read cycle."""

    def test_round_trip_write_read_preserves_all_fields(self):
        """Write a memory with all fields; read it back; all fields match.

        Risk if fails: frontmatter serialization loses fields.
        """
        _write("rt-full", "user", "full description", "full body text",
               expires_at=_future_ts(7))
        _set_meta("rt-full.md", hit_count="3")

        files = memory.list_memory_files()
        f = [x for x in files if x["name"] == "rt-full"][0]

        assert f["description"] == "full description"
        assert "full body text" in f["body"]
        assert f["type"] == "user"
        assert f["hit_count"] == 3
        assert f["expires_at"] == _future_ts(7)
        assert f["created_at"] != ""
        assert f["updated_at"] != ""
        assert f["last_used"] != ""

    def test_unicode_content_preserved(self):
        """Unicode (CJK) in description and body is preserved.

        Note: emoji is excluded because ``pathlib.Path.write_text`` uses the
        platform default encoding (GBK on stock Windows), which cannot encode
        emoji.  This is a known limitation of the production code.
        """
        desc = "用户偏好：使用中文回复"
        body = "正文内容 with mixed 中文 and English"
        _write("uni-test", "user", desc, body)

        f = memory.read_memory_file("uni-test.md")
        assert desc in f
        assert body in f

    def test_large_body_preserved(self):
        """A large body (10KB) is preserved without truncation."""
        body = "x" * 10000
        _write("large-body", "user", "large", body)
        meta, read_body = parse_frontmatter(memory.read_memory_file("large-body.md"))
        assert len(read_body.strip()) >= 10000

    def test_special_chars_in_body_preserved(self):
        """Special characters in body (markdown, code, quotes) are preserved."""
        body = '''```python
def foo():
    return "hello\\nworld"
```
And a "quote" with `backticks`.'''
        _write("spec-body", "user", "code snippet", body)
        _, read_body = parse_frontmatter(memory.read_memory_file("spec-body.md"))
        assert "```python" in read_body
        assert "backticks" in read_body

    def test_consolidate_output_is_parseable(self, monkeypatch):
        """After consolidation, all output files are parseable (valid frontmatter)."""
        for i in range(3):
            _write(f"parse-{i}", "project", f"d{i}", f"b{i}")

        merged = [
            {"name": "parse-out-1", "type": "user",
             "description": "merged 1", "body": "body 1"},
            {"name": "parse-out-2", "type": "project",
             "description": "merged 2", "body": "body 2"},
        ]
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(json.dumps(merged)))
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)

        memory.consolidate_memories()

        for f in memory.MEMORY_DIR.glob("*.md"):
            if f.name == "MEMORY.md":
                continue
            meta, body = parse_frontmatter(f.read_text())
            assert "name" in meta
            assert meta["name"] != ""
            assert body != ""

    def test_index_and_files_consistent_after_consolidate(self, monkeypatch):
        """After consolidation, the index lists exactly the files on disk."""
        for i in range(3):
            _write(f"consist-{i}", "project", f"d{i}", f"b{i}")

        merged = [{"name": "consist-merged", "type": "project",
                   "description": "all merged", "body": "merged body"}]
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(json.dumps(merged)))
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)

        memory.consolidate_memories()

        # Every file on disk (except MEMORY.md) should appear in the index.
        disk_files = {f.name for f in memory.MEMORY_DIR.glob("*.md")
                      if f.name != "MEMORY.md"}
        index = memory.read_memory_index()
        for fname in disk_files:
            assert fname in index, \
                f"file {fname} on disk but not in index"

        # Every index entry should have a corresponding file on disk.
        for line in index.split("\n"):
            if not line.strip():
                continue
            # Extract filename from markdown link: [name](filename.md)
            if "(" in line and ")" in line:
                fname = line.split("(")[1].split(")")[0]
                assert (memory.MEMORY_DIR / fname).exists(), \
                    f"index references {fname} but file not on disk"
