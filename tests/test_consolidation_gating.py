"""Consolidation gating evaluation.

Tests the four-layer gate (inspired by Claude Code autoDream) that controls
when ``consolidate_memories`` actually runs:

  Gate 0 (hard limit): file count >= CONSOLIDATE_HARD_LIMIT -> force merge
  Gate 1 (count floor): file count >= CONSOLIDATE_THRESHOLD
  Gate 2 (time cooldown): enough seconds since last consolidation
  Gate 3 (activity): enough new transcripts since last consolidation
  Gate 4 (cross-process lock): .consolidate-lock must be acquirable

Additionally tests:
  - Scan-throttle cache (list_memory_files caching + invalidation)
  - State persistence (consolidation_state.json load/save round-trip)
  - Cross-process lock acquire/release/stale-detection
  - Full pipeline integrity: extract -> write -> recall -> consolidate
    with gating in place
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcodecore import memory
from mcodecore import config
from mcodecore.context import ctx


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_llm_response(content: str):
    """Build a fake LLM response object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _write(name, mtype="user", desc="d", body="b", **kw):
    """Write a memory file (no index rebuild)."""
    return memory._write_memory_file_no_index(name, mtype, desc, body, **kw)


def _seed_files(n: int, prefix: str = "m") -> None:
    """Write *n* memory files named ``prefix-0``, ``prefix-1``, ..."""
    for i in range(n):
        _write(f"{prefix}-{i}", "user", f"desc {i}", f"body {i}")


def _clear_state() -> None:
    """Remove the consolidation state file if it exists."""
    path = memory._state_file_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _clear_lock() -> None:
    """Remove the consolidation lock file if it exists."""
    path = memory._lock_file_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _make_transcript(name: str, mtime_offset: float = 0.0) -> Path:
    """Create a dummy transcript file in TRANSCRIPT_DIR.

    ``mtime_offset`` > 0 sets the file's mtime to the future (so it's "new"
    relative to *now*).  ``mtime_offset`` < 0 sets it to the past (old).
    """
    td = memory.TRANSCRIPT_DIR
    td.mkdir(exist_ok=True)
    path = td / f"transcript_{name}.jsonl"
    path.write_text('{"role":"user","content":"x"}\n', encoding="utf-8")
    ts = time.time() + mtime_offset
    os.utime(str(path), (ts, ts))
    return path


def _clear_transcripts() -> None:
    """Remove all transcript files."""
    td = memory.TRANSCRIPT_DIR
    if td.exists():
        for f in td.glob("transcript_*.jsonl"):
            try:
                f.unlink()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# 1. Scan-throttle cache (Gate 3)
# --------------------------------------------------------------------------- #

class TestScanThrottleCache:
    """list_memory_files results are cached and invalidated correctly."""

    def test_cache_returns_same_object_within_ttl(self, monkeypatch):
        """Two calls within TTL return the same cached list object."""
        monkeypatch.setattr(memory, "MEMORY_CACHE_TTL", 30)
        memory._invalidate_memory_cache()
        _write("cache-1", "user", "d", "b")

        first = memory.list_memory_files()
        second = memory.list_memory_files()
        assert first is second, "second call within TTL must return cached object"

    def test_cache_expires_after_ttl(self, monkeypatch):
        """After TTL expires, a fresh scan is performed."""
        monkeypatch.setattr(memory, "MEMORY_CACHE_TTL", 0)
        memory._invalidate_memory_cache()
        _write("cache-expire-1", "user", "d", "b")

        first = memory.list_memory_files()
        # With TTL=0, the cache is immediately stale.
        _write("cache-expire-2", "user", "d", "b")
        second = memory.list_memory_files()
        assert len(second) == len(first) + 1, \
            "new file must be visible after cache expiry"

    def test_invalidate_forces_rescan(self, monkeypatch):
        """_invalidate_memory_cache forces the next call to rescan."""
        monkeypatch.setattr(memory, "MEMORY_CACHE_TTL", 999)
        memory._invalidate_memory_cache()
        _write("cache-inval-1", "user", "d", "b")
        first = memory.list_memory_files()

        _write("cache-inval-2", "user", "d", "b")
        memory._invalidate_memory_cache()
        second = memory.list_memory_files()
        assert len(second) == len(first) + 1, \
            "invalidation must cause fresh scan"

    def test_write_invalidates_cache(self, monkeypatch):
        """write_memory_file invalidates the cache so the new file is visible."""
        monkeypatch.setattr(memory, "MEMORY_CACHE_TTL", 999)
        memory._invalidate_memory_cache()
        memory.list_memory_files()  # populate cache

        _write("cache-write-1", "user", "d", "b")
        files = memory.list_memory_files()
        names = [f["name"] for f in files]
        assert "cache-write-1" in names, \
            "write must invalidate cache so new file is visible"


# --------------------------------------------------------------------------- #
# 2. State persistence
# --------------------------------------------------------------------------- #

class TestStatePersistence:
    """consolidation_state.json load/save round-trip."""

    def test_load_returns_defaults_when_no_file(self):
        """Missing state file -> safe defaults (epoch 0)."""
        _clear_state()
        state = memory._load_consolidation_state()
        assert state["last_consolidate_ts"] == 0
        assert state["last_file_count"] == 0
        assert state["turns_since_last"] == 0

    def test_load_returns_defaults_on_corrupt_file(self):
        """Corrupt JSON -> safe defaults, no crash."""
        _clear_state()
        path = memory._state_file_path()
        path.write_text("{not valid json", encoding="utf-8")
        state = memory._load_consolidation_state()
        assert state["last_consolidate_ts"] == 0

    def test_save_then_load_round_trip(self):
        """save -> load preserves timestamp and file count."""
        _clear_state()
        ts = time.time() - 5000
        memory._save_consolidation_state(ts, 15)
        state = memory._load_consolidation_state()
        assert state["last_consolidate_ts"] == ts
        assert state["last_file_count"] == 15

    def test_load_handles_missing_keys(self):
        """State file with missing keys -> defaults for missing keys."""
        _clear_state()
        path = memory._state_file_path()
        path.write_text('{"last_consolidate_ts": 100}', encoding="utf-8")
        state = memory._load_consolidation_state()
        assert state["last_consolidate_ts"] == 100
        assert state["last_file_count"] == 0
        assert state["turns_since_last"] == 0


# --------------------------------------------------------------------------- #
# 3. Cross-process lock (Gate 4)
# --------------------------------------------------------------------------- #

class TestConsolidateLock:
    """.consolidate-lock acquire/release/stale-detection."""

    def test_acquire_then_release(self):
        """Acquire succeeds; release removes the lock file."""
        _clear_lock()
        assert memory._acquire_consolidate_lock() is True
        assert memory._lock_file_path().exists()
        memory._release_consolidate_lock()
        assert not memory._lock_file_path().exists()

    def test_second_acquire_fails_while_held(self):
        """A second acquire while the lock is held returns False."""
        _clear_lock()
        assert memory._acquire_consolidate_lock() is True
        assert memory._acquire_consolidate_lock() is False, \
            "second acquire must fail while lock is held"
        memory._release_consolidate_lock()

    def test_stale_lock_is_stolen(self, monkeypatch):
        """A lock older than CONSOLIDATE_LOCK_STALE is stolen."""
        _clear_lock()
        monkeypatch.setattr(memory, "CONSOLIDATE_LOCK_STALE", 1)

        # Write a stale lock (timestamp 100s ago).
        path = memory._lock_file_path()
        path.write_text(json.dumps({
            "pid": 99999, "ts": time.time() - 100
        }), encoding="utf-8")

        assert memory._acquire_consolidate_lock() is True, \
            "stale lock must be stealable"
        memory._release_consolidate_lock()

    def test_release_silent_on_missing_file(self):
        """Releasing when no lock file exists does not crash."""
        _clear_lock()
        memory._release_consolidate_lock()  # must not raise

    def test_corrupt_lock_is_stolen(self):
        """A corrupt lock file is treated as stale and stolen."""
        _clear_lock()
        path = memory._lock_file_path()
        path.write_text("garbage", encoding="utf-8")
        assert memory._acquire_consolidate_lock() is True
        memory._release_consolidate_lock()


# --------------------------------------------------------------------------- #
# 4. Activity counter
# --------------------------------------------------------------------------- #

class TestActivityCounter:
    """_count_new_transcripts counts files newer than the given timestamp."""

    def test_zero_when_no_transcripts(self):
        """No transcript files -> 0."""
        _clear_transcripts()
        assert memory._count_new_transcripts(time.time()) == 0

    def test_counts_recent_transcripts(self):
        """Transcripts modified after since_ts are counted."""
        _clear_transcripts()
        old_ts = time.time() - 100
        _make_transcript("old", mtime_offset=-200)   # before old_ts
        _make_transcript("new1", mtime_offset=10)    # after old_ts
        _make_transcript("new2", mtime_offset=20)    # after old_ts
        assert memory._count_new_transcripts(old_ts) == 2

    def test_counts_all_when_since_ts_is_zero(self):
        """since_ts=0 (epoch) counts all transcripts."""
        _clear_transcripts()
        _make_transcript("a", mtime_offset=-500)
        _make_transcript("b", mtime_offset=-300)
        assert memory._count_new_transcripts(0) == 2


# --------------------------------------------------------------------------- #
# 5. _should_consolidate gate logic
# --------------------------------------------------------------------------- #

class TestShouldConsolidate:
    """The four-layer gate decision logic."""

    def test_below_threshold_returns_false(self, monkeypatch):
        """Fewer files than CONSOLIDATE_THRESHOLD -> False."""
        _clear_state()
        _clear_lock()
        _clear_transcripts()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 10)
        monkeypatch.setattr(memory, "CONSOLIDATE_HARD_LIMIT", 50)
        _seed_files(5, "below")
        assert memory._should_consolidate() is False

    def test_hard_limit_forces_true(self, monkeypatch):
        """File count >= HARD_LIMIT returns True even without time/activity."""
        _clear_state()
        _clear_lock()
        _clear_transcripts()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 100)
        monkeypatch.setattr(memory, "CONSOLIDATE_HARD_LIMIT", 5)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_INTERVAL", 999999)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_TRANSCRIPTS", 999)
        _seed_files(6, "hard")
        assert memory._should_consolidate() is True, \
            "hard limit must bypass time and activity gates"

    def test_threshold_met_but_time_too_recent_returns_false(self, monkeypatch):
        """Threshold met but last consolidation was recent -> False."""
        _clear_lock()
        _clear_transcripts()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 3)
        monkeypatch.setattr(memory, "CONSOLIDATE_HARD_LIMIT", 50)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_INTERVAL", 3600)
        _seed_files(5, "time")

        # Save state as if we just consolidated.
        memory._save_consolidation_state(time.time(), 5)

        # Create new transcripts (activity gate would pass).
        _make_transcript("t1", mtime_offset=10)
        _make_transcript("t2", mtime_offset=20)
        _make_transcript("t3", mtime_offset=30)

        assert memory._should_consolidate() is False, \
            "time gate must block when last consolidation was recent"
        _clear_state()

    def test_threshold_and_time_met_but_no_activity_returns_false(self, monkeypatch):
        """Threshold + time met but no new transcripts -> False."""
        _clear_state()
        _clear_lock()
        _clear_transcripts()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 3)
        monkeypatch.setattr(memory, "CONSOLIDATE_HARD_LIMIT", 50)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_INTERVAL", 1)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_TRANSCRIPTS", 3)
        _seed_files(5, "noact")

        # State: consolidated long ago (time gate passes).
        memory._save_consolidation_state(time.time() - 10000, 5)

        # No new transcripts.
        assert memory._should_consolidate() is False, \
            "activity gate must block when no new transcripts"
        _clear_state()

    def test_all_gates_pass_returns_true(self, monkeypatch):
        """Threshold + time + activity all pass -> True."""
        _clear_state()
        _clear_lock()
        _clear_transcripts()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 3)
        monkeypatch.setattr(memory, "CONSOLIDATE_HARD_LIMIT", 50)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_INTERVAL", 1)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_TRANSCRIPTS", 2)
        _seed_files(5, "allpass")

        # State: consolidated long ago.
        old_ts = time.time() - 10000
        memory._save_consolidation_state(old_ts, 5)

        # Create new transcripts after old_ts.
        _make_transcript("a1", mtime_offset=10)
        _make_transcript("a2", mtime_offset=20)

        assert memory._should_consolidate() is True
        _clear_state()

    def test_no_state_file_allows_consolidation(self, monkeypatch):
        """No prior state (first run) + threshold + transcripts -> True."""
        _clear_state()
        _clear_lock()
        _clear_transcripts()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 3)
        monkeypatch.setattr(memory, "CONSOLIDATE_HARD_LIMIT", 50)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_INTERVAL", 1)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_TRANSCRIPTS", 1)
        _seed_files(5, "first")
        _make_transcript("first-run", mtime_offset=10)

        # No state file -> last_consolidate_ts=0 -> time gate passes.
        assert memory._should_consolidate() is True


# --------------------------------------------------------------------------- #
# 6. consolidate_memories integration with lock + state
# --------------------------------------------------------------------------- #

class TestConsolidateWithLockAndState:
    """consolidate_memories respects the lock and saves state on success."""

    def test_consolidate_saves_state_on_success(self, monkeypatch):
        """After a successful consolidation, state file is written."""
        _clear_state()
        _clear_lock()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)
        _write("state-test", "user", "d", "b")

        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(
                json.dumps([{"name": "merged", "type": "user",
                             "description": "d", "body": "b"}])))

        memory.consolidate_memories()

        state = memory._load_consolidation_state()
        assert state["last_consolidate_ts"] > 0, \
            "state must be saved after successful consolidation"
        assert state["last_file_count"] == 1
        _clear_state()

    def test_consolidate_releases_lock_on_success(self, monkeypatch):
        """Lock file is removed after successful consolidation."""
        _clear_lock()
        _clear_state()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)
        _write("lock-release", "user", "d", "b")

        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(
                json.dumps([{"name": "merged", "type": "user",
                             "description": "d", "body": "b"}])))

        memory.consolidate_memories()
        assert not memory._lock_file_path().exists(), \
            "lock must be released after successful consolidation"
        _clear_state()

    def test_consolidate_releases_lock_on_llm_failure(self, monkeypatch):
        """Lock is released even when the LLM call raises."""
        _clear_lock()
        _clear_state()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)
        _write("lock-fail", "user", "d", "b")

        def _raise(*a, **kw):
            raise RuntimeError("LLM down")
        monkeypatch.setattr(memory.client.chat.completions, "create", _raise)

        memory.consolidate_memories()
        assert not memory._lock_file_path().exists(), \
            "lock must be released even on LLM failure"

    def test_consolidate_skipped_when_lock_held(self, monkeypatch):
        """consolidate_memories returns early if another process holds the lock."""
        _clear_lock()
        _clear_state()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 1)
        _write("lock-held", "user", "d", "b")

        called = {"create": False}
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: called.__setitem__("create", True))

        # Pre-acquire the lock (simulating another process).
        assert memory._acquire_consolidate_lock()
        try:
            memory.consolidate_memories()
            assert called["create"] is False, \
                "LLM must not be called when lock is held by another process"
        finally:
            memory._release_consolidate_lock()

    def test_consolidate_below_threshold_skips_lock(self, monkeypatch):
        """Below threshold, consolidate returns without touching the lock."""
        _clear_lock()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 100)
        _write("skip-thresh", "user", "d", "b")

        called = {"create": False}
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: called.__setitem__("create", True))

        memory.consolidate_memories()
        assert called["create"] is False
        assert not memory._lock_file_path().exists(), \
            "lock must not be acquired below threshold"


# --------------------------------------------------------------------------- #
# 7. _post_turn_memory gating integration
# --------------------------------------------------------------------------- #

class TestPostTurnGating:
    """_post_turn_memory only calls consolidate_memories when gate passes."""

    def test_consolidate_not_called_when_gate_false(self, monkeypatch):
        """When _should_consolidate returns False, consolidate is skipped."""
        called = {"consolidate": False}
        monkeypatch.setattr(memory, "extract_memories", lambda msgs: None)
        monkeypatch.setattr(memory, "cleanup_stale_memories", lambda: 0)
        monkeypatch.setattr(memory, "_should_consolidate", lambda: False)
        monkeypatch.setattr(
            memory, "consolidate_memories",
            lambda: called.__setitem__("consolidate", True))

        memory._post_turn_memory([{"role": "user", "content": "hi"}])
        assert called["consolidate"] is False, \
            "consolidate must not run when gate returns False"

    def test_consolidate_called_when_gate_true(self, monkeypatch):
        """When _should_consolidate returns True, consolidate runs."""
        called = {"consolidate": False}
        monkeypatch.setattr(memory, "extract_memories", lambda msgs: None)
        monkeypatch.setattr(memory, "cleanup_stale_memories", lambda: 0)
        monkeypatch.setattr(memory, "_should_consolidate", lambda: True)
        monkeypatch.setattr(
            memory, "consolidate_memories",
            lambda: called.__setitem__("consolidate", True))

        memory._post_turn_memory([{"role": "user", "content": "hi"}])
        assert called["consolidate"] is True, \
            "consolidate must run when gate returns True"

    def test_post_turn_ordering_with_gate(self, monkeypatch):
        """Gate is checked after extract + cleanup, before consolidate."""
        order = []
        monkeypatch.setattr(memory, "extract_memories",
                            lambda msgs: order.append("extract"))
        monkeypatch.setattr(memory, "cleanup_stale_memories",
                            lambda: (order.append("cleanup"), 0)[1])
        monkeypatch.setattr(memory, "_should_consolidate",
                            lambda: (order.append("gate"), True)[1])
        monkeypatch.setattr(memory, "consolidate_memories",
                            lambda: order.append("consolidate"))

        memory._post_turn_memory([{"role": "user", "content": "hi"}])
        assert order == ["extract", "cleanup", "gate", "consolidate"]


# --------------------------------------------------------------------------- #
# 8. Full pipeline integrity with gating
# --------------------------------------------------------------------------- #

class TestFullPipelineWithGating:
    """End-to-end: extract -> write -> recall -> consolidate with gates active.

    Verifies that the gating mechanism does not break the normal memory
    lifecycle.  Memories must still be extractable, writable, recallable,
    and consolidatable when gates are satisfied.
    """

    def test_extract_write_recall_works_with_gating(self, monkeypatch):
        """Memories can be written and recalled even when consolidation is gated off."""
        _clear_state()
        _clear_lock()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 999)  # gate off

        # Write a memory.
        _write("pipeline-user-pref", "user",
               "User prefers dark mode", "Always use dark theme")

        # Recall must find it.
        from mcodecore.utils import parse_frontmatter
        files = memory.list_memory_files()
        names = [f["name"] for f in files]
        assert "pipeline-user-pref" in names, \
            "written memory must be recallable"

    def test_consolidate_runs_and_preserves_data(self, monkeypatch):
        """A full consolidation with gates satisfied preserves memory data."""
        _clear_state()
        _clear_lock()
        _clear_transcripts()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 2)
        monkeypatch.setattr(memory, "CONSOLIDATE_HARD_LIMIT", 50)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_INTERVAL", 1)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_TRANSCRIPTS", 1)

        # Seed memories.
        _write("pipe-a", "user", "User likes Python", "Prefers Python 3.12")
        _write("pipe-b", "user", "User likes Python", "Uses Python daily")

        # Create activity (transcript).
        _make_transcript("pipe-act", mtime_offset=10)

        # Mock LLM to merge the two into one.
        monkeypatch.setattr(
            memory.client.chat.completions, "create",
            lambda *a, **kw: _make_llm_response(
                json.dumps([{
                    "name": "pipe-merged",
                    "type": "user",
                    "description": "User likes Python",
                    "body": "Prefers Python 3.12 and uses it daily"
                }])))

        # Gate should pass.
        assert memory._should_consolidate() is True

        memory.consolidate_memories()

        # Verify merged memory exists and originals are gone.
        memory._invalidate_memory_cache()
        files = memory.list_memory_files()
        names = [f["name"] for f in files]
        assert "pipe-merged" in names
        assert "pipe-a" not in names
        assert "pipe-b" not in names

        # State must be saved.
        state = memory._load_consolidation_state()
        assert state["last_consolidate_ts"] > 0

        _clear_state()

    def test_consolidate_state_prevents_immediate_remerge(self, monkeypatch):
        """After consolidation, the state file prevents immediate re-merge."""
        _clear_state()
        _clear_lock()
        _clear_transcripts()
        monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 2)
        monkeypatch.setattr(memory, "CONSOLIDATE_HARD_LIMIT", 50)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_INTERVAL", 3600)
        monkeypatch.setattr(memory, "CONSOLIDATE_MIN_TRANSCRIPTS", 1)

        _write("remege-a", "user", "d", "b")
        _write("remege-b", "user", "d", "b")

        # Simulate just-consolidated state.
        memory._save_consolidation_state(time.time(), 2)

        # Even with transcripts, time gate blocks.
        _make_transcript("remege-act", mtime_offset=10)
        assert memory._should_consolidate() is False, \
            "time gate must block re-merge within cooldown"

        _clear_state()
