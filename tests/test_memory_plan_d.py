"""Plan D tests: TTL expiry + dead-memory cleanup.

Verifies:
- is_expired: memory with expires_at in the past is flagged expired
- is_expired: memory with expires_at in the future is NOT expired
- is_expired: memory with no expires_at is never expired
- is_dead_memory: hit_count=0 + old last_used => dead
- is_dead_memory: hit_count>0 => NOT dead (even if old)
- is_dead_memory: hit_count=0 + recent => NOT dead
- cleanup_stale_memories removes expired files
- cleanup_stale_memories removes dead files
- cleanup_stale_memories does NOT remove feedback memories (even if dead/expired)
- cleanup_stale_memories does NOT remove recently-used memories
- _write_memory_file_no_index accepts expires_at parameter
- consolidate catalog marks expired/dead memories with tags
"""

from __future__ import annotations

import time
from pathlib import Path

from mcodecore import memory
from mcodecore.utils import parse_frontmatter


def _past_ts(days_ago: int = 10) -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.localtime(time.time() - days_ago * 86400))


def _future_ts(days_ahead: int = 10) -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.localtime(time.time() + days_ahead * 86400))


# --------------------------------------------------------------------------- #
# is_expired
# --------------------------------------------------------------------------- #

def test_is_expired_past():
    meta = {"expires_at": _past_ts(10)}
    assert memory.is_expired(meta) is True


def test_is_expired_future():
    meta = {"expires_at": _future_ts(10)}
    assert memory.is_expired(meta) is False


def test_is_expired_none():
    assert memory.is_expired({}) is False
    assert memory.is_expired({"expires_at": ""}) is False


# --------------------------------------------------------------------------- #
# is_dead_memory
# --------------------------------------------------------------------------- #

def test_is_dead_never_used_old():
    """hit_count=0 + old last_used => dead."""
    meta = {"hit_count": "0", "last_used": _past_ts(10)}
    assert memory.is_dead_memory(meta) is True


def test_is_dead_used_not_dead():
    """hit_count>0 => NOT dead even if old."""
    meta = {"hit_count": "5", "last_used": _past_ts(30)}
    assert memory.is_dead_memory(meta) is False


def test_is_dead_recent_not_dead():
    """hit_count=0 + recent => NOT dead."""
    meta = {"hit_count": "0", "last_used": _now_iso_recent()}
    assert memory.is_dead_memory(meta) is False


def test_is_dead_no_timestamp():
    """No last_used and no created_at => not dead (can't determine age)."""
    meta = {"hit_count": "0"}
    assert memory.is_dead_memory(meta) is False


def _now_iso_recent() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


# --------------------------------------------------------------------------- #
# cleanup_stale_memories: expired
# --------------------------------------------------------------------------- #

def test_cleanup_removes_expired():
    """Expired non-feedback memory is removed."""
    memory._write_memory_file_no_index(
        "cleanup-exp", "project", "d", "b",
        expires_at=_past_ts(5))
    assert (memory.MEMORY_DIR / "cleanup-exp.md").exists()

    removed = memory.cleanup_stale_memories()
    assert removed >= 1
    assert not (memory.MEMORY_DIR / "cleanup-exp.md").exists()


def test_cleanup_removes_dead():
    """Dead memory (hit_count=0, old last_used) is removed."""
    memory._write_memory_file_no_index("cleanup-dead", "project", "d", "b")
    # Manually set last_used to old time and hit_count to 0.
    path = memory.MEMORY_DIR / "cleanup-dead.md"
    raw = path.read_text()
    meta, body = parse_frontmatter(raw)
    meta["hit_count"] = "0"
    meta["last_used"] = _past_ts(10)
    path.write_text(f"{memory._build_frontmatter(meta)}\n\n{body}\n")

    removed = memory.cleanup_stale_memories()
    assert removed >= 1
    assert not path.exists()


# --------------------------------------------------------------------------- #
# cleanup_stale_memories: feedback never removed
# --------------------------------------------------------------------------- #

def test_cleanup_preserves_feedback_even_if_dead():
    """Feedback memory is never auto-removed even if dead/expired."""
    memory._write_memory_file_no_index(
        "cleanup-fb", "feedback", "d", "b",
        expires_at=_past_ts(5))
    path = memory.MEMORY_DIR / "cleanup-fb.md"
    assert path.exists()

    removed = memory.cleanup_stale_memories()
    assert path.exists(), "feedback must survive cleanup"


# --------------------------------------------------------------------------- #
# cleanup_stale_memories: fresh memories preserved
# --------------------------------------------------------------------------- #

def test_cleanup_preserves_fresh():
    """Recently-created, never-used memory is NOT dead (age < threshold)."""
    memory._write_memory_file_no_index("cleanup-fresh", "project", "d", "b")
    path = memory.MEMORY_DIR / "cleanup-fresh.md"
    assert path.exists()

    removed = memory.cleanup_stale_memories()
    assert path.exists(), "fresh memory must survive cleanup"


def test_cleanup_preserves_used():
    """Used memory (hit_count>0) is NOT dead."""
    memory._write_memory_file_no_index("cleanup-used", "project", "d", "b")
    path = memory.MEMORY_DIR / "cleanup-used.md"
    raw = path.read_text()
    meta, body = parse_frontmatter(raw)
    meta["hit_count"] = "3"
    meta["last_used"] = _past_ts(30)  # old but used
    path.write_text(f"{memory._build_frontmatter(meta)}\n\n{body}\n")

    memory.cleanup_stale_memories()
    assert path.exists(), "used memory must survive cleanup"


# --------------------------------------------------------------------------- #
# expires_at parameter in writer
# --------------------------------------------------------------------------- #

def test_writer_accepts_expires_at():
    """_write_memory_file_no_index writes expires_at when provided."""
    future = _future_ts(5)
    memory._write_memory_file_no_index(
        "exp-param", "project", "d", "b", expires_at=future)
    meta, _ = parse_frontmatter(memory.read_memory_file("exp-param.md"))
    assert meta.get("expires_at") == future


def test_writer_expires_at_preserved_on_update():
    """expires_at is preserved when updating without specifying it."""
    future = _future_ts(5)
    memory._write_memory_file_no_index(
        "exp-upd", "project", "v1", "b1", expires_at=future)
    # Update without expires_at -- should preserve.
    memory._write_memory_file_no_index("exp-upd", "project", "v2", "b2")
    meta, _ = parse_frontmatter(memory.read_memory_file("exp-upd.md"))
    assert meta.get("expires_at") == future


# --------------------------------------------------------------------------- #
# Consolidate catalog marks expired/dead
# --------------------------------------------------------------------------- #

def test_consolidate_catalog_marks_expired(monkeypatch):
    """Consolidation catalog includes [EXPIRED] tag for expired memories."""
    memory._write_memory_file_no_index(
        "cons-exp", "project", "d", "b",
        expires_at=_past_ts(5))
    # Add more files to reach threshold.
    for i in range(memory.CONSOLIDATE_THRESHOLD):
        memory._write_memory_file_no_index(
            f"cons-fill-{i}", "project", f"d{i}", f"b{i}")

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

    # Temporarily lower threshold so we can trigger consolidation.
    monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 2)
    memory.consolidate_memories()

    prompt = captured.get("prompt", "")
    assert "[EXPIRED]" in prompt


def test_consolidate_catalog_marks_dead(monkeypatch):
    """Consolidation catalog includes [DEAD] tag for dead memories."""
    memory._write_memory_file_no_index("cons-dead", "project", "d", "b")
    path = memory.MEMORY_DIR / "cons-dead.md"
    raw = path.read_text()
    meta, body = parse_frontmatter(raw)
    meta["hit_count"] = "0"
    meta["last_used"] = _past_ts(10)
    path.write_text(f"{memory._build_frontmatter(meta)}\n\n{body}\n")
    # Add filler to reach threshold.
    memory._write_memory_file_no_index("cons-fill-dead", "project", "d2", "b2")

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
    monkeypatch.setattr(memory, "CONSOLIDATE_THRESHOLD", 2)

    memory.consolidate_memories()

    prompt = captured.get("prompt", "")
    assert "[DEAD" in prompt
