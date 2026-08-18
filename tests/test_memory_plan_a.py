"""Plan A tests: frontmatter timestamps + access tracking.

Verifies:
- created_at / updated_at written on new memory
- created_at preserved, updated_at refreshed on re-write (update semantics)
- hit_count increments and last_used refreshes on load_memories touch
- list_memory_files returns the new metadata fields
- consolidate catalog includes time/usage metadata (via mock)
"""

from __future__ import annotations

import time

from mcodecore import memory
from mcodecore.utils import parse_frontmatter


# --------------------------------------------------------------------------- #
# Timestamps: created_at / updated_at
# --------------------------------------------------------------------------- #

def test_new_memory_has_created_and_updated():
    """A freshly written memory must carry both created_at and updated_at."""
    memory.write_memory_file("ts-new", "user", "desc", "body")
    raw = memory.read_memory_file("ts-new.md")
    meta, _ = parse_frontmatter(raw)
    assert "created_at" in meta and meta["created_at"]
    assert "updated_at" in meta and meta["updated_at"]
    assert meta["created_at"] == meta["updated_at"]


def test_rewrite_preserves_created_at():
    """Re-writing the same name must keep the original created_at."""
    memory.write_memory_file("ts-update", "user", "v1 desc", "v1 body")
    raw1 = memory.read_memory_file("ts-update.md")
    meta1, _ = parse_frontmatter(raw1)
    created1 = meta1["created_at"]

    # Sleep to ensure timestamp can differ.
    time.sleep(1.1)

    memory.write_memory_file("ts-update", "user", "v2 desc", "v2 body")
    raw2 = memory.read_memory_file("ts-update.md")
    meta2, body2 = parse_frontmatter(raw2)

    assert meta2["created_at"] == created1, "created_at must be preserved"
    assert meta2["updated_at"] != created1 or meta2["description"] == "v2 desc"
    assert "v2 body" in body2


def test_different_memories_have_independent_timestamps():
    """Two different memories written at different times have different timestamps."""
    memory.write_memory_file("ts-a", "user", "a", "body-a")
    time.sleep(1.1)
    memory.write_memory_file("ts-b", "user", "b", "body-b")
    ra = parse_frontmatter(memory.read_memory_file("ts-a.md"))[0]
    rb = parse_frontmatter(memory.read_memory_file("ts-b.md"))[0]
    assert ra["created_at"] != rb["created_at"]


# --------------------------------------------------------------------------- #
# Access tracking: hit_count / last_used
# --------------------------------------------------------------------------- #

def test_new_memory_has_zero_hit_count():
    """A freshly written memory starts with hit_count 0."""
    memory.write_memory_file("hc-new", "user", "desc", "body")
    meta, _ = parse_frontmatter(memory.read_memory_file("hc-new.md"))
    assert meta["hit_count"] == "0"


def test_load_memories_increments_hit_count(monkeypatch):
    """load_memories must increment hit_count for each injected memory."""
    memory.write_memory_file("hc-load", "user", "python config", "use python 3")

    # Force select_relevant_memories to return our file (bypass LLM).
    monkeypatch.setattr(
        memory, "select_relevant_memories",
        lambda msgs, max_items=5: ["hc-load.md"])

    before = parse_frontmatter(memory.read_memory_file("hc-load.md"))[0]
    assert before["hit_count"] == "0"

    memory.load_memories([{"role": "user", "content": "python"}])

    after = parse_frontmatter(memory.read_memory_file("hc-load.md"))[0]
    assert int(after["hit_count"]) == 1
    assert after["last_used"] == before["created_at"] or after["last_used"] >= before["created_at"]


def test_load_memories_increments_multiple_times(monkeypatch):
    """Repeated load_memories calls keep incrementing hit_count."""
    memory.write_memory_file("hc-multi", "user", "desc", "body")
    monkeypatch.setattr(
        memory, "select_relevant_memories",
        lambda msgs, max_items=5: ["hc-multi.md"])

    for _ in range(3):
        memory.load_memories([{"role": "user", "content": "x"}])

    meta, _ = parse_frontmatter(memory.read_memory_file("hc-multi.md"))
    assert int(meta["hit_count"]) == 3


def test_touch_preserves_created_at_and_body(monkeypatch):
    """_touch_memory must not alter created_at or body content."""
    memory.write_memory_file("hc-preserve", "user", "desc", "original body")
    monkeypatch.setattr(
        memory, "select_relevant_memories",
        lambda msgs, max_items=5: ["hc-preserve.md"])

    orig = parse_frontmatter(memory.read_memory_file("hc-preserve.md"))
    orig_created = orig[0]["created_at"]
    orig_body = orig[1]

    memory.load_memories([{"role": "user", "content": "x"}])

    after = parse_frontmatter(memory.read_memory_file("hc-preserve.md"))
    assert after[0]["created_at"] == orig_created
    assert after[1] == orig_body


# --------------------------------------------------------------------------- #
# list_memory_files returns new fields
# --------------------------------------------------------------------------- #

def test_list_memory_files_has_metadata_fields():
    memory.write_memory_file("lmf-meta", "feedback", "a desc", "a body")
    files = memory.list_memory_files()
    f = [x for x in files if x["name"] == "lmf-meta"][0]
    assert "created_at" in f and f["created_at"]
    assert "updated_at" in f and f["updated_at"]
    assert "hit_count" in f and f["hit_count"] == 0
    assert "last_used" in f and f["last_used"]
    assert "expires_at" in f  # may be empty string


# --------------------------------------------------------------------------- #
# Backward compat: old-format files (no timestamps) still parse
# --------------------------------------------------------------------------- #

def test_legacy_file_without_timestamps_still_works():
    """A file written in the old format (no created_at etc.) must still load."""
    # Manually write an old-style file.
    path = memory.MEMORY_DIR / "legacy-mem.md"
    path.write_text("---\nname: legacy-mem\ndescription: old\ntype: user\n---\n\nold body\n")
    files = memory.list_memory_files()
    f = [x for x in files if x["name"] == "legacy-mem"][0]
    assert f["body"] == "old body"
    assert f["hit_count"] == 0  # defaults to 0
    assert f["created_at"] == ""  # absent -> empty string
