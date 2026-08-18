"""Plan B tests: deterministic dedup on write (slug collision + update detection).

Verifies:
- Same name re-write = update (created_at preserved, body overwritten)
- Different names that slug-collide do NOT overwrite (coexist with suffix)
- Special characters in names are handled safely
- Collision detection works inside batch writer (_write_memory_file_no_index)
"""

from __future__ import annotations

from mcodecore import memory
from mcodecore.utils import parse_frontmatter


# --------------------------------------------------------------------------- #
# Update semantics: same name = update
# --------------------------------------------------------------------------- #

def test_same_name_updates_existing():
    """Re-writing the same name overwrites the file content (update path)."""
    memory.write_memory_file("upd-same", "user", "v1", "body-v1")
    memory.write_memory_file("upd-same", "user", "v2", "body-v2")
    raw = memory.read_memory_file("upd-same.md")
    meta, body = parse_frontmatter(raw)
    assert "body-v2" in body
    assert meta["description"] == "v2"
    # Only one file with this slug should exist.
    assert (memory.MEMORY_DIR / "upd-same.md").exists()
    assert not (memory.MEMORY_DIR / "upd-same-2.md").exists()


def test_same_name_preserves_created_at_on_update():
    """Update path must preserve the original created_at."""
    memory.write_memory_file("upd-ts", "user", "d1", "b1")
    m1 = parse_frontmatter(memory.read_memory_file("upd-ts.md"))[0]
    memory.write_memory_file("upd-ts", "user", "d2", "b2")
    m2 = parse_frontmatter(memory.read_memory_file("upd-ts.md"))[0]
    assert m2["created_at"] == m1["created_at"]


# --------------------------------------------------------------------------- #
# Collision avoidance: different name, same slug
# --------------------------------------------------------------------------- #

def test_slug_collision_creates_separate_file():
    """Two different names that slugify identically must coexist, not overwrite."""
    # "User Pref" and "user-pref" both slugify to "user-pref"
    memory.write_memory_file("User Pref", "user", "first memory", "body one")
    memory.write_memory_file("user-pref", "user", "second memory", "body two")

    f1 = memory.read_memory_file("user-pref.md")
    f2 = memory.read_memory_file("user-pref-2.md")
    assert f1 is not None and f2 is not None
    assert "body one" in f1
    assert "body two" in f2
    # Both files have different stored names.
    assert parse_frontmatter(f1)[0]["name"] == "User Pref"
    assert parse_frontmatter(f2)[0]["name"] == "user-pref"


def test_slug_collision_three_way():
    """Three names that slugify to the same slug get -2, -3 suffixes."""
    memory.write_memory_file("Test Mem", "user", "m1", "b1")
    memory.write_memory_file("test-mem", "user", "m2", "b2")
    memory.write_memory_file("Test  Mem", "user", "m3", "b3")  # double space -> same slug

    assert (memory.MEMORY_DIR / "test-mem.md").exists()
    assert (memory.MEMORY_DIR / "test-mem-2.md").exists()
    assert (memory.MEMORY_DIR / "test-mem-3.md").exists()


# --------------------------------------------------------------------------- #
# Special characters
# --------------------------------------------------------------------------- #

def test_special_chars_in_name():
    """Characters illegal in Windows filenames are stripped from the slug."""
    path = memory.write_memory_file("file:name?", "user", "d", "b")
    # ':' and '?' must be stripped.
    assert ":" not in path.name
    assert "?" not in path.name
    assert path.exists()


def test_slugify_cjk_preserved():
    """CJK characters in the name are preserved in the slug (not stripped)."""
    slug = memory._slugify("用户偏好")
    assert "用户偏好" in slug


# --------------------------------------------------------------------------- #
# Batch writer collision avoidance
# --------------------------------------------------------------------------- #

def test_no_index_writer_collision():
    """_write_memory_file_no_index also avoids slug collisions."""
    d = memory.MEMORY_DIR
    memory._write_memory_file_no_index("Batch Mem", "user", "d1", "b1", d)
    memory._write_memory_file_no_index("batch-mem", "user", "d2", "b2", d)

    f1 = (d / "batch-mem.md").read_text()
    f2 = (d / "batch-mem-2.md").read_text()
    assert "b1" in f1
    assert "b2" in f2


def test_no_index_writer_update_preserves_created_at():
    """Batch writer update path also preserves created_at."""
    d = memory.MEMORY_DIR
    memory._write_memory_file_no_index("batch-upd", "user", "d1", "b1", d)
    m1 = parse_frontmatter((d / "batch-upd.md").read_text())[0]
    memory._write_memory_file_no_index("batch-upd", "user", "d2", "b2", d)
    m2 = parse_frontmatter((d / "batch-upd.md").read_text())[0]
    assert m2["created_at"] == m1["created_at"]
    assert "b2" in (d / "batch-upd.md").read_text()


# --------------------------------------------------------------------------- #
# list_memory_files after collision
# --------------------------------------------------------------------------- #

def test_list_after_collision_shows_both():
    memory.write_memory_file("Dup Name", "user", "first", "b1")
    memory.write_memory_file("dup-name", "user", "second", "b2")
    files = memory.list_memory_files()
    names = [f["name"] for f in files]
    assert "Dup Name" in names
    assert "dup-name" in names
