"""Memory read/write and index tests.

Covers ``mcodecore.memory``:
write_memory_file / read_memory_index / read_memory_file / list_memory_files.
(extract/consolidate involve the LLM; select_relevant_memories semantics are
tested separately via mocking.)
"""

from __future__ import annotations

from mcodecore import memory


def _write(name, mtype="preference", desc="d", body="b"):
    return memory.write_memory_file(name, mtype, desc, body)


# --------------------------------------------------------------------------- #
# write_memory_file / read_memory_file
# --------------------------------------------------------------------------- #

def test_write_memory_file_creates_file():
    path = _write("test-mem", "preference", "a test memory", "content body")
    assert path.exists()
    assert path.suffix == ".md"


def test_write_memory_file_slugifies_name():
    path = _write("My Memory Name", "preference", "desc", "body")
    assert path.name == "my-memory-name.md"


def test_write_memory_file_overwrites():
    _write("dup", "preference", "v1", "body1")
    _write("dup", "preference", "v2", "body2")
    text = memory.read_memory_file("dup.md")
    assert "body2" in text
    assert "body1" not in text


def test_read_memory_file_not_found():
    res = memory.read_memory_file("does_not_exist.md")
    assert res is None


def test_read_memory_file_roundtrip():
    _write("roundtrip", "preference", "a desc", "body content")
    text = memory.read_memory_file("roundtrip.md")
    assert "body content" in text
    assert "name: roundtrip" in text


def test_written_file_has_frontmatter():
    _write("fm", "project", "the desc", "the body")
    text = memory.read_memory_file("fm.md")
    assert text.startswith("---")
    assert "name: fm" in text
    assert "description: the desc" in text
    assert "type: project" in text


# --------------------------------------------------------------------------- #
# read_memory_index
# --------------------------------------------------------------------------- #

def test_read_memory_index_empty():
    assert memory.read_memory_index() == ""


def test_read_memory_index_lists_memories():
    _write("alpha", "preference", "alpha desc", "A body")
    _write("beta", "backend", "beta desc", "B body")
    index = memory.read_memory_index()
    assert "alpha" in index
    assert "beta" in index
    assert "alpha desc" in index


# --------------------------------------------------------------------------- #
# list_memory_files
# --------------------------------------------------------------------------- #

def test_list_memory_files():
    _write("m1", "preference", "one desc", "1")
    _write("m2", "preference", "two desc", "2")
    files = memory.list_memory_files()
    names = [f["name"] for f in files]
    assert "m1" in names
    assert "m2" in names


def test_list_memory_files_empty():
    assert memory.list_memory_files() == []


def test_list_memory_files_fields():
    _write("fld", "project", "fld desc", "fld body")
    f = memory.list_memory_files()[0]
    assert f["filename"] == "fld.md"
    assert f["name"] == "fld"
    assert f["description"] == "fld desc"
    assert f["type"] == "project"
    assert f["body"] == "fld body"


# --------------------------------------------------------------------------- #
# select_relevant_memories (keyword fallback when LLM fails)
# --------------------------------------------------------------------------- #

def test_select_relevant_memories_keyword_fallback(monkeypatch):
    """Falls back to keyword matching when the LLM call fails."""
    _write("python-config", "preference", "python configuration", "use python 3")
    _write("rust-config", "preference", "rust configuration", "use rust")

    def _raise(*a, **k):
        raise Exception("LLM unavailable")
    monkeypatch.setattr(memory.client.chat.completions, "create", _raise)

    selected = memory.select_relevant_memories(
        [{"role": "user", "content": "how to configure python"}])
    # Keywords "python" and "configure" match python-config
    assert any("python-config" in s for s in selected)


def test_select_relevant_memories_no_files():
    assert memory.select_relevant_memories([{"role": "user", "content": "x"}]) == []


def test_select_relevant_memories_no_recent_text():
    _write("x", "preference", "d", "b")
    assert memory.select_relevant_memories([]) == []


# --------------------------------------------------------------------------- #
# Index auto-update
# --------------------------------------------------------------------------- #

def test_index_updated_after_write():
    _write("idxtest", "preference", "idx desc", "content")
    index = memory.read_memory_index()
    assert "idxtest" in index
    assert memory.MEMORY_INDEX.exists()
