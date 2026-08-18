"""Filesystem & shell tool tests.

Covers ``mcodecore.fsops``:
safe_path (out-of-bounds protection) / run_read / run_write / run_edit /
run_glob / run_grep.
Note: ``run_bash`` involves a real subprocess and is given a lightweight
smoke test separately.
"""

from __future__ import annotations

import pytest

from mcodecore import fsops


# --------------------------------------------------------------------------- #
# safe_path
# --------------------------------------------------------------------------- #

def test_safe_path_relative_resolves_in_workdir(tmp_path):
    p = fsops.safe_path("foo.txt")
    assert p.parent == tmp_path


def test_safe_path_subdir_allowed():
    p = fsops.safe_path("sub/dir/file.txt")
    assert p.is_relative_to(fsops.WORKDIR)


def test_safe_path_traversal_blocked():
    with pytest.raises(ValueError, match="escapes workspace"):
        fsops.safe_path("../../etc/passwd")


def test_safe_path_absolute_outside_blocked():
    with pytest.raises(ValueError):
        fsops.safe_path("C:/Windows/System32/drivers/etc/hosts")


# --------------------------------------------------------------------------- #
# run_write / run_read
# --------------------------------------------------------------------------- #

def test_run_write_creates_file():
    res = fsops.run_write("note.txt", "hello world")
    assert "Wrote" in res
    assert (fsops.WORKDIR / "note.txt").read_text() == "hello world"


def test_run_write_creates_parent_dirs():
    fsops.run_write("a/b/c.txt", "nested")
    assert (fsops.WORKDIR / "a" / "b" / "c.txt").read_text() == "nested"


def test_run_read_returns_numbered_lines():
    fsops.run_write("lines.txt", "one\ntwo\nthree")
    out = fsops.run_read("lines.txt")
    assert "one" in out and "two" in out and "three" in out
    assert "1->" in out or "1->" in out  # line-number prefix


def test_run_read_offset_pagination():
    fsops.run_write("p.txt", "\n".join(f"line{i}" for i in range(10)))
    out = fsops.run_read("p.txt", offset=3, limit=2)
    assert "line2" in out
    assert "line3" in out
    assert "line4" not in out


def test_run_read_offset_beyond_end():
    fsops.run_write("small.txt", "only")
    out = fsops.run_read("small.txt", offset=999)
    assert "beyond end" in out


def test_run_read_empty_file():
    fsops.run_write("empty.txt", "")
    out = fsops.run_read("empty.txt")
    assert "empty file" in out


def test_run_read_directory_error():
    fsops.WORKDIR.joinpath("adir").mkdir(exist_ok=True)
    out = fsops.run_read("adir")
    assert "directory" in out.lower()


def test_run_read_binary_detection():
    p = fsops.WORKDIR / "bin.dat"
    p.write_bytes(b"\x00\x01\x02\x03binary")
    out = fsops.run_read("bin.dat")
    assert "Binary file" in out


# --------------------------------------------------------------------------- #
# run_edit
# --------------------------------------------------------------------------- #

def test_run_edit_replaces_first_match():
    fsops.run_write("e.txt", "foo bar foo")
    fsops.run_edit("e.txt", "foo", "QUX")
    assert (fsops.WORKDIR / "e.txt").read_text() == "QUX bar foo"  # only the first match is replaced


def test_run_edit_not_found():
    fsops.run_write("e.txt", "hello")
    res = fsops.run_edit("e.txt", "zzz", "yyy")
    assert "not found" in res
    assert (fsops.WORKDIR / "e.txt").read_text() == "hello"


# --------------------------------------------------------------------------- #
# run_glob
# --------------------------------------------------------------------------- #

def test_run_glob_matches_pattern():
    fsops.run_write("alpha.py", "x")
    fsops.run_write("beta.py", "y")
    fsops.run_write("gamma.md", "z")
    out = fsops.run_glob("*.py")
    assert "alpha.py" in out
    assert "beta.py" in out
    assert "gamma.md" not in out


def test_run_glob_recursive():
    fsops.run_write("pkg/deep/mod.py", "x")
    out = fsops.run_glob("**/*.py")
    assert "mod.py" in out


def test_run_glob_no_matches():
    out = fsops.run_glob("*.nonexistent_ext")
    assert "no matches" in out


# --------------------------------------------------------------------------- #
# run_grep
# --------------------------------------------------------------------------- #

def test_run_grep_finds_pattern():
    fsops.run_write("g.py", "def hello():\n    pass\ndef world():\n    return 1")
    out = fsops.run_grep("def ", ".")
    assert "hello" in out
    assert "world" in out


def test_run_grep_include_filter():
    fsops.run_write("match.py", "target_line")
    fsops.run_write("match.txt", "target_line")
    out = fsops.run_grep("target_line", ".", include="*.py")
    assert "match.py" in out
    assert "match.txt" not in out


def test_run_grep_no_matches():
    fsops.run_write("g.py", "nothing here")
    out = fsops.run_grep("zzznomatch", ".")
    assert "no matches" in out


def test_run_grep_invalid_regex():
    out = fsops.run_grep("(unclosed", ".")
    assert "invalid regex" in out


# --------------------------------------------------------------------------- #
# run_bash smoke test (real subprocess, kept lightweight)
# --------------------------------------------------------------------------- #

def test_run_bash_simple_echo():
    out = fsops.run_bash("echo pytest_smoke_test")
    assert "pytest_smoke_test" in out


def test_run_bash_handles_error():
    out = fsops.run_bash("nonexistent_command_xyz123 2>nul")
    # When the command doesn't exist the shell returns an error message or empty; passing means no exception is raised
    assert isinstance(out, str)
