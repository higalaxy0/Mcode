"""General helper function tests.

Covers ``mcodecore.utils``:
parse_frontmatter / parse_bg_command / parse_explicit_timeout / truncate / new_request_id.
"""

from __future__ import annotations

import re

from mcodecore import utils


# --------------------------------------------------------------------------- #
# parse_frontmatter
# --------------------------------------------------------------------------- #

def test_parse_frontmatter_with_meta():
    raw = "---\nname: foo\ndescription: bar\ntype: user\n---\n\nbody text here"
    meta, body = utils.parse_frontmatter(raw)
    assert meta["name"] == "foo"
    assert meta["description"] == "bar"
    assert meta["type"] == "user"
    assert body.strip() == "body text here"


def test_parse_frontmatter_no_meta():
    raw = "just plain text\nline2"
    meta, body = utils.parse_frontmatter(raw)
    assert meta == {}
    assert body == raw


def test_parse_frontmatter_empty_meta():
    raw = "---\n---\n\nbody"
    meta, body = utils.parse_frontmatter(raw)
    assert meta == {}
    assert body.strip() == "body"


# --------------------------------------------------------------------------- #
# parse_bg_command
# --------------------------------------------------------------------------- #

def test_parse_bg_command_with_prefix():
    is_bg, log_name, cmd = utils.parse_bg_command("bg: python server.py")
    assert is_bg is True
    assert cmd == "python server.py"
    assert log_name is None


def test_parse_bg_command_with_named_log():
    is_bg, log_name, cmd = utils.parse_bg_command("bg: python -m http.server log=server.log")
    assert is_bg is True
    assert log_name == "server.log"
    assert cmd == "python -m http.server"


def test_parse_bg_command_without_prefix():
    is_bg, log_name, cmd = utils.parse_bg_command("echo hello")
    assert is_bg is False
    assert cmd == "echo hello"


# --------------------------------------------------------------------------- #
# parse_explicit_timeout
# --------------------------------------------------------------------------- #

def test_parse_explicit_timeout_with_marker():
    timeout, cmd = utils.parse_explicit_timeout("sleep 10 # timeout=600")
    assert timeout == 600
    assert "timeout" not in cmd
    assert "sleep 10" in cmd


def test_parse_explicit_timeout_without_marker():
    timeout, cmd = utils.parse_explicit_timeout("echo hi")
    # Falls back to default BASH_TIMEOUT when no marker is present
    from mcodecore.config import BASH_TIMEOUT
    assert timeout == BASH_TIMEOUT
    assert cmd == "echo hi"


# --------------------------------------------------------------------------- #
# truncate
# --------------------------------------------------------------------------- #

def test_truncate_short_unchanged():
    assert utils.truncate("short") == "short"


def test_truncate_long_capped():
    s = "x" * 500
    out = utils.truncate(s, limit=100)
    assert len(out) <= 103  # 100 + ellipsis


def test_truncate_custom_suffix_not_supported():
    # truncate has no suffix parameter; long text always appends "..."
    s = "x" * 500
    out = utils.truncate(s, limit=10)
    assert out.endswith("...")
    assert len(out) == 13  # 10 + "..."


# --------------------------------------------------------------------------- #
# new_request_id
# --------------------------------------------------------------------------- #

def test_new_request_id_format():
    from mcodecore.config import SESSION_ID
    rid = utils.new_request_id()
    # Session-scoped format: embeds SESSION_ID so ids never collide across
    # concurrently running mcode windows in the same folder.
    assert re.match(rf"^req_{re.escape(SESSION_ID)}_[0-9a-f]{{8}}$", rid)


def test_new_request_id_unique():
    ids = {utils.new_request_id() for _ in range(50)}
    assert len(ids) == 50  # almost certainly all unique
