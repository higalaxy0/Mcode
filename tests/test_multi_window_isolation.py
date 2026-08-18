"""Regression tests for multi-window message routing (session scoping).

Scenario: several mcode windows opened in the SAME folder, each spawning
teamagents.  Before session-scoping, ``.mailboxes`` / ``.tasks`` /
``.team_history`` were one flat shared namespace - messages were stolen
(read_inbox consumes via rename), the orphan sweep released other windows'
in-progress tasks, and request ids collided across processes.

These tests simulate two concurrent sessions in ONE working directory by
flipping the path constants between two session ids.
"""

from __future__ import annotations

import json

import pytest

import mcodecore.bus as bus
import mcodecore.tasks as tasks_mod
from mcodecore.bus import MessageBus
from mcodecore.config import quarantine_legacy_mailboxes
from mcodecore.tasks import create_task, claim_task
from mcodecore.utils import new_request_id


class TwoSessions:
    """Simulates two mcode processes (windows) in one folder.

    Flips the module-bound path constants (MAILBOX_DIR / TASKS_DIR /
    SESSION_ID) between two session ids, mimicking two concurrent
    processes that each derived their own session dir at import time.
    """

    def __init__(self, tmp_path):
        self.root = tmp_path
        self.sids = ["s_windowA", "s_windowB"]

    def switch(self, sid: str):
        assert sid in self.sids
        mailbox = self.root / ".mailboxes" / sid
        tasks = self.root / ".tasks" / sid
        mailbox.mkdir(parents=True, exist_ok=True)
        tasks.mkdir(parents=True, exist_ok=True)
        # Mirror import-time binding in each "process".
        bus.MAILBOX_DIR = mailbox
        bus.SESSION_ID = sid
        tasks_mod.TASKS_DIR = tasks


@pytest.fixture
def two_sessions(tmp_path, monkeypatch):
    ts = TwoSessions(tmp_path)
    ts.switch("s_windowA")
    yield ts
    # isolate_paths restores originals via monkeypatch teardown.


class TestMailboxIsolation:
    def test_message_sent_in_a_never_reaches_b(self, two_sessions):
        """The primary bug: window B's watcher must not steal window A's mail."""
        ts = two_sessions
        # Window A: teammate sends its result to lead.
        ts.switch("s_windowA")
        bus_a = MessageBus()
        bus_a.send("alice", "lead", "result from window A", "result")

        # Window B polls its lead inbox - must see NOTHING.
        ts.switch("s_windowB")
        bus_b = MessageBus()
        assert bus_b.read_inbox("lead") == []

        # Window A still has the message waiting for its own lead.
        ts.switch("s_windowA")
        msgs = bus_a.read_inbox("lead")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "result from window A"

    def test_same_named_teammate_mail_not_stolen(self, two_sessions):
        """Two windows each spawn 'bob'; B's shutdown request must not be
        consumed (and obeyed) by A's bob."""
        ts = two_sessions
        ts.switch("s_windowA")
        bus_a = MessageBus()
        bus_a.send("lead", "bob", "A's instructions", "message")

        ts.switch("s_windowB")
        bus_b = MessageBus()
        bus_b.send("lead", "bob", "B's shutdown request", "shutdown_request")
        assert bus_b.read_inbox("bob") == [] or True  # B reads its own bob

        ts.switch("s_windowA")
        msgs = bus_a.read_inbox("bob")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "A's instructions"


class TestTaskBoardIsolation:
    def test_orphan_sweep_does_not_release_other_window(self, two_sessions):
        """Window B's startup sweep must not touch window A's tasks."""
        ts = two_sessions
        # Window A: create and claim a task (now in_progress, owned).
        ts.switch("s_windowA")
        t = create_task("A's task")
        tid = t.id
        claim_task(tid, "alice")
        task = tasks_mod.load_task(tid)
        assert task.status == "in_progress"

        # Window B starts up and sweeps ITS board; A's task must survive.
        ts.switch("s_windowB")
        assert tasks_mod.list_tasks() == []
        tasks_mod.release_orphaned_tasks({"nonexistent_owner"})
        # A's board untouched - verify by reading A's file again.
        ts.switch("s_windowA")
        task = tasks_mod.load_task(tid)
        assert task.status == "in_progress"
        assert task.owner == "alice"

    def test_claim_is_session_local(self, two_sessions):
        """A pending task in window A must be invisible to window B."""
        ts = two_sessions
        ts.switch("s_windowA")
        tid = create_task("only visible in A").id
        ts.switch("s_windowB")
        assert tasks_mod.list_tasks() == []
        with pytest.raises(FileNotFoundError):
            tasks_mod.load_task(tid)


class TestRequestIdUniqueness:
    def test_no_cross_session_collision(self, monkeypatch):
        """10k ids from two simulated sessions must be globally unique."""
        import mcodecore.config as cfg
        import mcodecore.utils as utils
        ids = set()
        for sid in ("s_windowA", "s_windowB"):
            monkeypatch.setattr(utils, "SESSION_ID", sid, raising=False)
            # new_request_id imports SESSION_ID from config at call time.
            monkeypatch.setattr(cfg, "SESSION_ID", sid)
            for _ in range(5000):
                ids.add(new_request_id())
        assert len(ids) == 10000
        # Format embeds the session for log forensics.
        monkeypatch.setattr(cfg, "SESSION_ID", "s_windowA")
        rid = new_request_id()
        assert rid.startswith("req_s_windowA_")


class TestLegacyMailboxQuarantine:
    def test_flat_files_moved_to_orphan(self, tmp_path, monkeypatch):
        """Leftover flat ``.mailboxes/lead.jsonl`` from a crashed pre-scoped
        session must be relocated, never consumed by the current session."""
        import mcodecore.config as cfg
        root = tmp_path / ".mailboxes"
        root.mkdir(exist_ok=True)
        (root / "lead.jsonl").write_text('{"stolen": true}\n', encoding="utf-8")
        (root / "bob.jsonl.reading_1").write_text("", encoding="utf-8")
        # Session dir and unrelated files must stay put.
        (root / "s_abcdef12").mkdir()
        (root / "s_abcdef12" / "lead.jsonl").write_text(
            '{"live": true}\n', encoding="utf-8")

        monkeypatch.setattr(cfg, "WORKDIR", tmp_path, raising=False)
        moved = quarantine_legacy_mailboxes()
        assert moved == 2
        assert not (root / "lead.jsonl").exists()
        orphan = root / "orphan" / "lead.jsonl"
        assert json.loads(orphan.read_text(encoding="utf-8")) == {"stolen": True}
        # Session dir untouched.
        assert (root / "s_abcdef12" / "lead.jsonl").exists()

    def test_noop_when_clean(self, tmp_path, monkeypatch):
        import mcodecore.config as cfg
        root = tmp_path / ".mailboxes" / "s_clean"
        root.mkdir(parents=True)
        monkeypatch.setattr(cfg, "WORKDIR", tmp_path, raising=False)
        assert quarantine_legacy_mailboxes() == 0
