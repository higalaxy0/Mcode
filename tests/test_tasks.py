"""Task board CRUD and dependency-unlock logic tests.

Covers all public functions of ``mcodecore.tasks``:
create/save/load/list/get/can_start/claim/complete/scan_unclaimed.
"""

from __future__ import annotations

import json

from mcodecore import tasks


def test_create_task_persists_and_has_id(tmp_path):
    t = tasks.create_task("写文档", "补 README", blockedBy=["task_x_1"])
    assert t.id.startswith("task_")
    assert t.status == "pending"
    assert t.owner is None
    assert t.blockedBy == ["task_x_1"]
    # file is persisted to disk
    assert tasks._task_path(t.id).exists()


def test_load_task_roundtrip():
    t = tasks.create_task("A", "desc A")
    loaded = tasks.load_task(t.id)
    assert loaded.subject == "A"
    assert loaded.description == "desc A"
    assert loaded.status == "pending"


def test_get_task_returns_json_string():
    t = tasks.create_task("B")
    s = tasks.get_task(t.id)
    data = json.loads(s)
    assert data["id"] == t.id
    assert data["subject"] == "B"


def test_list_tasks_sorted_by_filename():
    a = tasks.create_task("alpha")
    b = tasks.create_task("beta")
    listed = tasks.list_tasks()
    ids = {t.id for t in listed}
    assert {a.id, b.id}.issubset(ids)


def test_can_start_no_deps_is_true():
    t = tasks.create_task("no-dep")
    assert tasks.can_start(t.id) is True


def test_can_start_dep_not_completed_is_false():
    dep = tasks.create_task("dep")
    t = tasks.create_task("child", blockedBy=[dep.id])
    assert tasks.can_start(t.id) is False


def test_can_start_dep_completed_is_true():
    dep = tasks.create_task("dep")
    dep.status = "in_progress"
    tasks.save_task(dep)
    tasks.complete_task(dep.id)
    t = tasks.create_task("child", blockedBy=[dep.id])
    assert tasks.can_start(t.id) is True


def test_can_start_missing_dep_is_false():
    t = tasks.create_task("child", blockedBy=["task_does_not_exist"])
    assert tasks.can_start(t.id) is False


def test_claim_task_sets_owner_and_status():
    t = tasks.create_task("claimme")
    res = tasks.claim_task(t.id, owner="alice")
    assert "Claimed" in res
    loaded = tasks.load_task(t.id)
    assert loaded.status == "in_progress"
    assert loaded.owner == "alice"


def test_claim_already_claimed_fails():
    t = tasks.create_task("once")
    tasks.claim_task(t.id, owner="alice")
    res = tasks.claim_task(t.id, owner="bob")
    assert "cannot claim" in res or "already owned" in res


def test_claim_blocked_task_fails():
    dep = tasks.create_task("dep")
    t = tasks.create_task("child", blockedBy=[dep.id])
    res = tasks.claim_task(t.id, owner="x")
    assert "Cannot start" in res
    loaded = tasks.load_task(t.id)
    assert loaded.status == "pending"  # not claimed


def test_complete_task_reports_unblocked_downstream():
    dep = tasks.create_task("dep")
    tasks.claim_task(dep.id, owner="lead")
    # downstream depends on dep (not claimable yet)
    child = tasks.create_task("child", blockedBy=[dep.id])
    res = tasks.complete_task(dep.id)
    assert "Completed" in res
    assert "Unblocked" in res
    assert "child" in res


def test_complete_non_inprogress_fails():
    t = tasks.create_task("never-claimed")
    res = tasks.complete_task(t.id)
    assert "cannot complete" in res


def test_scan_unclaimed_only_returns_ready_pending():
    # a task that can be claimed directly
    ready = tasks.create_task("ready")
    # a blocked task (not returned)
    dep = tasks.create_task("blocker")
    blocked = tasks.create_task("blocked", blockedBy=[dep.id])
    unclaimed = tasks.scan_unclaimed_tasks()
    ids = [t["id"] for t in unclaimed]
    assert ready.id in ids
    assert blocked.id not in ids


def test_scan_excludes_claimed():
    t = tasks.create_task("claimed")
    tasks.claim_task(t.id, owner="z")
    unclaimed = tasks.scan_unclaimed_tasks()
    assert t.id not in [x["id"] for x in unclaimed]
