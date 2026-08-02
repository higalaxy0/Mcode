"""Task board CRUD.

All task mutations (``claim_task``, ``complete_task``) are guarded by a
process-wide ``threading.Lock`` so that concurrent teammates cannot race on
the load-check-save cycle (TOCTOU).  Task IDs use ``uuid4`` to eliminate
same-second collisions.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import TASKS_DIR
from .context import ctx


# Process-wide lock for task mutations (claim, complete).
# This prevents the TOCTOU race where two agents load the same pending task,
# both pass the checks, and the last writer silently overwrites the owner.
_task_lock = ctx.memory_lock  # reuse existing lock to avoid adding new state


@dataclass
class Task:
    """A single task on the task board."""
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str]


def _task_path(task_id: str) -> Path:
    """Return the JSON file path for a task."""
    return TASKS_DIR / f"{task_id}.json"


def create_task(subject: str, description: str = "",
                blockedBy: list[str] | None = None) -> Task:
    """Create and persist a new task."""
    task = Task(
        id=f"task_{uuid.uuid4().hex[:12]}",
        subject=subject,
        description=description,
        status="pending",
        owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task


def save_task(task: Task) -> None:
    """Write a task to disk as JSON."""
    _task_path(task.id).write_text(
        json.dumps(asdict(task), indent=2, ensure_ascii=False))


def load_task(task_id: str) -> Task:
    """Load a task from disk."""
    return Task(**json.loads(_task_path(task_id).read_text()))


def list_tasks() -> list[Task]:
    """List all tasks (sorted by filename)."""
    return [Task(**json.loads(p.read_text()))
            for p in sorted(TASKS_DIR.glob("task_*.json"))]


def get_task(task_id: str) -> str:
    """Return a task as a JSON string."""
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2, ensure_ascii=False)


def _deps_ready(task: Task) -> bool:
    """Check whether all blocking dependencies of *task* are completed.

    Uses the already-loaded task object instead of re-reading from disk
    (avoids the redundant double-load in the old ``can_start``).
    """
    for dep_id in task.blockedBy:
        dep_path = _task_path(dep_id)
        if not dep_path.exists():
            return False
        if load_task(dep_id).status != "completed":
            return False
    return True


def can_start(task_id: str) -> bool:
    """Check whether all blocking dependencies of a task are completed."""
    task = load_task(task_id)
    return _deps_ready(task)


def claim_task(task_id: str, owner: str = "agent") -> str:
    """Claim a pending task.

    Sets status to ``in_progress`` and records the owner.
    The entire load-check-save sequence is guarded by ``_task_lock`` so
    concurrent callers cannot both see ``pending`` and overwrite each other.
    """
    with _task_lock:
        task = load_task(task_id)
        if task.status != "pending":
            return f"Task {task_id} is {task.status},cannot claim"
        if task.owner:
            return f"Task {task_id} already owned by {task.owner}"
        if not _deps_ready(task):
            deps = [d for d in task.blockedBy
                    if not _task_path(d).exists() or load_task(d).status != "completed"]
            missing = [d for d in task.blockedBy if not _task_path(d).exists()]
            parts = []
            if deps:
                parts.append(f"blocked by: {deps}")
            if missing:
                parts.append(f"missing deps: {missing}")
            return "Cannot start - " + ",".join(parts)
        task.owner = owner
        task.status = "in_progress"
        save_task(task)
        print(f"  \033[36m[claim] {task.subject} -> in_progress (owner:{owner})\033[0m")
        return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str, owner: str | None = None) -> str:
    """Complete an in_progress task and report unblocked downstream tasks.

    If *owner* is provided, the task's owner must match before completion
    is allowed (prevents a teammate from completing a task it does not own).
    """
    with _task_lock:
        task = load_task(task_id)
        if task.status != "in_progress":
            return f"Task {task_id} is {task.status},cannot complete"
        if owner is not None and task.owner and task.owner != owner:
            return (f"Task {task_id} is owned by {task.owner},"
                    f"not {owner},cannot complete")
        task.status = "completed"
        save_task(task)
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy and can_start(t.id)]
    print(f"  \033[32m[completed] {task.subject} \033[0m")
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked:{','.join(unblocked)}"
        print(f" \033[33m[unblocked]{', '.join(unblocked)}\033[0m")
    return msg


def list_owned_inprogress(owner: str) -> list[dict]:
    """Return all in_progress tasks owned by *owner*.

    Used by idle_poll to detect tasks the agent itself owns but has not
    yet completed (e.g. after turn-budget exhaustion).  Corrupt or
    unreadable task files are skipped.
    """
    owned = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        try:
            task = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if (task.get("status") == "in_progress"
                and task.get("owner") == owner):
            owned.append(task)
    return owned


def release_task(task_id: str, owner: str | None = None) -> str:
    """Release an in_progress task back to pending so others can claim it.

    Resets status to ``pending`` and clears the owner.  If *owner* is
    provided, the task's owner must match before release is allowed
    (prevents a teammate from releasing a task it does not own).
    """
    with _task_lock:
        task = load_task(task_id)
        if task.status != "in_progress":
            return f"Task {task_id} is {task.status},cannot release"
        if owner is not None and task.owner and task.owner != owner:
            return (f"Task {task_id} is owned by {task.owner},"
                    f"not {owner},cannot release")
        task.owner = None
        task.status = "pending"
        save_task(task)
    print(f"  \033[33m[release] {task.subject} -> pending "
          f"(was owner:{owner})\033[0m")
    return f"Released {task.id} ({task.subject})"


def scan_unclaimed_tasks() -> list[dict]:
    """Scan for all claimable pending tasks (no owner and dependencies met).

    Corrupt or unreadable task files are skipped instead of crashing the
    entire scan (which would take down all idle teammates).
    """
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        try:
            task = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            print(f"  \033[31m[scan] skipping corrupt task file {f.name}: "
                  f"{e}\033[0m")
            continue
        try:
            tid = task["id"]
        except (KeyError, TypeError):
            continue
        if (task.get("status") == "pending"
                and not task.get("owner")
                and can_start(tid)):
            unclaimed.append(task)
    return unclaimed
