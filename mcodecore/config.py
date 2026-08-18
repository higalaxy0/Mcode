"""Module-level configuration constants and OpenAI client.

These are *configuration* (fixed at startup, immutable at runtime) and are kept
as module-level constants rather than in :class:`AppContext`, which only holds
*mutable runtime state*.
"""

from __future__ import annotations

import os
import platform
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------------------------------- #
# API configuration
# --------------------------------------------------------------------------- #
API_BASE: str = ""
API_KEY: str = os.getenv("API_KEY", "")
LLM_MODEL: str = ""

# load_dotenv silently no-ops when .env is absent.
load_dotenv(override=True)

client: OpenAI = OpenAI(api_key=API_KEY, base_url=API_BASE)

# --------------------------------------------------------------------------- #
# Path constants
# --------------------------------------------------------------------------- #
AGENT_DIR: Path = Path(__file__).resolve().parent.parent          # points to src/
WORKDIR: Path = Path.cwd()

# Session id: unique per mcode process.  Multiple mcode windows opened in
# the SAME folder would otherwise share one flat on-disk namespace
# (``.mailboxes/lead.jsonl``, ``.tasks/``, ``.team_history/``) and steal
# each other's messages (read_inbox consumes via rename) and tasks
# (orphan sweep / claim races).  Scoping every cross-agent directory by
# SESSION_ID makes interference structurally impossible - a team (lead +
# teammates) always lives inside one process.  ``MCODE_SESSION_ID`` lets
# tests (or users) pin a known session for inspection.
SESSION_ID: str = os.getenv("MCODE_SESSION_ID") or f"s_{uuid.uuid4().hex[:8]}"

MEMORY_DIR: Path = WORKDIR / ".memory"
MEMORY_DIR.mkdir(exist_ok=True)
MEMORY_INDEX: Path = MEMORY_DIR / "MEMORY.md"
SKILLS_DIR: Path = AGENT_DIR / "skills"
TRANSCRIPT_DIR: Path = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR: Path = WORKDIR / ".task_outputs" / "tool-results"
# Task board and mailboxes are session-scoped (see SESSION_ID above).
TASKS_DIR: Path = WORKDIR / ".tasks" / SESSION_ID
TASKS_DIR.mkdir(parents=True, exist_ok=True)
DURABLE_PATH: Path = WORKDIR / ".scheduled_tasks.json"
MAILBOX_DIR: Path = WORKDIR / ".mailboxes" / SESSION_ID
MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
TEAM_HISTORY_DIR: Path = WORKDIR / ".team_history" / SESSION_ID
_BG_OUTPUT_DIR: Path = WORKDIR / ".task_outputs" / "bg-logs"

# MCP (Model Context Protocol) server configuration file path
MCP_CONFIG_PATH: Path = WORKDIR / ".mcp.json"

# --------------------------------------------------------------------------- #
# Miscellaneous constants
# --------------------------------------------------------------------------- #
MEMORY_TYPES: list[str] = ["user", "feedback", "project", "reference"]

BASH_TIMEOUT: int = 300
CONTEXT_LIMIT: int = 115200          # 128000 * 0.9
KEEP_RECENT_LOOP_TURN: int = 25
PERSIST_THRESHOLD: int = 30000
_MSG_OVERHEAD_TOKENS: int = 4
CONSOLIDATE_THRESHOLD: int = 10
# Four-layer gating (inspired by Claude Code autoDream):
#   Gate 0 (hard limit): file count >= CONSOLIDATE_HARD_LIMIT -> force merge
#   Gate 1 (count floor): file count >= CONSOLIDATE_THRESHOLD
#   Gate 2 (time cooldown): CONSOLIDATE_MIN_INTERVAL seconds since last merge
#   Gate 3 (activity): CONSOLIDATE_MIN_TRANSCRIPTS new transcripts since last merge
#   Gate 4 (cross-process lock): .consolidate-lock must be acquirable
CONSOLIDATE_HARD_LIMIT: int = 50
CONSOLIDATE_MIN_INTERVAL: int = 86400       # 24 hours
CONSOLIDATE_MIN_TRANSCRIPTS: int = 5
CONSOLIDATE_LOCK_STALE: int = 600           # 10 minutes; stale lock is stealable
# Scan-throttle cache: list_memory_files() results cached for this many seconds.
MEMORY_CACHE_TTL: int = 30

IDLE_POLL_INTERVAL: int = 5
IDLE_TIMEOUT: int = 360
MAX_REACTIVE_RETRIES: int = 3

# Teammate turn-budget controls (Fix #1 A+C):
#   TURN_BUDGET          - soft cap per work cycle; exhausted -> idle_poll
#   TURN_BUDGET_RENEWAL  - extra turns granted when the worker still owns
#                          in_progress tasks at exhaustion
#   TURN_BUDGET_HARD_CAP - absolute maximum total turns per work cycle,
#                          prevents infinite renewal loops
#   CLAIM_MIN_TURNS      - minimum remaining turns required to claim a new
#                          task; prevents claiming when budget is nearly
#                          exhausted (Fix #1C)
TURN_BUDGET: int = 50
TURN_BUDGET_RENEWAL: int = 20
TURN_BUDGET_HARD_CAP: int = 100
CLAIM_MIN_TURNS: int = 10

_DENY_LIST: list[str] = ["sudo", "shutdown", "reboot", "mkfs", "dd if=", "REN"]

# --------------------------------------------------------------------------- #
# Verbosity gate
# --------------------------------------------------------------------------- #
# MCODE_VERBOSE=1 enables debug-level prints (compaction internals, memory
# background activity, bus routing, idle-poll scheduling).  Default is 0
# (quiet) so that only high-signal messages reach the user.
VERBOSE: bool = os.getenv("MCODE_VERBOSE", "0") == "1"


def debug(msg: str) -> None:
    """Print *msg* only when MCODE_VERBOSE=1 is set.

    Use for diagnostic output that is useful when troubleshooting but
    would otherwise dilute the user's attention during normal operation.
    """
    if VERBOSE:
        print(msg)


_IS_WINDOWS: bool = platform.system() == "Windows"
_POPEN_KWARGS: dict = {}
if _IS_WINDOWS:
    import subprocess
    _POPEN_KWARGS["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP


def _enable_ansi() -> None:
    """Enable ANSI escape code processing on the Windows console."""
    if platform.system() == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


def quarantine_legacy_mailboxes() -> int:
    """Move legacy flat mailbox files left by pre-session-scoped versions.

    Before mailboxes became session-scoped, every mcode process in the same
    folder shared ``.mailboxes/<agent>.jsonl``; a leftover file from a
    crashed session would otherwise be readable by the CURRENT session's
    lead, which is precisely the cross-window message-theft bug.  Move any
    flat ``.jsonl`` (and stale ``.reading_*`` temps) into
    ``.mailboxes/orphan/`` instead of consuming them.  Returns the number
    of relocated files.
    """
    root = WORKDIR / ".mailboxes"
    if not root.is_dir():
        return 0
    moved = 0
    for entry in root.iterdir():
        # Session dirs (``s_xxxxxxxx``) and the orphan dir stay put.
        if entry.is_dir():
            continue
        if entry.suffix != ".jsonl" and ".reading_" not in entry.name:
            continue
        orphan = root / "orphan"
        orphan.mkdir(exist_ok=True)
        target = orphan / entry.name
        # A same-named orphan may already exist (repeated crashes); don't
        # clobber it - suffix with a counter instead.
        counter = 1
        while target.exists():
            target = orphan / f"{entry.stem}_{counter}{entry.suffix}"
            counter += 1
        try:
            entry.rename(target)
            moved += 1
        except OSError:
            pass  # locked by another process; leave it, it will be retried next run
    return moved
