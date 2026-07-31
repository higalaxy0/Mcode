"""Module-level configuration constants and OpenAI client.

These are *configuration* (fixed at startup, immutable at runtime) and are kept
as module-level constants rather than in :class:`AppContext`, which only holds
*mutable runtime state*.
"""

from __future__ import annotations

import os
import platform
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

MEMORY_DIR: Path = WORKDIR / ".memory"
MEMORY_DIR.mkdir(exist_ok=True)
MEMORY_INDEX: Path = MEMORY_DIR / "MEMORY.md"
SKILLS_DIR: Path = AGENT_DIR / "skills"
TRANSCRIPT_DIR: Path = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR: Path = WORKDIR / ".task_outputs" / "tool-results"
TASKS_DIR: Path = WORKDIR / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)
DURABLE_PATH: Path = WORKDIR / ".scheduled_tasks.json"
MAILBOX_DIR: Path = WORKDIR / ".mailboxes"
MAILBOX_DIR.mkdir(exist_ok=True)
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

IDLE_POLL_INTERVAL: int = 5
IDLE_TIMEOUT: int = 60
MAX_REACTIVE_RETRIES: int = 3

_DENY_LIST: list[str] = ["sudo", "shutdown", "reboot", "mkfs", "dd if=", "REN"]

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
