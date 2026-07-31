"""Tool schemas, handler mapping, and system prompt."""

from __future__ import annotations

import platform

from .config import WORKDIR
from .context import ctx
from .fsops import (run_bash, run_read, run_write, run_edit, run_glob, run_grep)
from .skills import list_skills
from .tasks import (create_task, list_tasks, get_task, claim_task, complete_task)
from .memory import read_memory_index


# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #

def build_system() -> str:
    """Assemble the lead agent system prompt."""
    catalog = list_skills()
    index = read_memory_index()
    memories_section = f"\n\nMemories available:\n{index}" if index else ""
    return (
        f"You are a coding agent at {WORKDIR}.Act,don't explain."
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
        f"For complex sub-problems, use the subagent tool to spawn a subagent. "
        f"OS is {platform.system()},ensure commands and file paths compatible with {platform.system()}."
        f"{memories_section}\n"
        "Relevant memories are injected below. Respect user preferences from memory.\n"
        "When the user says 'remember' or expresses a clear preference, [REMEMBER:...] in your reply and the system automatically extracts it."
    )
    # Append MCP status if any servers are connected
    mcp = ctx.mcp
    if mcp.is_connected:
        system += f"\n\nMCP servers connected:\n{mcp.status()}"
    return system


SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    f"OS is {platform.system()}"
)


# --------------------------------------------------------------------------- #
# Tool schemas
# --------------------------------------------------------------------------- #

def _bash_tool() -> dict:
    return {"type": "function", "function": {
        "name": "bash",
        "description": (f"Run a shell command.NOTE:The current OS is {platform.system()}."
                        "Ensure commands are valid for this environment."
                        "Timeout:300s default;append '# timeout=600' for longer."
                        "Background:prefix with 'bg: ' to run long tasks (monitors,servers) "
                        "in background - returns PID + log path;use read_file to check output later."),
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}


def _read_file_tool() -> dict:
    return {"type": "function", "function": {
        "name": "read_file",
        "description": "Read file contents. Returns lines with line-number prefixes. Use offset to start from a specific line and limit to control how many lines to read.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to workspace root."},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-based). Defaults to 1."},
            "limit": {"type": "integer", "description": "Maximum number of lines to read. Defaults to 2000."}},
            "required": ["path"]}}}


def _write_file_tool() -> dict:
    return {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}}


def _edit_file_tool() -> dict:
    return {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace exact text in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            "required": ["path", "old_text", "new_text"]}}}


def _glob_tool() -> dict:
    return {"type": "function", "function": {
        "name": "glob",
        "description": "Find files matching a glob pattern. Supports `**` for recursive directory matching (e.g. `**/*.py`). Returns paths relative to workspace root using forward slashes, sorted by modification time (newest first), limited to 100 results.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. `**/*.py` or `src/**/*.ts`."}},
            "required": ["pattern"]}}}


def _grep_tool() -> dict:
    return {"type": "function", "function": {
        "name": "grep",
        "description": "Search file contents for a regex pattern. Returns matching lines with file:line prefixes. Searches recursively under the given path (relative to workdir).",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {"type": "string", "description": "File or directory to search in (relative to workdir). Defaults to '.'."},
            "include": {"type": "string", "description": "Glob pattern to filter filenames, e.g. '*.py'."}},
            "required": ["pattern"]}}}


def _send_message_tool() -> dict:
    return {"type": "function", "function": {
        "name": "send_message",
        "description": "Send a message to another agent.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "The identifier or name of the target agent."},
            "content": {"type": "string", "description": "The message content to send."}},
            "required": ["to", "content"]}}}


def _submit_plan_tool() -> dict:
    return {"type": "function", "function": {
        "name": "submit_plan",
        "description": "Submit a plan for Lead approval.",
        "parameters": {"type": "object", "properties": {"plan": {"type": "string"}},
                       "required": ["plan"]}}}


def _list_tasks_tool() -> dict:
    return {"type": "function", "function": {
        "name": "list_tasks",
        "description": "List all tasks with status, owner, and dependencies.",
        "parameters": {"type": "object", "properties": {
            "include_completed": {"type": "boolean", "description": "If true,include completed tasks in the listing."}},
            "required": ["include_completed"]}}}


def _claim_task_tool() -> dict:
    return {"type": "function", "function": {
        "name": "claim_task",
        "description": "Claim a pending task.",
        "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}},
                       "required": ["task_id"]}}}


def _complete_task_tool() -> dict:
    return {"type": "function", "function": {
        "name": "complete_task",
        "description": "Mark an in-progress task as completed.",
        "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}},
                       "required": ["task_id"]}}}


# -- Base 6 tools (shared by subagent and teammate) --------------------------- #
_BASE_TOOLS = [_bash_tool(), _read_file_tool(), _write_file_tool(),
               _edit_file_tool(), _glob_tool(), _grep_tool()]

# Tools available to subagents (6 base tools)
SUB_TOOLS: list[dict] = list(_BASE_TOOLS)

# Tools available to teammates (6 base + send_message + submit_plan +
# list_tasks + claim_task + complete_task)
TEAMMATE_TOOLS: list[dict] = _BASE_TOOLS + [
    _send_message_tool(), _submit_plan_tool(),
    _list_tasks_tool(), _claim_task_tool(), _complete_task_tool(),
]

# Full tool set for the lead agent
TOOLS: list[dict] = SUB_TOOLS + [
    {"type": "function", "function": {
        "name": "todo_write",
        "description": "Create and manage a task list for your current coding session.",
        "parameters": {"type": "object", "properties": {
            "todos": {"type": "array", "items": {"type": "object", "properties": {
                "content": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
                "required": ["content", "status"]}}},
            "required": ["todos"]}}},
    {"type": "function", "function": {
        "name": "subagent",
        "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
        "parameters": {"type": "object", "properties": {"description": {"type": "string"}},
                       "required": ["description"]}}},
    {"type": "function", "function": {
        "name": "load_skill",
        "description": "Load the full content of a skill by name.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "create_task",
        "description": "Create a new task with optional blockedBy dependencies.",
        "parameters": {"type": "object", "properties": {
            "subject": {"type": "string"}, "description": {"type": "string"},
            "blockedBy": {"type": "array", "items": {"type": "string"}}},
            "required": ["subject"]}}},
    {"type": "function", "function": {
        "name": "list_tasks",
        "description": "List all tasks with status, owner, and dependencies.",
        "parameters": {"type": "object", "properties": {
            "include_completed": {"type": "boolean", "description": "If true,include completed tasks in the listing."}},
            "required": ["include_completed"]}}},
    {"type": "function", "function": {
        "name": "get_task",
        "description": "Get full details of a specific tasks by ID",
        "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}},
                       "required": ["task_id"]}}},
    {"type": "function", "function": {
        "name": "claim_task",
        "description": "Claim a pending task.Sets owner, changes status to in_progress",
        "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}},
                       "required": ["task_id"]}}},
    {"type": "function", "function": {
        "name": "complete_task",
        "description": "Complete an in-progress task. Reports unblocked downstream tasks",
        "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}},
                       "required": ["task_id"]}}},
    {"type": "function", "function": {
        "name": "spawn_teammate",
        "description": "Spawn an autonomous teammate.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}},
            "required": ["name", "role", "prompt"]}}},
    {"type": "function", "function": {
        "name": "send_message",
        "description": "Send message to a teammate.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "content": {"type": "string"}},
            "required": ["to", "content"]}}},
    {"type": "function", "function": {
        "name": "check_inbox",
        "description": "Check inbox for messages and protocol responses.",
        "parameters": {"type": "object", "properties": {
            "include_read": {"type": "boolean", "description": "If true, include already-read messages in the result. Defaults to false. "}},
            "required": ["include_read"]}}},
    {"type": "function", "function": {
        "name": "request_shutdown",
        "description": "Request a teammate to shut down.",
        "parameters": {"type": "object", "properties": {"teammate": {"type": "string"}},
                       "required": ["teammate"]}}},
    {"type": "function", "function": {
        "name": "request_plan",
        "description": "Ask a teammate to submit a plan for review.",
        "parameters": {"type": "object", "properties": {
            "teammate": {"type": "string"}, "task": {"type": "string"}},
            "required": ["teammate", "task"]}}},
    {"type": "function", "function": {
        "name": "review_plan",
        "description": "Approve or reject a submitted plan by request_id.",
        "parameters": {"type": "object", "properties": {
            "request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}},
            "required": ["request_id", "approve"]}}},
]


# --------------------------------------------------------------------------- #
# Tool handlers
# --------------------------------------------------------------------------- #

# The 6 base handlers used by subagents
SUB_HANDLERS: dict = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "grep": run_grep,
}


def run_todo_write(todos: list) -> str:
    """Update the todo list for the current session."""
    for i, t in enumerate(todos):
        if "content" not in t or "status" not in t:
            return f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return f"Error: todos[{i}] has invalid status '{t['status']}'"
    ctx.current_todos = todos
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for i, t in enumerate(ctx.current_todos):
        icon = {"pending": " ", "in_progress": "\033[36m▸\033[0m", "completed": "\033[31m✓\033[0m"}[t["status"]]
        lines.append(f"  [{icon}] #{str(i+1)}: {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(ctx.current_todos)} tasks"


def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    """Create a task and print it."""
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks(include_completed: bool = True) -> str:
    """List tasks (optionally excluding completed ones)."""
    if isinstance(include_completed, str):
        include_completed = include_completed.strip().lower() == "true"
    include_completed = bool(include_completed)
    tasks = list_tasks()
    if not include_completed:
        tasks = [t for t in tasks if t.status != "completed"]
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for t in tasks:
        icon = {"pending": "○", "in_progress": "●", "completed": "✓"}.get(t.status, "?")
        deps = f" (blockedBy: {', '.join(t.blockedBy)})" if t.blockedBy else ""
        owner = f" [{t.owner}]" if t.owner else ""
        lines.append(f"  {icon} {t.id}: {t.subject} [{t.status}]{owner}{deps}")
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    """Get task details."""
    try:
        return get_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


def run_claim_task(task_id: str) -> str:
    """Claim a task (owner=agent)."""
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    """Complete a task."""
    return complete_task(task_id)


def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    """Spawn a teammate (forwards to the teammates module)."""
    from .teammates import spawn_teammate_thread
    return spawn_teammate_thread(name, role, prompt)


# Full handler mapping for the lead agent
TOOL_HANDLERS: dict = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "grep": run_grep,
    "todo_write": run_todo_write,
    "subagent": None,           # filled lazily to avoid circular imports
    "load_skill": None,         # filled lazily
    "create_task": run_create_task, "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_claim_task, "complete_task": run_complete_task,
    "spawn_teammate": run_spawn_teammate,
    "send_message": None, "check_inbox": None,
    "request_shutdown": None, "request_plan": None, "review_plan": None,
}


def _inject_mcp_tools() -> None:
    """Inject all connected MCP server tools into the tool sets and handlers.

    Called from :func:`agent.main` after MCP initialization.  When no MCP
    server is connected this is a no-op, so the tool sets remain unchanged
    and existing behavior is preserved.
    """
    mcp = ctx.mcp
    if not mcp.is_connected:
        return
    schemas = mcp.list_all_tool_schemas()
    if not schemas:
        return
    handlers = mcp.build_handlers()
    # Detect name collisions with built-in tools
    existing = {t["function"]["name"] for t in TOOLS}
    for schema in schemas:
        name = schema["function"]["name"]
        if name in existing:
            print(f"\033[33m[MCP] Skipping duplicate tool name: {name}\033[0m")
            continue
        TOOLS.append(schema)
        SUB_TOOLS.append(schema)
        TOOL_HANDLERS[name] = handlers[name]
        SUB_HANDLERS[name] = handlers[name]


def _fill_delayed_handlers() -> None:
    """Fill in handlers that require lazy imports (to avoid circular imports)."""
    from .subagent import spawn_subagent
    from .skills import load_skill
    from .bus import (run_send_message, run_check_inbox,
                      run_request_shutdown, run_request_plan, run_review_plan)
    TOOL_HANDLERS["subagent"] = spawn_subagent
    TOOL_HANDLERS["load_skill"] = load_skill
    TOOL_HANDLERS["send_message"] = run_send_message
    TOOL_HANDLERS["check_inbox"] = run_check_inbox
    TOOL_HANDLERS["request_shutdown"] = run_request_shutdown
    TOOL_HANDLERS["request_plan"] = run_request_plan
    TOOL_HANDLERS["review_plan"] = run_review_plan


_fill_delayed_handlers()
