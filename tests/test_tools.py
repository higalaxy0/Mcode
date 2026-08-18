"""Tool definition table, handler mapping and system prompt tests.

Covers ``mcodecore.tools``:
build_system / SUB_SYSTEM / TOOLS / SUB_TOOLS / TEAMMATE_TOOLS /
TOOL_HANDLERS / SUB_HANDLERS / run_todo_write / run_create_task /
run_list_tasks / run_get_task / run_claim_task / run_complete_task.
"""

from __future__ import annotations

from mcodecore import tools
from mcodecore.context import ctx


# --------------------------------------------------------------------------- #
# Tool table completeness
# --------------------------------------------------------------------------- #

def test_tools_has_20_entries():
    assert len(tools.TOOLS) == 21


def test_sub_tools_has_6_base():
    assert len(tools.SUB_TOOLS) == 6
    names = {t["function"]["name"] for t in tools.SUB_TOOLS}
    assert names == {"bash", "read_file", "write_file", "edit_file", "glob", "grep"}


def test_teammate_tools_has_11():
    assert len(tools.TEAMMATE_TOOLS) == 11


def test_tool_handlers_keys_match_tools():
    tool_names = {t["function"]["name"] for t in tools.TOOLS}
    handler_names = set(tools.TOOL_HANDLERS.keys())
    assert tool_names == handler_names


def test_all_handlers_filled_no_none():
    # _fill_delayed_handlers already ran at import; there should be no None
    for name, h in tools.TOOL_HANDLERS.items():
        assert h is not None, f"handler for {name} is None"


def test_sub_handlers_keys_match_sub_tools():
    sub_names = {t["function"]["name"] for t in tools.SUB_TOOLS}
    assert set(tools.SUB_HANDLERS.keys()) == sub_names


def test_tool_schemas_have_required_fields():
    for t in tools.TOOLS:
        assert t["type"] == "function"
        fn = t["function"]
        assert "name" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        # required fields are all string arrays (or absent)
        if "required" in fn["parameters"]:
            assert all(isinstance(r, str) for r in fn["parameters"]["required"])


# --------------------------------------------------------------------------- #
# build_system / SUB_SYSTEM
# --------------------------------------------------------------------------- #

def test_build_system_contains_workdir():
    s = tools.build_system()
    assert "coding agent" in s
    assert "Skills available" in s


def test_build_system_includes_skills_catalog():
    s = tools.build_system()
    # Shows (no skills found) when there are no skills
    assert "Skills available" in s


def test_sub_system_is_string():
    assert isinstance(tools.SUB_SYSTEM, str)
    assert "coding agent" in tools.SUB_SYSTEM


# --------------------------------------------------------------------------- #
# run_todo_write
# --------------------------------------------------------------------------- #

def test_run_todo_write_valid():
    res = tools.run_todo_write([
        {"content": "task A", "status": "pending"},
        {"content": "task B", "status": "in_progress"},
        {"content": "task C", "status": "completed"},
    ])
    assert "Updated 3 tasks" in res
    assert len(ctx.current_todos) == 3


def test_run_todo_write_missing_content():
    res = tools.run_todo_write([{"status": "pending"}])
    assert "Error" in res
    assert "content" in res


def test_run_todo_write_missing_status():
    res = tools.run_todo_write([{"content": "x"}])
    assert "Error" in res
    assert "status" in res


def test_run_todo_write_invalid_status():
    res = tools.run_todo_write([{"content": "x", "status": "done"}])
    assert "Error" in res
    assert "invalid status" in res


def test_run_todo_write_replaces_existing():
    tools.run_todo_write([{"content": "first", "status": "pending"}])
    tools.run_todo_write([{"content": "second", "status": "completed"}])
    assert len(ctx.current_todos) == 1
    assert ctx.current_todos[0]["content"] == "second"


# --------------------------------------------------------------------------- #
# run_create_task / run_list_tasks / run_get_task / run_claim / run_complete
# --------------------------------------------------------------------------- #

def test_run_create_task_returns_message():
    res = tools.run_create_task("subject", "desc")
    assert "Created" in res
    assert "task_" in res


def test_run_create_task_with_blockers():
    res = tools.run_create_task("child", "d", blockedBy=["task_x_1"])
    assert "blockedBy: task_x_1" in res


def test_run_list_tasks_empty():
    res = tools.run_list_tasks(include_completed=True)
    assert "No tasks" in res


def test_run_list_tasks_after_create():
    tools.run_create_task("visible")
    res = tools.run_list_tasks(include_completed=True)
    assert "visible" in res


def test_run_list_tasks_excludes_completed():
    t = tools.run_create_task("temp")
    # parse the id
    tid = t.split(":")[0].replace("Created ", "")
    tools.run_claim_task(tid)
    tools.run_complete_task(tid)
    res = tools.run_list_tasks(include_completed=False)
    assert tid not in res


def test_run_list_tasks_string_bool_coercion():
    # include_completed passed as the string "true"
    tools.run_create_task("x")
    res = tools.run_list_tasks(include_completed="true")
    assert "x" in res


def test_run_get_task_existing():
    res = tools.run_create_task("findme")
    tid = res.split(":")[0].replace("Created ", "")
    detail = tools.run_get_task(tid)
    assert "findme" in detail


def test_run_get_task_not_found():
    res = tools.run_get_task("task_nonexistent")
    assert "not found" in res


def test_run_claim_and_complete_flow():
    create_res = tools.run_create_task("flow")
    tid = create_res.split(":")[0].replace("Created ", "")
    claim = tools.run_claim_task(tid)
    assert "Claimed" in claim
    comp = tools.run_complete_task(tid)
    assert "Completed" in comp
