"""mcode thin-entry re-export completeness tests.

Verifies that all public symbols re-exported by ``mcode`` are accessible
and have the correct type.
"""

from __future__ import annotations

import mcode


def test_config_symbols():
    assert isinstance(mcode.API_BASE, str)
    assert mcode.LLM_MODEL
    assert mcode.WORKDIR is not None
    assert mcode.TASKS_DIR is not None
    assert mcode.MAILBOX_DIR is not None
    assert mcode.MEMORY_DIR is not None
    assert mcode.SKILLS_DIR is not None
    assert mcode.TRANSCRIPT_DIR is not None
    assert mcode.BASH_TIMEOUT > 0
    assert mcode.CONTEXT_LIMIT > 0


def test_context_symbols():
    from mcodecore.context import AppContext
    assert isinstance(mcode.ctx, AppContext)


def test_exception_symbol():
    assert issubclass(mcode.AgentInterrupt, Exception)


def test_utils_symbols():
    assert callable(mcode.parse_frontmatter)
    assert callable(mcode.parse_bg_command)
    assert callable(mcode.parse_explicit_timeout)
    assert callable(mcode.truncate)
    assert callable(mcode.new_request_id)


def test_tasks_symbols():
    from mcodecore.tasks import Task
    assert mcode.Task is Task
    assert callable(mcode.create_task)
    assert callable(mcode.save_task)
    assert callable(mcode.load_task)
    assert callable(mcode.list_tasks)
    assert callable(mcode.get_task)
    assert callable(mcode.can_start)
    assert callable(mcode.claim_task)
    assert callable(mcode.complete_task)
    assert callable(mcode.scan_unclaimed_tasks)


def test_bus_symbols():
    assert mcode.MessageBus
    assert mcode.ProtocolState
    assert callable(mcode.match_response)
    assert callable(mcode.consume_lead_inbox)
    assert callable(mcode.idle_poll)
    assert callable(mcode.run_send_message)
    assert callable(mcode.run_check_inbox)
    assert callable(mcode.run_request_shutdown)
    assert callable(mcode.run_request_plan)
    assert callable(mcode.run_review_plan)


def test_hooks_symbols():
    assert callable(mcode.register_hook)
    assert callable(mcode.trigger_hooks)
    assert callable(mcode.permission_hook)
    assert callable(mcode.log_hook)
    assert callable(mcode.context_inject_hook)
    assert callable(mcode.summary_hook)
    assert callable(mcode.install_default_hooks)


def test_skills_symbols():
    assert isinstance(mcode.SKILL_REGISTRY, dict)
    assert callable(mcode.list_skills)
    assert callable(mcode.load_skill)


def test_memory_symbols():
    assert callable(mcode.write_memory_file)
    assert callable(mcode.read_memory_index)
    assert callable(mcode.read_memory_file)
    assert callable(mcode.list_memory_files)
    assert callable(mcode.select_relevant_memories)
    assert callable(mcode.load_memories)
    assert callable(mcode.extract_memories)


def test_fsops_symbols():
    assert callable(mcode.safe_path)
    assert callable(mcode.run_bash)
    assert callable(mcode.run_read)
    assert callable(mcode.run_write)
    assert callable(mcode.run_edit)
    assert callable(mcode.run_glob)
    assert callable(mcode.run_grep)


def test_streaming_symbols():
    assert mcode.ToolCall
    assert mcode.ToolCallFunction
    assert mcode.StreamMessage
    assert mcode.StreamChoice
    assert mcode.StreamResponse
    assert callable(mcode.stream_response)


def test_calibrator_symbol():
    assert mcode.TokenCalibrator


def test_compact_symbols():
    assert callable(mcode.estimate_tokens_messages)
    assert callable(mcode.ensure_valid_start)
    assert callable(mcode.group_turns)
    assert callable(mcode.snip_compact)
    assert callable(mcode.micro_compact)
    assert callable(mcode.persist_large_output)
    assert callable(mcode.tool_result_budget)
    assert callable(mcode.write_transcript)
    assert callable(mcode.summarize_history)
    assert callable(mcode.compact_history)
    assert callable(mcode.reactive_compact)


def test_tools_symbols():
    assert callable(mcode.build_system)
    assert isinstance(mcode.SUB_SYSTEM, str)
    assert isinstance(mcode.TOOLS, list)
    assert isinstance(mcode.SUB_TOOLS, list)
    assert isinstance(mcode.TEAMMATE_TOOLS, list)
    assert isinstance(mcode.TOOL_HANDLERS, dict)
    assert isinstance(mcode.SUB_HANDLERS, dict)
    assert callable(mcode.run_todo_write)
    assert callable(mcode.run_create_task)
    assert callable(mcode.run_list_tasks)
    assert callable(mcode.run_get_task)
    assert callable(mcode.run_claim_task)
    assert callable(mcode.run_complete_task)
    assert callable(mcode.run_spawn_teammate)


def test_agent_symbols():
    assert callable(mcode.agent_loop)
    assert callable(mcode._run_agent_turn)
    assert callable(mcode._drain_inbox)
    assert callable(mcode.main)
    assert callable(mcode.spawn_subagent)
    assert callable(mcode.spawn_teammate_thread)


def test_tool_table_count_consistency():
    assert len(mcode.TOOLS) == 21
    assert len(mcode.TOOL_HANDLERS) == 21
