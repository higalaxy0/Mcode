#!/usr/bin/env python3
"""mcode entry point.

Thin re-export shim over the ``mcodecore`` package.
"""

from __future__ import annotations

# -- Re-export public symbols for backward compatibility --------------------- #
from mcodecore.config import (
    API_BASE, API_KEY, LLM_MODEL, client,
    AGENT_DIR, WORKDIR, MEMORY_DIR, MEMORY_INDEX, SKILLS_DIR,
    TRANSCRIPT_DIR, TOOL_RESULTS_DIR, TASKS_DIR, DURABLE_PATH,
    MAILBOX_DIR, _BG_OUTPUT_DIR, BASH_TIMEOUT, CONTEXT_LIMIT,
    KEEP_RECENT_LOOP_TURN, PERSIST_THRESHOLD, CONSOLIDATE_THRESHOLD,
    IDLE_POLL_INTERVAL, IDLE_TIMEOUT, MAX_REACTIVE_RETRIES,
    _DENY_LIST, _IS_WINDOWS, _enable_ansi,
)
from mcodecore.context import ctx, AppContext
from mcodecore.exceptions import AgentInterrupt
from mcodecore.utils import (parse_frontmatter, parse_bg_command,
                             parse_explicit_timeout, truncate, new_request_id)
from mcodecore.tasks import (Task, create_task, save_task, load_task,
                             list_tasks, get_task, can_start, claim_task,
                             complete_task, scan_unclaimed_tasks)
from mcodecore.bus import (MessageBus, ProtocolState, match_response,
                           consume_lead_inbox, idle_poll,
                           _teammate_submit_plan, run_request_shutdown,
                           run_request_plan, run_review_plan,
                           run_send_message, run_check_inbox)
from mcodecore.hooks import (register_hook, trigger_hooks,
                             permission_hook, log_hook,
                             context_inject_hook, summary_hook,
                             install_default_hooks)
from mcodecore.skills import (SKILL_REGISTRY, list_skills, load_skill)
from mcodecore.memory import (write_memory_file, read_memory_index,
                              read_memory_file, list_memory_files,
                              select_relevant_memories, load_memories,
                              extract_memories, consolidate_memories,
                              _post_turn_memory, _load_memories_async,
                              _await_memories)
from mcodecore.fsops import (safe_path, _kill_process_tree, _run_background,
                             run_bash, run_read, run_write, run_edit,
                             run_glob, run_grep)
from mcodecore.streaming import (ToolCall, ToolCallFunction, StreamMessage,
                                 StreamChoice, StreamResponse, stream_response)
from mcodecore.calibrator import TokenCalibrator
from mcodecore.compact import (estimate_tokens_messages, ensure_valid_start,
                               group_turns, snip_compact, micro_compact,
                               persist_large_output, tool_result_budget,
                               write_transcript, summarize_history,
                               _build_post_compact_context, compact_history,
                               reactive_compact)
from mcodecore.tools import (build_system, SUB_SYSTEM, SUB_TOOLS,
                             TEAMMATE_TOOLS, TOOLS, SUB_HANDLERS,
                             TOOL_HANDLERS, run_todo_write, run_create_task,
                             run_list_tasks, run_get_task, run_claim_task,
                             run_complete_task, run_spawn_teammate)
from mcodecore.subagent import spawn_subagent
from mcodecore.teammates import spawn_teammate_thread
from mcodecore.agent import agent_loop, _run_agent_turn, _drain_inbox, main


if __name__ == "__main__":
    main()
