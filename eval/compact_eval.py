"""Evaluation harness for the compaction layer.

Goal: verify that the compression pipeline keeps the agent oriented in long,
multi-task sessions and supports clean task transitions.  Six realistic
scenarios are generated, each run through the full pipeline
(tool_result_budget -> snip_compact -> micro_compact), then scored against
seven quantitative metrics.

Run:  python -m eval.compact_eval
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Callable

# Force UTF-8 so emoji / box-drawing chars render on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mcodecore import compact
from mcodecore.context import ctx


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _tc(cid: str, name: str = "bash", args: str = "{}") -> dict:
    """An assistant message with a single tool_call."""
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name, "arguments": args}}]}


def _tool(cid: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": cid, "content": content}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _tool_turn(label: str, cid: str, tool_name: str = "bash",
               result: str = "ok") -> list[dict]:
    """A full user -> tool_call -> tool_result mini-turn."""
    return [
        _user(f"[{label}] step"),
        _tc(cid, name=tool_name),
        _tool(cid, result),
    ]


def _realistic_result(label: str, idx: int, ext: str = "py") -> str:
    """Generate a realistic-size tool result (~1500-3000 chars).

    Real ``edit_file`` / ``read_file`` / ``bash`` outputs are hundreds to
    thousands of characters, not bare ``"ok"``.  Using realistic sizes makes
    the compression-ratio metric meaningful and exercises the
    ``micro_compact`` 120-char threshold.
    """
    lines = [
        f"# File: {label}_{idx}.{ext}",
        f"# Auto-generated for scenario testing",
        f"# Path: src/{label}/{label}_{idx}.{ext}",
        f"# Lines: 60",
        "",
        f"import os",
        f"import sys",
        f"import json",
        f"from typing import Dict, List, Optional, Any",
        f"from dataclasses import dataclass, field",
        "",
        f"@dataclass",
        f"class {label.capitalize()}Config{idx}:",
        f'    """Configuration for {label} module {idx}."""',
        f"    name: str = '{label}_{idx}'",
        f"    version: str = '1.0.{idx}'",
        f"    enabled: bool = True",
        f"    options: Dict[str, Any] = field(default_factory=dict)",
        f"",
        f"    def validate(self) -> bool:",
        f"        if not self.name:",
        f"            return False",
        f"        if self.version.count('.') < 2:",
        f"            return False",
        f"        return True",
        f"",
        f"class {label.capitalize()}Module{idx}:",
        f'    """Module {idx} for {label}."""',
        f"    def __init__(self, config: {label.capitalize()}Config{idx}):",
        f"        self.config = config",
        f"        self._state = {{'index': {idx}, 'label': '{label}'}}",
        f"        self._cache: Dict[str, Any] = {{}}",
        f"        self._initialized = False",
        f"",
        f"    def initialize(self) -> None:",
        f"        if self._initialized:",
        f"            return",
        f"        self._initialized = True",
        f"        self._load_defaults()",
        f"",
        f"    def _load_defaults(self) -> None:",
        f"        defaults = {{",
        f"            'timeout': 30,",
        f"            'retries': 3,",
        f"            'batch_size': 100,",
        f"        }}",
        f"        self._cache.update(defaults)",
        f"",
        f"    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:",
        f"        self.initialize()",
        f"        result = {{}}",
        f"        for key, value in payload.items():",
        f"            result[key] = self._transform(value)",
        f"        result['_meta'] = {{'module': self.config.name, 'idx': {idx}}}",
        f"        return result",
        f"",
        f"    def _transform(self, value: Any) -> Any:",
        f"        if isinstance(value, str):",
        f"            return value.strip().lower()",
        f"        elif isinstance(value, (int, float)):",
        f"            return value * 2",
        f"        elif isinstance(value, list):",
        f"            return [self._transform(v) for v in value]",
        f"        return value",
        f"",
        f"    def cleanup(self) -> None:",
        f"        self._cache.clear()",
        f"        self._initialized = False",
        f"",
    ]
    return "\n".join(lines)


def _real_user_prompts(messages: list[dict]) -> list[str]:
    """Extract genuine user task prompts (same rule as _is_task_anchor)."""
    out = []
    for m in messages:
        if compact._is_task_anchor(m):
            out.append(m["content"])
    return out


# --------------------------------------------------------------------------- #
# Metric calculations
# --------------------------------------------------------------------------- #

@dataclass
class MetricResult:
    name: str
    value: float
    target: float
    passed: bool
    detail: str = ""


def _metric_task_identity(before: list[dict], after: list[dict]) -> MetricResult:
    """% of real user prompts that survive compaction."""
    before_prompts = _real_user_prompts(before)
    if not before_prompts:
        return MetricResult("Task Identity Preservation", 1.0, 1.0, True, "no prompts to check")
    after_prompts = set(_real_user_prompts(after))
    kept = sum(1 for p in before_prompts if p in after_prompts)
    ratio = kept / len(before_prompts)
    return MetricResult(
        "Task Identity Preservation", ratio, 1.0, ratio == 1.0,
        f"{kept}/{len(before_prompts)} prompts kept",
    )


def _metric_pair_safety(after: list[dict]) -> MetricResult:
    """0 orphan tool messages + 0 dangling tool_calls at boundaries."""
    orphans = sum(1 for i, m in enumerate(after)
                  if m.get("role") == "tool" and i == 0)
    # dangling: assistant.tool_calls not followed by matching tool result
    dangling = 0
    for i, m in enumerate(after):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tids = {tc["id"] for tc in m["tool_calls"]}
            responded = set()
            j = i + 1
            while j < len(after) and after[j].get("role") == "tool":
                responded.add(after[j].get("tool_call_id"))
                j += 1
            if not tids.issubset(responded):
                dangling += 1
    count = orphans + dangling
    return MetricResult("Pair Safety", float(count == 0), 0.0, count == 0,
                        f"{orphans} orphan head + {dangling} dangling")


def _metric_compression_ratio(before: list[dict], after: list[dict]) -> MetricResult:
    """tokens_after / tokens_before -- lower is better, target < 0.5."""
    tb = compact.estimate_tokens_messages(before)
    ta = compact.estimate_tokens_messages(after)
    if tb == 0:
        return MetricResult("Compression Ratio", 1.0, 0.5, True, "empty input")
    ratio = ta / tb
    return MetricResult("Compression Ratio", ratio, 0.5, ratio < 0.5,
                        f"{ta}/{tb} tokens ({ratio:.1%})")


def _metric_context_block(after: list[dict]) -> MetricResult:
    """Snip placeholder must carry rebuilt context (not just '[snipped N turns]')."""
    markers = [m for m in after if (m.get("content") or "").startswith("[snipped")]
    if not markers:
        # may not have triggered snip; pass if conversation was short enough
        return MetricResult("Context Block Integrity", 1.0, 1.0, True,
                            "snip not triggered (short conversation)")
    has_context = any(len(m["content"]) > 60 for m in markers)
    return MetricResult("Context Block Integrity", float(has_context), 1.0,
                        has_context,
                        f"marker len={len(markers[0]['content']) if markers else 0}")


def _metric_current_task_focus(after: list[dict]) -> MetricResult:
    """The last real user prompt must survive and be reachable as the active task.

    Uses the snip placeholder boundary instead of a fixed percentage position:
    the tail window starts right after the ``[snipped ...]`` marker message.
    A prompt is "in focus" iff it appears *after* that marker (inside the tail
    window) **or** is the sole pinned prompt before the marker (single-task
    case where the task identity is pinned and there is nothing else to focus
    on).  When no snip marker is present (conversation was short enough), any
    surviving prompt passes.
    """
    prompts = _real_user_prompts(after)
    if not prompts:
        return MetricResult("Current Task Focus", 0.0, 1.0, False, "no prompts found")
    last_prompt = prompts[-1]
    idx = next((i for i, m in enumerate(after) if m.get("content") == last_prompt), -1)
    if idx == -1:
        return MetricResult("Current Task Focus", 0.0, 1.0, False, "last prompt missing")
    # Find the snip marker; tail window starts right after it.
    marker_idx = next((i for i, m in enumerate(after)
                       if (m.get("content") or "").startswith("[snipped")), -1)
    if marker_idx == -1:
        # No snip occurred -- conversation fit entirely, prompt is present.
        return MetricResult("Current Task Focus", 1.0, 1.0, True,
                            f"no snip, prompt at {idx}/{len(after)}")
    # Prompt is in the tail window (after marker) -> focused.
    if idx > marker_idx:
        return MetricResult("Current Task Focus", 1.0, 1.0, True,
                            f"prompt at {idx}/{len(after)}, marker at {marker_idx}")
    # Prompt is before marker (pinned).  Acceptable only when it is the *sole*
    # prompt -- i.e. a single-task session where the one task prompt is pinned
    # as task identity and the entire tail is that same task's execution.
    if len(prompts) == 1:
        return MetricResult("Current Task Focus", 1.0, 1.0, True,
                            f"sole pinned prompt at {idx}/{len(after)}, "
                            f"marker at {marker_idx}")
    return MetricResult("Current Task Focus", 0.0, 1.0, False,
                        f"prompt at {idx}/{len(after)} (pinned, "
                        f"marker at {marker_idx}, {len(prompts)} prompts)")


def _metric_task_transition(before: list[dict], after: list[dict]) -> MetricResult:
    """For multi-task scenarios: old task identity kept + new task prompt present."""
    prompts_before = _real_user_prompts(before)
    if len(prompts_before) < 2:
        return MetricResult("Task Transition", 1.0, 1.0, True, "single task")
    prompts_after = set(_real_user_prompts(after))
    old_kept = prompts_before[0] in prompts_after
    new_kept = prompts_before[-1] in prompts_after
    ok = old_kept and new_kept
    return MetricResult("Task Transition", float(ok), 1.0, ok,
                        f"old={old_kept}, new={new_kept}")


def _metric_structural_validity(after: list[dict]) -> MetricResult:
    """Output must not start with a tool message and must be non-empty."""
    if not after:
        return MetricResult("Structural Validity", 0.0, 1.0, False, "empty output")
    if after[0].get("role") == "tool":
        return MetricResult("Structural Validity", 0.0, 1.0, False,
                            "starts with orphan tool")
    return MetricResult("Structural Validity", 1.0, 1.0, True, "valid")


METRICS: list[Callable] = [
    _metric_task_identity,
    _metric_pair_safety,
    _metric_compression_ratio,
    _metric_context_block,
    _metric_current_task_focus,
    _metric_task_transition,
    _metric_structural_validity,
]


# --------------------------------------------------------------------------- #
# Pipeline runner
# --------------------------------------------------------------------------- #

def run_pipeline(messages: list[dict]) -> list[dict]:
    """Run the full L1-L3 pipeline (no LLM): tool_result_budget -> snip -> micro."""
    import copy
    msgs = copy.deepcopy(messages)
    msgs[:] = compact.tool_result_budget(msgs)
    msgs[:] = compact.snip_compact(msgs)
    msgs[:] = compact.micro_compact(msgs)
    return msgs


# --------------------------------------------------------------------------- #
# Scenario builders
# --------------------------------------------------------------------------- #

@dataclass
class Scenario:
    name: str
    description: str
    messages: list[dict] = field(default_factory=list)


def _scenario_single_task_long_with_confirmations() -> Scenario:
    """S1: one task, multiple user confirmations, lots of tool calls.

    Simulates: user sets goal -> agent proposes plan -> user confirms ->
    agent executes 50+ tool calls -> user reviews and requests changes ->
    agent continues.
    """
    msgs = []
    msgs.append(_user("Implement user authentication system with JWT"))
    # plan proposal + confirmation round
    msgs.append(_assistant("I propose: 1) models 2) routes 3) tests"))
    msgs.append(_user("Looks good, proceed with the plan"))
    for i in range(25):
        msgs.extend(_tool_turn("auth", f"a{i}", "edit_file",
                               _realistic_result("auth", i)))
    # user review + change request
    msgs.append(_user("The token expiry should be 24h not 1h, also add refresh tokens"))
    for i in range(25, 50):
        msgs.extend(_tool_turn("auth-fix", f"b{i}", "edit_file",
                               _realistic_result("authfix", i)))
    msgs.append(_assistant("Authentication system complete with 24h tokens and refresh"))
    return Scenario("S1_single_task_long_confirmations", "", msgs)


def _scenario_related_task_transition() -> Scenario:
    """S2: task A completes, task B references A's output.

    Simulates: build feature A -> A done -> user asks to extend with B that
    builds on A.
    """
    msgs = []
    msgs.append(_user("Build a user profile page showing name and email"))
    for i in range(30):
        msgs.extend(_tool_turn("profile", f"a{i}", "edit_file",
                               _realistic_result("profile", i)))
    msgs.append(_assistant("Profile page complete"))
    # new task referencing the old one
    msgs.append(_user("Now add an avatar upload to the profile page we just built"))
    for i in range(30, 55):
        msgs.extend(_tool_turn("avatar", f"b{i}", "edit_file",
                               _realistic_result("avatar", i)))
    msgs.append(_assistant("Avatar upload added to profile page"))
    return Scenario("S2_related_task_transition", "", msgs)


def _scenario_unrelated_task_transition() -> Scenario:
    """S3: task A completes, task B is completely unrelated.

    Simulates: finish auth work -> user pivots to a totally different feature.
    """
    msgs = []
    msgs.append(_user("Set up CI/CD pipeline with GitHub Actions"))
    for i in range(35):
        msgs.extend(_tool_turn("cicd", f"a{i}", "write_file",
                               _realistic_result("cicd", i, "yml")))
    msgs.append(_assistant("CI/CD pipeline configured"))
    # completely unrelated pivot
    msgs.append(_user("Now write unit tests for the math utility module"))
    for i in range(35, 60):
        msgs.extend(_tool_turn("math-test", f"b{i}", "edit_file",
                               _realistic_result("mathtest", i)))
    msgs.append(_assistant("Math utility tests written"))
    return Scenario("S3_unrelated_task_transition", "", msgs)


def _scenario_multi_task_accumulation() -> Scenario:
    """S4: three sequential tasks A -> B -> C.

    Tests that all three task prompts survive compaction.
    """
    msgs = []
    # Task A
    msgs.append(_user("Create the database schema for users table"))
    for i in range(20):
        msgs.extend(_tool_turn("db", f"a{i}", "write_file",
                               _realistic_result("schema", i, "sql")))
    msgs.append(_assistant("Schema created"))
    # Task B
    msgs.append(_user("Build the REST API endpoints for CRUD operations"))
    for i in range(20, 40):
        msgs.extend(_tool_turn("api", f"b{i}", "edit_file",
                               _realistic_result("api", i)))
    msgs.append(_assistant("API endpoints built"))
    # Task C
    msgs.append(_user("Add input validation and error handling to the API"))
    for i in range(40, 65):
        msgs.extend(_tool_turn("validation", f"c{i}", "edit_file",
                               _realistic_result("validation", i)))
    msgs.append(_assistant("Validation and error handling added"))
    return Scenario("S4_multi_task_accumulation", "", msgs)


def _scenario_extreme_length_single_task() -> Scenario:
    """S5: a single task with 100 rounds of tool calls.

    Tests compression ratio under extreme length without losing the task prompt.
    """
    msgs = []
    msgs.append(_user("Refactor the entire codebase to use async/await"))
    for i in range(100):
        msgs.extend(_tool_turn("refactor", f"r{i}", "edit_file",
                               _realistic_result("refactor", i)))
    msgs.append(_assistant("Codebase fully refactored to async/await"))
    return Scenario("S5_extreme_length_single_task", "", msgs)


def _scenario_mid_task_constraint_addition() -> Scenario:
    """S6: user adds an important constraint mid-task.

    Simulates: task in progress -> user adds a critical constraint that must
    be remembered.
    """
    msgs = []
    msgs.append(_user("Build a search feature for the application"))
    for i in range(25):
        msgs.extend(_tool_turn("search", f"a{i}", "edit_file",
                               _realistic_result("search", i)))
    # critical mid-task constraint
    msgs.append(_user("IMPORTANT: search must be case-insensitive and support Chinese"))
    for i in range(25, 50):
        msgs.extend(_tool_turn("search-fix", f"b{i}", "edit_file",
                               _realistic_result("searchci", i)))
    msgs.append(_assistant("Search feature with case-insensitive Chinese support done"))
    return Scenario("S6_mid_task_constraint_addition", "", msgs)


SCENARIO_BUILDERS: list[Callable[[], Scenario]] = [
    _scenario_single_task_long_with_confirmations,
    _scenario_related_task_transition,
    _scenario_unrelated_task_transition,
    _scenario_multi_task_accumulation,
    _scenario_extreme_length_single_task,
    _scenario_mid_task_constraint_addition,
]


# --------------------------------------------------------------------------- #
# Evaluation runner
# --------------------------------------------------------------------------- #

@dataclass
class ScenarioReport:
    scenario: str
    description: str
    metrics: list[MetricResult]
    before_msgs: int
    after_msgs: int
    before_tokens: int
    after_tokens: int


def evaluate_scenario(scenario: Scenario) -> ScenarioReport:
    before = scenario.messages
    after = run_pipeline(before)
    metrics = [m(before, after) if m.__code__.co_argcount == 2 else m(after)
               for m in METRICS]
    return ScenarioReport(
        scenario=scenario.name,
        description=scenario.description,
        metrics=metrics,
        before_msgs=len(before),
        after_msgs=len(after),
        before_tokens=compact.estimate_tokens_messages(before),
        after_tokens=compact.estimate_tokens_messages(after),
    )


def print_report(reports: list[ScenarioReport]) -> None:
    print("=" * 90)
    print("COMPACTION LAYER EVALUATION REPORT")
    print("=" * 90)
    print()
    all_pass = True
    for r in reports:
        print(f"\n{'─' * 90}")
        print(f"Scenario: {r.scenario}")
        print(f"  msgs: {r.before_msgs} -> {r.after_msgs}  |  "
              f"tokens: {r.before_tokens} -> {r.after_tokens}")
        print(f"{'─' * 90}")
        for m in r.metrics:
            status = "✅" if m.passed else "❌"
            print(f"  {status} {m.name:<35} {m.value:>8.2%}  (target: {m.target:.0%})  {m.detail}")
            if not m.passed:
                all_pass = False
    print(f"\n{'=' * 90}")
    overall = "ALL PASSED ✅" if all_pass else "SOME FAILED ❌"
    print(f"OVERALL: {overall}")
    print(f"{'=' * 90}")

    # Summary table
    print(f"\n{'Scenario':<45} {'Identity':>9} {'Pairs':>7} {'Ratio':>7} "
          f"{'CtxBlk':>7} {'Focus':>7} {'Trans':>7} {'Valid':>7}")
    print("─" * 100)
    for r in reports:
        vals = {m.name: m for m in r.metrics}
        def _v(key):
            m = vals.get(key)
            return f"{m.value:.0%}" if m else "N/A"
        print(f"{r.scenario:<45} "
              f"{_v('Task Identity Preservation'):>9} "
              f"{_v('Pair Safety'):>7} "
              f"{_v('Compression Ratio'):>7} "
              f"{_v('Context Block Integrity'):>7} "
              f"{_v('Current Task Focus'):>7} "
              f"{_v('Task Transition'):>7} "
              f"{_v('Structural Validity'):>7}")
    print()


def main():
    reports = []
    for builder in SCENARIO_BUILDERS:
        scenario = builder()
        report = evaluate_scenario(scenario)
        reports.append(report)
    print_report(reports)

    # write JSON summary
    import pathlib
    out_dir = pathlib.Path(__file__).resolve().parent
    summary = []
    for r in reports:
        summary.append({
            "scenario": r.scenario,
            "before_msgs": r.before_msgs,
            "after_msgs": r.after_msgs,
            "before_tokens": r.before_tokens,
            "after_tokens": r.after_tokens,
            "metrics": [{"name": m.name, "value": m.value, "target": m.target,
                         "passed": m.passed, "detail": m.detail}
                        for m in r.metrics],
        })
    (out_dir / "compact_eval_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"JSON summary written to {out_dir / 'compact_eval_result.json'}")


if __name__ == "__main__":
    main()
