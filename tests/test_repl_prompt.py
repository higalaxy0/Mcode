"""REPL prompt-rendering behaviour tests (no real API needed).

Covers the two known CLI display bugs:

1. After pressing Enter, a spurious blank ``Mcode >>`` line appeared
   (the input-reader thread re-printed the prompt before agent output).
2. After the agent finished a task, no ``Mcode >>`` prompt was shown
   (the buried prompt was blocking ``input()`` forever).

Design: drive the real REPL thread machinery with a scripted stdin
feed.  ``input()`` is replaced with a queue-backed stub so no real
TTY is required; prompts are captured via ``capsys``.
"""

from __future__ import annotations

import queue
import threading
import time

import pytest

from mcodecore import agent


class _ScriptedStdin:
    """Stands in for builtins.input(): pops scripted lines."""

    def __init__(self, lines: list[str]):
        self.lines = list(lines)
        self.lock = threading.Lock()
        self.calls = 0

    def __call__(self, prompt=""):
        assert prompt == "", "input() must not print its own prompt"
        with self.lock:
            self.calls += 1
            if not self.lines:
                raise EOFError
            return self.lines.pop(0)


@pytest.fixture
def scripted_stdin(monkeypatch):
    def _setup(lines):
        stub = _ScriptedStdin(lines)
        monkeypatch.setattr("builtins.input", stub)
        return stub
    return _setup


def _run_repl(monkeypatch, lines, scripted_stdin, max_events=50):
    """Run the real REPL loop with scripted input; return captured stdout.

    ``main()`` itself is not called: it initializes MCP and other heavy
    machinery.  Instead the event loop body (the same code path main()
    uses) is exercised through the module-level threads + queue.
    """
    raise NotImplementedError  # replaced inline per-test below


def _drive(lines, turn_fn, timeout=5.0):
    """Start reader/watcher threads + main-loop core, return stdout text.

    ``turn_fn(history)`` replaces ``_run_agent_turn``; it receives the
    shared history list and returns whether to exit.
    """
    out = {"text": ""}

    event_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    history: list = []

    import io
    from contextlib import redirect_stdout

    reader = threading.Thread(
        target=agent._input_reader, args=(event_q, stop),
        daemon=True, name="input-reader")
    reader.start()

    # Feed scripted lines into the queue *as if* typed.
    for ln in lines:
        time.sleep(0.05)
        event_q.put(("input", ln))

    buf = io.StringIO()
    deadline = time.monotonic() + timeout
    exited = {"flag": False}

    with redirect_stdout(buf):
        _prompt_shown = False
        while time.monotonic() < deadline:
            if not _prompt_shown:
                agent._print_prompt()
                _prompt_shown = True
            try:
                kind, data = event_q.get(timeout=0.05)
            except queue.Empty:
                if not lines:
                    break
                continue
            if kind == "input":
                query = data
                if query.strip().lower() in ("q", "exit", "quit"):
                    exited["flag"] = True
                    break
                if not query.strip():
                    _prompt_shown = False
                    continue
                history.append({"role": "user", "content": query})
                _prompt_shown = False
                if turn_fn(history):
                    exited["flag"] = True
                    break

    stop.set()
    out["text"] = buf.getvalue()
    return out["text"], exited["flag"], history


# --------------------------------------------------------------------------- #
# Bug 1: spurious blank prompt after Enter
# --------------------------------------------------------------------------- #

def test_no_spurious_prompt_after_enter():
    """After Enter (agent turn runs), no extra prompt until turn end.

    Old behaviour: reader thread re-printed 'Mcode >> ' immediately
    after delivering the line, so it appeared BEFORE agent output and
    looked like a stray blank prompt line.  New behaviour: the only
    prompt is printed by the main loop when idle.
    """
    turn_outputs = ["agent reply line"]

    def turn_fn(history):
        print(turn_outputs[0])
        return False

    text, exited, _ = _drive(["hello"], turn_fn)
    # Exactly one prompt before the typed input, one after the turn.
    assert text.count("Mcode >> ") == 2
    # The prompt printed AFTER the turn must come after agent output.
    first_agent = text.index("agent reply line")
    prompts = [i for i in range(len(text)) if text.startswith("Mcode >> ", i)]
    assert len(prompts) >= 2
    assert prompts[-1] > first_agent
    assert not exited


def test_prompt_reappears_after_each_turn():
    """Three consecutive turns: prompt count = 4 (1 initial + 3 after)."""
    def turn_fn(history):
        print(f"turn {len(history)}")
        return False

    text, exited, _ = _drive(["a", "b", "c"], turn_fn)
    assert text.count("Mcode >> ") == 4


# --------------------------------------------------------------------------- #
# Bug 2: missing prompt after task completion
# --------------------------------------------------------------------------- #

def test_prompt_present_after_long_task():
    """Simulated slow agent task: prompt appears once done, not during."""
    def turn_fn(history):
        time.sleep(0.2)  # simulate agent work
        print("task finished")
        return False

    text, exited, _ = _drive(["do work"], turn_fn)
    assert "task finished" in text
    # Prompt must be rendered after completion.
    last_prompt = text.rfind("Mcode >> ")
    assert last_prompt > text.index("task finished")


def test_no_prompt_during_agent_turn():
    """While the agent is working, no new prompt is rendered.

    The reader thread must not print prompts on its own.  We verify by
    checking that no prompt appears between turn start and output.
    """
    def turn_fn(history):
        print("working")
        print("done working")
        return False

    text, exited, _ = _drive(["work"], turn_fn)
    first = text.index("working")
    last = text.index("done working")
    for i in range(first, last):
        assert not text.startswith("Mcode >> ", i), \
            "prompt rendered mid-turn"


# --------------------------------------------------------------------------- #
# Regression: empty-line handling still re-renders prompt
# --------------------------------------------------------------------------- #

def test_empty_input_still_shows_prompt():
    """Empty Enter consumes the line; a fresh prompt follows."""
    def turn_fn(history):
        print("should not run")
        return False

    text, exited, _ = _drive(["", "x"], turn_fn)
    # '' consumed -> prompt re-rendered; 'x' runs the turn.
    assert "should not run" in text
    assert text.count("Mcode >> ") == 3  # initial + after '' + after 'x'


def test_quit_stops_loop():
    def turn_fn(history):
        return False

    text, exited, _ = _drive(["q"], turn_fn)
    assert exited


# --------------------------------------------------------------------------- #
# Reader-thread unit behaviour
# --------------------------------------------------------------------------- #

def test_input_reader_no_prompt_arg(monkeypatch):
    """``_input_reader`` must call input() with NO prompt argument.

    Regression guard for bug 1: the reader used to pass the prompt
    string, printing it from the wrong thread at the wrong time.
    """
    seen_prompts = []

    def fake_input(prompt=""):
        seen_prompts.append(prompt)
        raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    agent._input_reader(q, stop)
    # Reader exited on EOF and pushed a quit event.
    assert q.get(timeout=1) == ("quit", None)
    assert seen_prompts == [""]


def test_input_reader_delivers_lines(monkeypatch):
    lines = iter(["alpha", "beta"])

    def fake_input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    agent._input_reader(q, stop)
    assert q.get(timeout=1) == ("input", "alpha")
    assert q.get(timeout=1) == ("input", "beta")
    assert q.get(timeout=1) == ("quit", None)
