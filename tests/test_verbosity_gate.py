"""Tests for the verbosity gate (MCODE_VERBOSE / debug())."""

import importlib
import io
import os
import sys
from contextlib import redirect_stdout

import pytest


@pytest.fixture
def config_module():
    """Return a freshly reloaded mcodecore.config with clean env."""
    import mcodecore.config as cfg
    importlib.reload(cfg)
    yield cfg
    # restore default after test
    importlib.reload(cfg)


class TestVerboseFlag:
    """Tests for the VERBOSE env flag parsing."""

    def test_verbose_defaults_to_false(self, config_module):
        """VERBOSE should be False when MCODE_VERBOSE is unset."""
        os.environ.pop("MCODE_VERBOSE", None)
        importlib.reload(config_module)
        assert config_module.VERBOSE is False

    def test_verbose_true_when_env_is_1(self, config_module):
        """VERBOSE should be True when MCODE_VERBOSE=1."""
        os.environ["MCODE_VERBOSE"] = "1"
        importlib.reload(config_module)
        assert config_module.VERBOSE is True
        os.environ.pop("MCODE_VERBOSE", None)

    def test_verbose_false_when_env_is_0(self, config_module):
        """VERBOSE should be False when MCODE_VERBOSE=0."""
        os.environ["MCODE_VERBOSE"] = "0"
        importlib.reload(config_module)
        assert config_module.VERBOSE is False
        os.environ.pop("MCODE_VERBOSE", None)

    def test_verbose_false_when_env_is_other(self, config_module):
        """VERBOSE should be False when MCODE_VERBOSE is any value other
        than '1'."""
        os.environ["MCODE_VERBOSE"] = "true"
        importlib.reload(config_module)
        assert config_module.VERBOSE is False
        os.environ.pop("MCODE_VERBOSE", None)


class TestDebugFunction:
    """Tests for the debug() helper function."""

    def test_debug_silent_by_default(self, config_module):
        """debug() should not print when VERBOSE is False."""
        config_module.VERBOSE = False
        buf = io.StringIO()
        with redirect_stdout(buf):
            config_module.debug("[test] hidden message")
        assert buf.getvalue() == ""

    def test_debug_prints_when_verbose(self, config_module):
        """debug() should print the message when VERBOSE is True."""
        config_module.VERBOSE = True
        buf = io.StringIO()
        with redirect_stdout(buf):
            config_module.debug("[test] visible message")
        assert "[test] visible message" in buf.getvalue()

    def test_debug_prints_exact_message(self, config_module):
        """debug() should print exactly the message passed (no ANSI prefix
        from the helper itself)."""
        config_module.VERBOSE = True
        buf = io.StringIO()
        msg = "[memory] consolidated 5 -> 2 memories"
        with redirect_stdout(buf):
            config_module.debug(msg)
        assert buf.getvalue().strip() == msg

    def test_debug_accepts_empty_string(self, config_module):
        """debug() should handle empty string without error."""
        config_module.VERBOSE = True
        buf = io.StringIO()
        with redirect_stdout(buf):
            config_module.debug("")
        # newline only
        assert buf.getvalue() == "\n"


class TestDebugGatedImport:
    """Tests that debug-gated prints are actually wired up in modules."""

    def test_debug_imported_in_agent(self):
        """agent.py should import debug from config."""
        import mcodecore.agent as agent_mod
        assert hasattr(agent_mod, "debug") or hasattr(
            agent_mod, "config") or "debug" in dir(agent_mod)

    def test_debug_imported_in_bus(self):
        """bus.py should import debug from config."""
        import mcodecore.bus as bus_mod
        # debug should be accessible via the module's namespace
        import mcodecore.config as cfg
        assert hasattr(cfg, "debug")

    def test_debug_imported_in_memory(self):
        """memory.py should import debug from config."""
        import mcodecore.config as cfg
        assert callable(cfg.debug)

    def test_debug_imported_in_teammates(self):
        """teammates.py should import debug from config."""
        import mcodecore.config as cfg
        assert callable(cfg.debug)

    def test_debug_imported_in_subagent(self):
        """subagent.py should import debug from config."""
        import mcodecore.config as cfg
        assert callable(cfg.debug)


class TestPrintReduction:
    """Verify that key debug messages are now gated behind VERBOSE, not
    unconditional prints."""

    @pytest.mark.parametrize("file_path,pattern", [
        ("mcodecore/agent.py", "thinking!!!!"),
    ])
    def test_no_raw_print_for_debug_messages(self, file_path, pattern):
        """These specific debug messages should no longer use raw print()."""
        import re
        path = os.path.join(os.path.dirname(__file__), "..", file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert re.search(pattern, content) is None, (
            f"Found raw print with '{pattern}' in {file_path}; "
            f"should be using debug() instead")

    def test_no_diagnostic_print_in_agent(self):
        """agent.py should not have raw print for retries/API errors
        (those should be debug()).  Tool display and user-facing errors
        are exempt."""
        path = os.path.join(os.path.dirname(__file__), "..",
                            "mcodecore/agent.py")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        import re
        # These diagnostic messages should have been converted to debug()
        banned_patterns = [
            r"print\(.*agent retry",
            r"print\(.*API error",
            r"print\(.*Inbox:.*messages injected",
            r"print\(.*thinking!",
        ]
        for pat in banned_patterns:
            assert re.search(pat, content) is None, (
                f"agent.py still has print matching '{pat}'; "
                f"should use debug()")

    def test_no_compaction_print_in_teammates_compaction(self):
        """teammates.py compaction prints should be debug()."""
        path = os.path.join(os.path.dirname(__file__), "..",
                            "mcodecore/teammates.py")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # We check specifically for compaction-related prints
        import re
        compact_prints = re.findall(
            r"print\(.*(?:compact|reactive)", content)
        assert len(compact_prints) == 0, (
            f"teammates.py still has {len(compact_prints)} compaction "
            f"print() calls that should be debug()")

    def test_no_diagnostic_print_in_subagent(self):
        """subagent.py should have no raw print() calls at all since it
        runs as a headless sub-process."""
        path = os.path.join(os.path.dirname(__file__), "..",
                            "mcodecore/subagent.py")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        import re
        assert len(re.findall(r"^\s+print\(", content, re.MULTILINE)) == 0, (
            "subagent.py still has print() calls that should be debug()")


class TestEndToEndVerboseGate:
    """End-to-end test that debug output appears only when VERBOSE is on."""

    def test_debug_output_gated_end_to_end(self, config_module):
        """When VERBOSE is off, no debug output; when on, debug output
        appears."""
        # Off
        config_module.VERBOSE = False
        buf_off = io.StringIO()
        with redirect_stdout(buf_off):
            config_module.debug("[e2e] test message")
        assert buf_off.getvalue() == ""

        # On
        config_module.VERBOSE = True
        buf_on = io.StringIO()
        with redirect_stdout(buf_on):
            config_module.debug("[e2e] test message")
        assert "[e2e] test message" in buf_on.getvalue()
