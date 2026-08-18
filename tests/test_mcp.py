"""Tests for the MCP (Model Context Protocol) client module.

These tests verify the synchronous facade logic (config parsing, tool schema
generation, name routing, handler generation) without requiring a live MCP
server.  The async transport layer (``MCPClient.connect``/``call_tool``) is
exercised via mocks where needed.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mcodecore.mcp import (
    MCPManager, MCPClient, MCPServerConfig,
    MCP_PREFIX, init_mcp, shutdown_mcp,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_tool(name, description="desc", input_schema=None):
    """Build a lightweight object mimicking mcp.types.Tool."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema or {"type": "object", "properties": {}},
    )


def _make_fake_client(server_name, tools):
    """Build a fake MCPClient (not connected) with pre-populated tool cache."""
    client = MagicMock(spec=MCPClient)
    client.config = MCPServerConfig(name=server_name, url="http://x")
    client.list_tools.return_value = tools
    client.call_tool.return_value = "ok"
    return client


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #

class TestConfigParsing:
    def test_missing_config_file_returns_empty(self, tmp_path):
        mgr = MCPManager()
        with patch("mcodecore.mcp.MCP_CONFIG_PATH", tmp_path / "noexist.json"):
            assert mgr._load_config() == []

    def test_empty_mcp_servers_returns_empty(self, tmp_path):
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
        mgr = MCPManager()
        with patch("mcodecore.mcp.MCP_CONFIG_PATH", cfg):
            assert mgr._load_config() == []

    def test_parses_server_with_url(self, tmp_path):
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "weather": {"url": "http://localhost:8000/mcp"}
            }
        }), encoding="utf-8")
        mgr = MCPManager()
        with patch("mcodecore.mcp.MCP_CONFIG_PATH", cfg):
            configs = mgr._load_config()
        assert len(configs) == 1
        assert configs[0].name == "weather"
        assert configs[0].url == "http://localhost:8000/mcp"
        assert configs[0].enabled is True

    def test_parses_headers_and_disabled(self, tmp_path):
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "api": {
                    "url": "http://x",
                    "headers": {"Authorization": "Bearer tok"},
                    "enabled": False,
                }
            }
        }), encoding="utf-8")
        mgr = MCPManager()
        with patch("mcodecore.mcp.MCP_CONFIG_PATH", cfg):
            configs = mgr._load_config()
        assert configs[0].headers == {"Authorization": "Bearer tok"}
        assert configs[0].enabled is False

    def test_server_without_url_is_skipped(self, tmp_path):
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {"bad": {"headers": {}}}
        }), encoding="utf-8")
        mgr = MCPManager()
        with patch("mcodecore.mcp.MCP_CONFIG_PATH", cfg):
            assert mgr._load_config() == []

    def test_invalid_json_returns_empty(self, tmp_path, capsys):
        cfg = tmp_path / ".mcp.json"
        cfg.write_text("{not valid json", encoding="utf-8")
        mgr = MCPManager()
        with patch("mcodecore.mcp.MCP_CONFIG_PATH", cfg):
            assert mgr._load_config() == []


# --------------------------------------------------------------------------- #
# Tool schema generation
# --------------------------------------------------------------------------- #

class TestSchemaGeneration:
    def test_prefixed_tool_name(self):
        mgr = MCPManager()
        mgr._clients["srv"] = _make_fake_client("srv", [
            _make_tool("get_weather", "Get weather"),
        ])
        schemas = mgr.list_all_tool_schemas()
        assert len(schemas) == 1
        name = schemas[0]["function"]["name"]
        assert name == f"{MCP_PREFIX}srv__get_weather"
        assert "[MCP:srv]" in schemas[0]["function"]["description"]

    def test_input_schema_propagated(self):
        mgr = MCPManager()
        mgr._clients["srv"] = _make_fake_client("srv", [
            _make_tool("calc", "Calc", {
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            }),
        ])
        schemas = mgr.list_all_tool_schemas()
        params = schemas[0]["function"]["parameters"]
        assert params["properties"]["x"]["type"] == "number"
        assert params["required"] == ["x"]

    def test_default_input_schema_when_missing(self):
        mgr = MCPManager()
        fake = _make_fake_client("srv", [
            _make_tool("ping", "ping", None),
        ])
        mgr._clients["srv"] = fake
        schemas = mgr.list_all_tool_schemas()
        params = schemas[0]["function"]["parameters"]
        assert params == {"type": "object", "properties": {}}

    def test_multiple_servers_multiple_tools(self):
        mgr = MCPManager()
        mgr._clients["a"] = _make_fake_client("a", [_make_tool("t1")])
        mgr._clients["b"] = _make_fake_client("b", [_make_tool("t2"), _make_tool("t3")])
        schemas = mgr.list_all_tool_schemas()
        names = sorted(s["function"]["name"] for s in schemas)
        assert names == ["mcp__a__t1", "mcp__b__t2", "mcp__b__t3"]

    def test_list_tools_failure_skips_server(self, capsys):
        mgr = MCPManager()
        fake = MagicMock(spec=MCPClient)
        fake.list_tools.side_effect = RuntimeError("boom")
        mgr._clients["srv"] = fake
        assert mgr.list_all_tool_schemas() == []


# --------------------------------------------------------------------------- #
# Call routing
# --------------------------------------------------------------------------- #

class TestCallRouting:
    def test_call_routes_to_correct_server(self):
        mgr = MCPManager()
        fake_a = _make_fake_client("a", [_make_tool("t1")])
        fake_b = _make_fake_client("b", [_make_tool("t2")])
        mgr._clients["a"] = fake_a
        mgr._clients["b"] = fake_b
        mgr.call("mcp__a__t1", {"q": "hi"})
        fake_a.call_tool.assert_called_once_with("t1", {"q": "hi"})
        fake_b.call_tool.assert_not_called()

    def test_call_invalid_name_returns_error(self):
        mgr = MCPManager()
        result = mgr.call("not_a_mcp_tool", {})
        assert "Error" in result

    def test_call_unknown_server_returns_error(self):
        mgr = MCPManager()
        result = mgr.call("mcp__ghost__t1", {})
        assert "not connected" in result


# --------------------------------------------------------------------------- #
# Handler generation
# --------------------------------------------------------------------------- #

class TestHandlerGeneration:
    def test_build_handlers_returns_callable(self):
        mgr = MCPManager()
        mgr._clients["srv"] = _make_fake_client("srv", [_make_tool("echo")])
        handlers = mgr.build_handlers()
        assert "mcp__srv__echo" in handlers
        result = handlers["mcp__srv__echo"](msg="hello")
        assert result == "ok"

    def test_handler_routes_through_call(self):
        mgr = MCPManager()
        mgr._clients["srv"] = _make_fake_client("srv", [_make_tool("echo")])
        handlers = mgr.build_handlers()
        handlers["mcp__srv__echo"](msg="hello")
        mgr._clients["srv"].call_tool.assert_called_once_with(
            "echo", {"msg": "hello"}
        )


# --------------------------------------------------------------------------- #
# Status / lifecycle
# --------------------------------------------------------------------------- #

class TestStatus:
    def test_is_connected_false_when_no_clients(self):
        mgr = MCPManager()
        assert mgr.is_connected is False

    def test_is_connected_true_with_clients(self):
        mgr = MCPManager()
        mgr._clients["srv"] = _make_fake_client("srv", [])
        mgr._connected = True
        assert mgr.is_connected is True

    def test_status_no_servers(self):
        mgr = MCPManager()
        assert "no servers" in mgr.status()

    def test_status_lists_servers(self):
        mgr = MCPManager()
        mgr._clients["srv"] = _make_fake_client("srv", [_make_tool("t")])
        status = mgr.status()
        assert "srv" in status
        assert "1 tools" in status

    def test_shutdown_clears_clients(self):
        mgr = MCPManager()
        fake = MagicMock(spec=MCPClient)
        mgr._clients["srv"] = fake
        mgr._connected = True
        mgr.shutdown()
        assert mgr._clients == {}
        assert mgr.is_connected is False
        fake.close.assert_called_once()


# --------------------------------------------------------------------------- #
# init fault tolerance
# --------------------------------------------------------------------------- #

class TestInitFaultTolerance:
    def test_init_with_missing_config_is_noop(self, tmp_path):
        mgr = MCPManager()
        with patch("mcodecore.mcp.MCP_CONFIG_PATH", tmp_path / "nope.json"):
            mgr.init()
        assert mgr.is_connected is False

    def test_init_skips_connection_failures(self, tmp_path):
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {"bad": {"url": "http://localhost:1/mcp"}}
        }), encoding="utf-8")
        mgr = MCPManager()
        with patch("mcodecore.mcp.MCP_CONFIG_PATH", cfg):
            mgr.init()  # should not raise
        assert mgr.is_connected is False

    def test_init_skips_disabled_servers(self, tmp_path):
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {"off": {"url": "http://x", "enabled": False}}
        }), encoding="utf-8")
        mgr = MCPManager()
        with patch("mcodecore.mcp.MCP_CONFIG_PATH", cfg):
            with patch.object(MCPClient, "connect", side_effect=AssertionError(
                    "should not connect disabled server")):
                mgr.init()
        assert mgr.is_connected is False


# --------------------------------------------------------------------------- #
# Module-level convenience functions
# --------------------------------------------------------------------------- #

class TestModuleFunctions:
    def test_init_mcp_swallows_exceptions(self):
        """init_mcp must never propagate exceptions."""
        with patch.object(MCPManager, "init", side_effect=RuntimeError("boom")):
            init_mcp()  # should not raise

    def test_shutdown_mcp_swallows_exceptions(self):
        with patch.object(MCPManager, "shutdown",
                          side_effect=RuntimeError("boom")):
            shutdown_mcp()  # should not raise


# --------------------------------------------------------------------------- #
# Context integration
# --------------------------------------------------------------------------- #

class TestContextIntegration:
    def test_ctx_mcp_returns_singleton(self):
        from mcodecore.context import ctx
        a = ctx.mcp
        b = ctx.mcp
        assert a is b
        assert isinstance(a, MCPManager)


# --------------------------------------------------------------------------- #
# _inject_mcp_tools (tools.py integration)
# --------------------------------------------------------------------------- #

class TestInjectMcpTools:
    def test_inject_noop_when_not_connected(self):
        from mcodecore.tools import _inject_mcp_tools, TOOLS
        before = len(TOOLS)
        _inject_mcp_tools()
        assert len(TOOLS) == before

    def test_inject_adds_tools_when_connected(self):
        import mcodecore.tools as tools_mod
        from mcodecore.tools import TOOLS, SUB_TOOLS, TOOL_HANDLERS, SUB_HANDLERS

        # Capture original state to restore after test
        orig_tools = list(TOOLS)
        orig_sub_tools = list(SUB_TOOLS)
        orig_handlers = dict(TOOL_HANDLERS)
        orig_sub_handlers = dict(SUB_HANDLERS)

        try:
            fake_mgr = MagicMock(spec=MCPManager)
            fake_mgr.is_connected = True
            fake_mgr.list_all_tool_schemas.return_value = [{
                "type": "function",
                "function": {
                    "name": "mcp__srv__echo",
                    "description": "[MCP:srv] echo",
                    "parameters": {"type": "object", "properties": {}},
                },
            }]
            fake_mgr.build_handlers.return_value = {
                "mcp__srv__echo": lambda **kw: "echoed"
            }
            # Patch the private attribute backing the lazy property
            orig_mcp = tools_mod.ctx._mcp_manager
            tools_mod.ctx._mcp_manager = fake_mgr
            try:
                tools_mod._inject_mcp_tools()
            finally:
                tools_mod.ctx._mcp_manager = orig_mcp

            assert any(
                t["function"]["name"] == "mcp__srv__echo" for t in TOOLS
            )
            assert "mcp__srv__echo" in TOOL_HANDLERS
            assert "mcp__srv__echo" in SUB_HANDLERS
            assert TOOL_HANDLERS["mcp__srv__echo"]() == "echoed"
        finally:
            TOOLS[:] = orig_tools
            SUB_TOOLS[:] = orig_sub_tools
            TOOL_HANDLERS.clear()
            TOOL_HANDLERS.update(orig_handlers)
            SUB_HANDLERS.clear()
            SUB_HANDLERS.update(orig_sub_handlers)
