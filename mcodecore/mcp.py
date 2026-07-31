"""MCP (Model Context Protocol) client - Streamable HTTP transport.

Provides a synchronous facade (``MCPManager``) over the async ``mcp`` SDK so
that the rest of the codebase (which is entirely synchronous) can discover and
call tools exposed by remote MCP servers without any async boilerplate.

Architecture::

    MCPManager (sync facade, main thread)
     └─ MCPClient × N  (one per configured server)
         ├─ dedicated asyncio event loop running on a daemon thread
         ├─ streamablehttp_client + ClientSession kept alive via AsyncExitStack
         └─ sync methods bridge to the loop via run_coroutine_threadsafe

Key design decisions:
- Each ``MCPClient`` owns a *dedicated* event loop on a daemon thread.  This
  keeps ``ClientSession`` alive across multiple ``call_tool`` invocations
  (sessions are bound to the loop that created them, so ``asyncio.run()``
  per-call would not work).
- Tool names are prefixed ``mcp__{server}__{tool}`` to avoid collisions with
  built-in tools and to enable unambiguous routing.
- The entire subsystem is fault-tolerant: missing config file, bad JSON,
  connection failures, or call errors never crash the agent.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from .config import MCP_CONFIG_PATH
from .context import ctx


# --------------------------------------------------------------------------- #
# Config dataclass
# --------------------------------------------------------------------------- #

@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server connection."""

    name: str                       # server identifier (namespace prefix)
    url: str                        # Streamable HTTP endpoint
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


# --------------------------------------------------------------------------- #
# MCPClient - single server, async core + sync facade
# --------------------------------------------------------------------------- #

class MCPClient:
    """Asynchronous MCP client for a single server, wrapped in a sync facade.

    A dedicated asyncio event loop runs on a daemon thread.  All async
    operations (connect / list_tools / call_tool) are dispatched to that loop
    via :func:`asyncio.run_coroutine_threadsafe`, keeping the caller's thread
    synchronous.
    """

    #: timeout for a single tool call (seconds), aligned with BASH_TIMEOUT
    CALL_TIMEOUT: int = 300

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._session = None          # mcp.client.session.ClientSession
        self._exit_stack: AsyncExitStack | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._tools_cache: list | None = None   # list[Tool] after first list_tools

    # -- lifecycle -------------------------------------------------------- #

    def connect(self) -> None:
        """Start the background loop thread and establish the MCP session.

        Raises on failure (caller decides whether to skip the server).
        """
        # 1. create a dedicated event loop (not bound to main thread)
        self._loop = asyncio.new_event_loop()

        ready = threading.Event()
        loop_error: list = []

        def _run_loop() -> None:
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_forever()
            finally:
                # drain pending tasks before closing
                try:
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                except Exception:
                    pass
                try:
                    self._loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(
            target=_run_loop, name=f"mcp-{self.config.name}", daemon=True
        )
        self._thread.start()

        # 2. perform the async connect on that loop
        future = asyncio.run_coroutine_threadsafe(
            self._async_connect(), self._loop
        )
        try:
            future.result(timeout=30)
        except Exception as exc:
            loop_error.append(exc)
            self._stop_loop()
            raise

    def close(self) -> None:
        """Gracefully close the session and stop the background loop."""
        if self._loop is None:
            return
        # close the async exit stack (sends DELETE to server)
        if self._exit_stack is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._exit_stack.aclose(), self._loop
                )
                future.result(timeout=10)
            except Exception:
                pass
            self._exit_stack = None
            self._session = None
        self._stop_loop()

    def _stop_loop(self) -> None:
        """Stop the event loop and join the thread."""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None

    # -- async core (runs on the dedicated loop) -------------------------- #

    async def _async_connect(self) -> None:
        from mcp.client.streamable_http import streamablehttp_client
        from mcp.client.session import ClientSession

        self._exit_stack = AsyncExitStack()
        try:
            read_stream, write_stream, _get_session_id = await (
                self._exit_stack.enter_async_context(
                    streamablehttp_client(
                        self.config.url,
                        headers=self.config.headers or None,
                    )
                )
            )
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
        except Exception:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            raise

    async def _async_list_tools(self) -> list:
        from mcp import types as mcp_types
        result = await self._session.list_tools()
        return list(result.tools)

    async def _async_call_tool(self, name: str, args: dict) -> str:
        from mcp import types as mcp_types
        result = await self._session.call_tool(name, args)
        parts: list[str] = []
        for c in result.content:
            if isinstance(c, mcp_types.TextContent):
                parts.append(c.text)
            elif isinstance(c, (mcp_types.ImageContent, mcp_types.AudioContent)):
                parts.append(f"[{c.type} content omitted]")
            elif isinstance(c, mcp_types.EmbeddedResource):
                parts.append(f"[embedded resource: {c.resource.type}]")
            else:
                parts.append(f"[unknown content type: {getattr(c, 'type', '?')}]")
        text = "\n".join(parts) if parts else "(no output)"
        if result.isError:
            return f"[MCP Error] {text}"
        return text

    # -- sync facade (callable from main thread) -------------------------- #

    def list_tools(self) -> list:
        """Return cached tool list, fetching on first call."""
        if self._tools_cache is not None:
            return self._tools_cache
        if self._session is None:
            return []
        future = asyncio.run_coroutine_threadsafe(
            self._async_list_tools(), self._loop
        )
        tools = future.result(timeout=30)
        self._tools_cache = tools
        return tools

    def call_tool(self, name: str, args: dict) -> str:
        """Call a tool on this server and return the text result."""
        if self._session is None:
            return f"Error: MCP server '{self.config.name}' not connected"
        future = asyncio.run_coroutine_threadsafe(
            self._async_call_tool(name, args), self._loop
        )
        try:
            return future.result(timeout=self.CALL_TIMEOUT)
        except Exception as exc:
            return f"Error: MCP call '{name}' failed: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# MCPManager - registry / sync API
# --------------------------------------------------------------------------- #

#: prefix used for all MCP tools to avoid namespace collisions
MCP_PREFIX = "mcp__"


class MCPManager:
    """Manages multiple ``MCPClient`` instances and provides a sync API.

    Usage::

        mgr = ctx.mcp
        mgr.init()                         # connect to all configured servers
        schemas = mgr.list_all_tool_schemas()
        handlers = mgr.build_handlers()
        # ... register into TOOLS / TOOL_HANDLERS ...
        mgr.shutdown()                     # clean exit
    """

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._connected: bool = False

    # -- lifecycle -------------------------------------------------------- #

    def init(self) -> None:
        """Load ``.mcp.json`` and connect to every enabled server.

        Fault-tolerant: missing file / bad JSON / connection failures are
        logged and skipped; the manager continues with whatever servers
        connected successfully.
        """
        configs = self._load_config()
        if not configs:
            return
        for cfg in configs:
            if not cfg.enabled:
                continue
            try:
                client = MCPClient(cfg)
                client.connect()
                self._clients[cfg.name] = client
            except Exception as exc:
                print(
                    f"\033[33m[MCP] Failed to connect '{cfg.name}': "
                    f"{type(exc).__name__}: {exc}\033[0m"
                )
        self._connected = bool(self._clients)

    def shutdown(self) -> None:
        """Close all client connections."""
        for client in list(self._clients.values()):
            try:
                client.close()
            except Exception:
                pass
        self._clients.clear()
        self._connected = False

    def _load_config(self) -> list[MCPServerConfig]:
        """Parse ``.mcp.json`` into a list of :class:`MCPServerConfig`."""
        if not MCP_CONFIG_PATH.exists():
            return []
        try:
            raw = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"\033[33m[MCP] Config parse error: {exc}\033[0m")
            return []
        servers = raw.get("mcpServers", {})
        if not isinstance(servers, dict):
            return []
        configs: list[MCPServerConfig] = []
        for name, spec in servers.items():
            if not isinstance(spec, dict):
                continue
            url = spec.get("url")
            if not url:
                continue
            configs.append(MCPServerConfig(
                name=name,
                url=url,
                headers=spec.get("headers", {}) or {},
                enabled=spec.get("enabled", True),
            ))
        return configs

    # -- tool schema generation ------------------------------------------- #

    def list_all_tool_schemas(self) -> list[dict]:
        """Return all MCP tools in OpenAI function-call schema format.

        Each tool name is prefixed ``mcp__{server}__{tool}``.
        """
        schemas: list[dict] = []
        for server_name, client in self._clients.items():
            try:
                tools = client.list_tools()
            except Exception as exc:
                print(
                    f"\033[33m[MCP] list_tools failed for '{server_name}': "
                    f"{exc}\033[0m"
                )
                continue
            for tool in tools:
                prefixed = f"{MCP_PREFIX}{server_name}__{tool.name}"
                desc = tool.description or tool.name
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": prefixed,
                        "description": f"[MCP:{server_name}] {desc}",
                        "parameters": tool.inputSchema or {
                            "type": "object", "properties": {},
                        },
                    },
                })
        return schemas

    # -- tool call routing ------------------------------------------------ #

    def call(self, full_name: str, args: dict) -> str:
        """Route a prefixed tool name to the appropriate server.

        ``full_name`` must match ``mcp__{server}__{tool}``.
        """
        parts = full_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            return f"Error: invalid MCP tool name: {full_name}"
        _, server, tool = parts
        client = self._clients.get(server)
        if client is None:
            return f"Error: MCP server '{server}' not connected"
        return client.call_tool(tool, args)

    # -- handler generation ----------------------------------------------- #

    def build_handlers(self) -> dict[str, callable]:
        """Generate a sync handler closure for each MCP tool."""
        handlers: dict[str, callable] = {}
        for schema in self.list_all_tool_schemas():
            name = schema["function"]["name"]
            handlers[name] = self._make_handler(name)
        return handlers

    def _make_handler(self, full_name: str):
        """Create a closure that routes to :meth:`call`."""
        def _handler(**kwargs) -> str:
            return self.call(full_name, kwargs)
        _handler.__name__ = full_name
        return _handler

    # -- status ----------------------------------------------------------- #

    @property
    def is_connected(self) -> bool:
        """Whether at least one MCP server is connected."""
        return self._connected and bool(self._clients)

    def status(self) -> str:
        """Human-readable connection summary."""
        if not self._clients:
            return "(no servers connected)"
        lines = []
        for name, client in self._clients.items():
            try:
                n = len(client.list_tools())
            except Exception:
                n = "?"
            lines.append(f"  - {name}: {n} tools @ {client.config.url}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Module-level convenience functions
# --------------------------------------------------------------------------- #

def init_mcp() -> None:
    """Initialize MCP at startup (connect to all configured servers)."""
    try:
        ctx.mcp.init()
        if ctx.mcp.is_connected:
            print(f"\033[36m[MCP] Connected:\n{ctx.mcp.status()}\033[0m")
    except Exception as exc:
        print(
            f"\033[33m[MCP] init failed (MCP tools disabled): "
            f"{type(exc).__name__}: {exc}\033[0m"
        )


def shutdown_mcp() -> None:
    """Shut down MCP at exit (close all sessions)."""
    try:
        ctx.mcp.shutdown()
    except Exception:
        pass
