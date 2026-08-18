"""Integration smoke test: spin up a real MCP server, connect, list + call tools.

Run manually (requires network-free local server):
    python tests/_mcp_e2e.py
"""
import asyncio
import sys

from mcp.server.fastmcp import FastMCP

server = FastMCP("echo-server")
server.settings.host = "127.0.0.1"
server.settings.port = 8765


@server.tool()
def echo(message: str) -> str:
    """Echo back the provided message."""
    return f"echo: {message}"


@server.tool()
def add(a: int, b: int) -> str:
    """Add two numbers and return the sum as a string."""
    return str(a + b)


async def main():
    # Run the server using Streamable HTTP transport (uses settings.host/port)
    await server.run_streamable_http_async()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
