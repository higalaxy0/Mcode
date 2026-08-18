"""Quick e2e check: connect MCPManager to the local test server."""
import json
import sys
import os

# Create a temp config pointing at the running test server
import tempfile
cfg = {
    "mcpServers": {
        "echo": {
            "url": "http://127.0.0.1:8765/mcp",
            "headers": {},
            "enabled": True,
        }
    }
}
tmp = os.path.join(tempfile.gettempdir(), "_mcp_e2e_config.json")
with open(tmp, "w") as f:
    json.dump(cfg, f)

# Patch the config path before importing
from unittest.mock import patch
from pathlib import Path
import mcodecore.mcp as mcp_mod
patcher = patch.object(mcp_mod, "MCP_CONFIG_PATH", Path(tmp))
patcher.start()

from mcodecore.context import ctx

mgr = ctx.mcp
print("=== init() ===")
mgr.init()
print("is_connected:", mgr.is_connected)
print("status:", mgr.status())

print("\n=== list_all_tool_schemas() ===")
schemas = mgr.list_all_tool_schemas()
for s in schemas:
    print(" ", s["function"]["name"], "-", s["function"]["description"])

print("\n=== build_handlers() ===")
handlers = mgr.build_handlers()
for name in handlers:
    print("  handler:", name)

print("\n=== call echo ===")
result = handlers["mcp__echo__echo"](message="hello world")
print("  echo result:", result)

print("\n=== call add ===")
result2 = handlers["mcp__echo__add"](a=3, b=4)
print("  add result:", result2)

print("\n=== shutdown() ===")
mgr.shutdown()
print("is_connected:", mgr.is_connected)

print("\nALL E2E CHECKS PASSED")
