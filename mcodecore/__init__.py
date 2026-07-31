"""mcodecore core package.

Modules:
- ``config``      path constants / API config / OpenAI client
- ``exceptions``  custom exceptions
- ``context``     AppContext - container for mutable runtime state
- ``utils``       generic helper functions
- ``tasks``       task board CRUD
- ``bus``         message bus + protocol state machine
- ``hooks``       hook registration and dispatch
- ``skills``      skill registry
- ``memory``      memory read/write / extract / consolidate
- ``fsops``       filesystem & shell tools
- ``streaming``   streaming response wrappers (dataclass)
- ``compact``     token estimation / context compaction / persistence
- ``mcp``         MCP (Model Context Protocol) Streamable HTTP client
- ``tools``       tool schemas + handler mapping + system prompt
- ``subagent``    synchronous sub-agent
- ``teammates``   threaded teammate agents
- ``agent``       lead agent main loop + REPL entry
"""

from .exceptions import AgentInterrupt
from .hooks import install_default_hooks

# Register default hooks at import time.
install_default_hooks()

__all__ = ["AgentInterrupt"]
