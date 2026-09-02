"""Programmatic editing of the Delta Drills course content.

Two front ends over one registry of operations: an MCP server (`server.py`)
so Claude Code can edit the content conversationally, and a CLI (`cli.py`)
for scripts and humans. Both dispatch through `ops.call`.
"""

__version__ = "1.0.0"
