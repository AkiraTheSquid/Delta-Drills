"""MCP server over stdio — zero dependencies.

Deliberately hand-rolled rather than built on the `mcp` SDK: a contributor
should be able to clone this repo and point Claude Code at it with nothing
installed but a Python 3.9+ interpreter. The protocol surface an editing
server needs is small — initialize, tools/list, tools/call — and every tool
comes from `ops.REGISTRY`, so this file contains no knowledge of the content
itself.

🔴 stdout carries the JSON-RPC stream and NOTHING else. Every diagnostic goes
to stderr; one stray print here and the client sees a parse error instead of a
tool list.
"""

from __future__ import annotations

import json
import sys
import traceback

from . import ops

SERVER_NAME = "delta-drills-content"
SERVER_VERSION = "1.0.0"
DEFAULT_PROTOCOL = "2025-06-18"
# Versions whose tools surface this server actually implements. A client
# asking for one of these gets it back; anything else is answered with the
# default rather than agreed to blindly, which is what the spec asks for.
SUPPORTED_PROTOCOLS = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}

INSTRUCTIONS = """Programmatic editing of the Delta Drills course content.

Reading is open. Everything that writes a file needs the shared editing
password: call content_login once per session.

The three layers of content:
  * KP pages    — lesson_list / lesson_read / lesson_edit / lesson_write
  * the graph   — graph_list / graph_read / graph_add_kc / graph_update_kc
  * the drills  — drill_search / drill_read / drill_add / drill_update

Read lesson_authoring_guide before writing a page — the four-rung format is a
contract the validator enforces. After any change run pipeline_check; a change
that has not passed it has not landed. A snapshot of all content is taken
automatically before the first write of each day; backup_restore undoes a bad
session.
"""


def _tools() -> list[dict]:
    return [
        {
            "name": spec.name,
            "description": spec.summary + (" [requires content_login]" if spec.auth else ""),
            "inputSchema": spec.schema(),
        }
        for spec in sorted(ops.REGISTRY.values(), key=lambda s: s.name)
    ]


def _result(request_id, payload):
    return {"jsonrpc": "2.0", "id": request_id, "result": payload}


def _error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(message: dict):
    """Return a response dict, or None for a notification."""
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        requested = params.get("protocolVersion")
        agreed = requested if requested in SUPPORTED_PROTOCOLS else DEFAULT_PROTOCOL
        return _result(request_id, {
            "protocolVersion": agreed,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": INSTRUCTIONS,
        })

    if method in ("notifications/initialized", "notifications/cancelled", "initialized"):
        return None

    if method == "ping":
        return _result(request_id, {})

    if method == "tools/list":
        return _result(request_id, {"tools": _tools()})

    if method in ("resources/list", "prompts/list"):
        key = "resources" if method.startswith("resources") else "prompts"
        return _result(request_id, {key: []})

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            payload = ops.call(name, arguments)
            return _result(request_id, {
                "content": [{"type": "text", "text": ops.as_json(payload)}],
                "isError": False,
            })
        except Exception as err:  # a tool failure is a result, not a transport error
            detail = f"{type(err).__name__}: {err}"
            print(f"[{SERVER_NAME}] {name} failed: {detail}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return _result(request_id, {
                "content": [{"type": "text", "text": detail}],
                "isError": True,
            })

    if request_id is None:
        return None
    return _error(request_id, -32601, f"Method not found: {method}")


def serve(stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as err:
            stdout.write(json.dumps(_error(None, -32700, f"Parse error: {err}")) + "\n")
            stdout.flush()
            continue
        try:
            response = handle(message)
        except Exception as err:  # never let one bad message kill the server
            print(f"[{SERVER_NAME}] handler crashed: {err}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            response = _error(message.get("id"), -32603, f"Internal error: {err}")
        if response is not None:
            stdout.write(json.dumps(response, default=str) + "\n")
            stdout.flush()


def main() -> int:
    print(f"[{SERVER_NAME}] ready — {len(ops.REGISTRY)} tools", file=sys.stderr)
    try:
        serve()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
