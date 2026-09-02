"""watch.py — health checks for content-mcp/bin

These two scripts are what an MCP client and a human actually launch, so they
are checked by RUNNING them, not by reading them. The failure this guards
against is a banner on stdout, which breaks every MCP client while still
looking fine to a person running the command by hand.

Runs via `mod watch` — exit 0 = PASS, non-zero = FAIL.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CLI = os.path.join(HERE, "dd-content")
MCP = os.path.join(HERE, "dd-content-mcp")


def check_both_entry_points_are_executable():
    for path in (CLI, MCP):
        assert os.path.exists(path), f"missing {path}"
        assert os.access(path, os.X_OK), f"{os.path.basename(path)} is not executable"


def check_scripts_exec_rather_than_wrap():
    """An MCP client stops the server by signalling the process it spawned; a
    wrapper shell would swallow that and leave the server running."""
    for path in (CLI, MCP):
        body = open(path, encoding="utf-8").read()
        assert "exec " in body, f"{os.path.basename(path)} does not exec its target"


def check_mcp_entry_point_prints_only_json_to_stdout():
    request = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
    }) + "\n"
    proc = subprocess.run([MCP], input=request, capture_output=True, text=True, timeout=120)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"no response on stdout; stderr was: {proc.stderr[-400:]}"
    for line in lines:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            raise AssertionError(f"non-JSON on the JSON-RPC stream: {line[:200]!r}")
    reply = json.loads(lines[0])
    assert reply["result"]["serverInfo"]["name"] == "delta-drills-content"
    assert reply["result"]["capabilities"]["tools"] is not None


def check_cli_entry_point_runs_a_read_tool():
    proc = subprocess.run([CLI, "drill_next_id"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"dd-content failed: {proc.stderr[-400:]}"
    payload = json.loads(proc.stdout)
    assert isinstance(payload.get("next_id"), int), payload


def check_cli_reports_a_failure_through_its_exit_code():
    proc = subprocess.run([CLI, "lesson_read", "--kc", "nope.nope"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0, "a failing tool must fail the shell, not just print"


def check_registered_in_mcp_json():
    config_path = os.path.join(REPO, ".mcp.json")
    assert os.path.exists(config_path), ".mcp.json is missing — no client would find the server"
    servers = json.load(open(config_path, encoding="utf-8"))["mcpServers"]
    entry = servers.get("delta-drills-content")
    assert entry, "delta-drills-content is not registered in .mcp.json"
    command = os.path.join(REPO, entry["command"]) if entry["command"].startswith(".") else entry["command"]
    assert os.path.abspath(command) == os.path.abspath(MCP), (
        f".mcp.json points at {entry['command']}, not this folder's dd-content-mcp"
    )


if __name__ == "__main__":
    checks = [
        check_both_entry_points_are_executable,
        check_scripts_exec_rather_than_wrap,
        check_mcp_entry_point_prints_only_json_to_stdout,
        check_cli_entry_point_runs_a_read_tool,
        check_cli_reports_a_failure_through_its_exit_code,
        check_registered_in_mcp_json,
    ]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"PASS content-mcp/bin ({len(checks)} checks)")
