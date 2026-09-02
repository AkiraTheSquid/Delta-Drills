"""watch.py — health checks for content_mcp

Module-level invariants. The cross-cutting ones (registry vs CLI vs server,
the password gate, snapshot coverage) live in `../watch.py`.

Runs via `mod watch` — exit 0 = PASS, non-zero = FAIL.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = ["paths", "auth", "backup", "lessons", "graph", "drills", "pipeline", "ops",
           "server", "cli"]


def check_imports():
    import importlib
    for name in MODULES:
        importlib.import_module(f"content_mcp.{name}")


def check_no_third_party_imports():
    """Standard library only, so a fresh clone works with no pip install."""
    import ast
    here = os.path.dirname(os.path.abspath(__file__))
    # The two repo modules this package deliberately reuses instead of copying:
    # the canonical KP parser and the authority on question-id assignment. Both
    # are imported lazily, inside the function that needs them.
    repo_modules = {"lesson_lib", "export_questions_json"}
    allowed = set(sys.stdlib_module_names) | repo_modules | {"content_mcp", ""}
    for name in MODULES:
        tree = ast.parse(open(os.path.join(here, f"{name}.py"), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]] if node.level == 0 else [""]
            else:
                continue
            for root in roots:
                assert root in allowed, (
                    f"{name}.py imports third-party '{root}' — this package must run "
                    "on a bare interpreter"
                )


def check_server_writes_nothing_to_stdout():
    """stdout is the JSON-RPC transport. One stray print breaks every client."""
    from content_mcp import server
    captured = io.StringIO()
    real, sys.stdout = sys.stdout, captured
    try:
        server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        server.handle({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}})
        server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "does_not_exist", "arguments": {}}})
    finally:
        sys.stdout = real
    assert captured.getvalue() == "", f"server wrote to stdout: {captured.getvalue()[:200]!r}"


def check_a_failing_tool_is_a_result_not_a_crash():
    from content_mcp import server
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "lesson_read", "arguments": {"kc": "nope.nope"}}})
    assert reply["result"]["isError"] is True, "a bad argument must come back as isError"
    assert "error" not in reply, "a tool failure must not be a JSON-RPC transport error"


def check_notifications_get_no_reply():
    from content_mcp import server
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def check_content_actually_parses():
    """Every KP page on disk must still parse — this is the read path the model
    depends on, and a half-written page would silently disappear from listings."""
    from content_mcp import lessons
    pages = lessons.list_kps()
    broken = [p for p in pages if "parse_error" in p]
    assert not broken, f"unparseable KP page(s): {[p['file'] for p in broken]}"
    assert len(pages) > 30, f"only {len(pages)} KP pages found — the lessons path looks wrong"


def check_registry_is_structurally_sound():
    from content_mcp import graph
    problems = graph.check(graph.load())
    assert not problems, f"kc_registry.json: {problems}"


def check_override_whitelist_matches_the_exporter():
    """A field the exporter does not whitelist is silently ignored, so an edit
    using one would report success and change nothing."""
    from content_mcp import drills
    source = open(drills._export_module().__file__, encoding="utf-8").read()
    for field in sorted(drills.OVERRIDE_FIELDS):
        assert f'"{field}"' in source, (
            f"'{field}' is offered as an override field but the exporter never reads it"
        )


if __name__ == "__main__":
    checks = [
        check_imports,
        check_no_third_party_imports,
        check_server_writes_nothing_to_stdout,
        check_a_failing_tool_is_a_result_not_a_crash,
        check_notifications_get_no_reply,
        check_content_actually_parses,
        check_registry_is_structurally_sound,
        check_override_whitelist_matches_the_exporter,
    ]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"PASS content_mcp ({len(checks)} checks)")
