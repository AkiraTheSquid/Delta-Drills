"""watch.py — health checks for content-mcp

The contract this folder has to keep: one registry, two front ends that cannot
drift from it, and a password on every op that writes.

Runs via `mod watch` — exit 0 = PASS, non-zero = FAIL.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# Every op that changes a file on disk. Kept here rather than derived, so
# adding a write op without a password gate FAILS instead of passing quietly.
WRITE_OPS = frozenset({
    "backup_now", "backup_restore",
    "drill_add", "drill_retire", "drill_update",
    "graph_add_kc", "graph_update_kc",
    "lesson_create", "lesson_edit", "lesson_write",
    "pipeline_check", "pipeline_step",
})


def check_imports():
    from content_mcp import auth, backup, cli, drills, graph, lessons, ops, pipeline, server  # noqa: F401
    assert ops.REGISTRY, "op registry is empty"


def check_every_op_is_well_formed():
    from content_mcp import ops
    for name, spec in ops.REGISTRY.items():
        assert spec.summary.strip(), f"{name} has no summary — it is what the model reads"
        assert callable(spec.handler), f"{name} has no handler"
        schema = spec.schema()
        assert schema["type"] == "object", f"{name} schema is not an object"
        unknown = set(spec.required) - set(schema["properties"])
        assert not unknown, f"{name} requires undeclared param(s): {sorted(unknown)}"


def check_writes_are_password_gated():
    from content_mcp import ops
    gated = {name for name, spec in ops.REGISTRY.items() if spec.auth}
    missing = WRITE_OPS - gated
    assert not missing, f"write op(s) with NO password gate: {sorted(missing)}"
    extra = gated - WRITE_OPS
    assert not extra, (
        f"op(s) gated but not listed as writes: {sorted(extra)}. "
        "Add them to WRITE_OPS here if they really write."
    )


def check_cli_covers_every_op():
    from content_mcp import cli, ops
    parser = cli.build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    subcommands = set()
    for action in actions:
        subcommands.update(action.choices)
    missing = set(ops.REGISTRY) - subcommands
    assert not missing, f"CLI is missing subcommand(s) for: {sorted(missing)}"


def check_server_lists_every_op():
    from content_mcp import ops, server
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = listed["result"]["tools"]
    assert len(tools) == len(ops.REGISTRY), (
        f"tools/list returned {len(tools)} for {len(ops.REGISTRY)} ops"
    )
    for tool in tools:
        assert tool["description"], f"{tool['name']} reaches the model with no description"
        assert tool["inputSchema"]["type"] == "object"


def check_initialize_negotiates_a_supported_version():
    from content_mcp import server
    for requested, expected in (
        ("2025-06-18", "2025-06-18"),
        ("1999-01-01", server.DEFAULT_PROTOCOL),
        (None, server.DEFAULT_PROTOCOL),
    ):
        reply = server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": requested},
        })
        got = reply["result"]["protocolVersion"]
        assert got == expected, f"asked {requested}, server agreed to {got}"


def check_password_is_stored_as_a_digest():
    from content_mcp import auth
    record = auth._stored()
    assert record, "no password configured — writes would be unusable"
    assert record.get("digest") and record.get("salt"), "auth.json is missing salt/digest"
    blob = json.dumps(record).lower()
    assert "password" not in record, "auth.json must never hold a plaintext password"
    assert "arenautodidacts" not in blob, "the password itself leaked into auth.json"


def check_backup_covers_the_content():
    from content_mcp import paths
    missing = [rel for rel, _ in paths.CONTENT_PATHS if not (paths.REPO / rel).exists()]
    assert not missing, f"snapshot spec names path(s) that do not exist: {missing}"
    # Name the actual files an author can destroy, not the folders that hold
    # them: the folder list is allowed to change shape, the coverage is not.
    must_cover = {
        "the KP pages": paths.LESSONS / "numpy",
        "the concept graph": paths.KC_REGISTRY,
        "authored drills": paths.CURATED_CSV,
        "drill overrides": paths.CURATED_OVERRIDES,
        "the retirement list": paths.RETIRED_IDS,
    }
    covered = [paths.REPO / rel for rel, _ in paths.CONTENT_PATHS]
    for label, target in must_cover.items():
        assert any(target == root or root in target.parents for root in covered), (
            f"the snapshot does not cover {label} ({target}) — "
            "a mistake there would be unrecoverable"
        )


def check_a_graph_write_does_not_reformat_the_file():
    """A one-node addition must be a one-node diff.

    `kc_registry.json` is written with indent 1. Rewriting it with the json
    module's default produced a 414-line diff for a single new concept —
    unreviewable, and a guaranteed conflict with any other session editing it.
    """
    import json as _json
    from content_mcp import graph, paths
    on_disk = paths.KC_REGISTRY.read_text()
    indent = paths.json_indent_of(paths.KC_REGISTRY)
    round_tripped = _json.dumps(graph.load(), indent=indent) + "\n"
    assert round_tripped == on_disk, (
        "load -> save is not byte-identical: a graph write would reformat the "
        "whole registry instead of touching the lines it changed"
    )


def check_local_state_is_gitignored():
    """The snapshot is 2.7MB of duplicated content and the session file is a
    credential. Neither belongs in a commit."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", ".content-mcp/content-backup.tar.gz"],
        cwd=REPO, capture_output=True,
    )
    assert result.returncode == 0, ".content-mcp/ is not gitignored"


def check_restore_never_rotates_the_snapshot_first():
    """The 2026-09-02 near-miss: `ops.call` snapshots before every write, and
    `backup_restore` is a write. With a stale snapshot that rotated the broken
    tree over the good copy and then restored the breakage."""
    from content_mcp import backup, ops
    assert ops.REGISTRY["backup_restore"].snapshot is False, (
        "backup_restore would snapshot before restoring — that destroys the copy "
        "being restored from whenever the snapshot is stale"
    )
    assert ops.REGISTRY["backup_now"].snapshot is False, (
        "backup_now manages its own rotation; snapshotting first is a double rotation"
    )

    # Behavioural: a stale snapshot must survive a restore untouched.
    token = _probe_token()
    if token is None:
        return  # not logged in; the declarative assertions above still hold

    rotated, restored = [], []
    real_ensure, real_restore = backup.ensure, backup.restore
    backup.ensure = lambda: rotated.append(True) or {}
    backup.restore = lambda *a, **k: restored.append(True) or {"restored_count": 0}
    try:
        ops.call("backup_restore", {"confirm": True, "token": token})
    finally:
        backup.ensure, backup.restore = real_ensure, real_restore
    # Without this the check goes green when the call never ran at all.
    assert restored, "the probe never reached backup.restore — check is not proving anything"
    assert not rotated, "restore rotated the snapshot before restoring"


def check_a_failed_snapshot_blocks_the_write():
    """Fail closed: no safety copy, no mutation."""
    from content_mcp import backup
    assert hasattr(backup, "SnapshotError"), "ensure() has no failure type — it swallows"
    real = backup.snapshot
    backup.snapshot = lambda force=False: (_ for _ in ()).throw(OSError("disk full"))
    try:
        backup.ensure()
    except backup.SnapshotError:
        pass
    except OSError:
        raise AssertionError("ensure() leaked a raw OSError instead of failing closed")
    else:
        raise AssertionError("ensure() returned normally when the snapshot failed")
    finally:
        backup.snapshot = real


def check_a_rejected_drill_leaves_no_row_behind():
    """An invalid override used to raise AFTER the CSV row was committed,
    so a reported failure still added a drill and a retry added a second."""
    from content_mcp import drills, paths
    before = paths.CURATED_CSV.read_bytes()
    try:
        drills.add("Python", "Getting started", "q", "a",
                   override={"not_a_real_field": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("an override with an unknown field was accepted")
    assert paths.CURATED_CSV.read_bytes() == before, (
        "a rejected drill_add left a row in curated_additions.csv"
    )


def check_unknown_drill_ids_are_refused():
    from content_mcp import drills
    ghost = drills.next_id() + 5000
    for label, fn in (("update", lambda: drills.update(ghost, {"question_text": "x"})),
                      ("retire", lambda: drills.retire(ghost))):
        try:
            fn()
        except KeyError:
            continue
        raise AssertionError(f"drill_{label} accepted an id that is not in the bank")


def check_password_change_needs_the_current_password():
    from content_mcp import auth
    try:
        auth.write_password("something-else")
    except auth.AuthError:
        return
    raise AssertionError("the editing password can be replaced without knowing it")


def _probe_token():
    """A token the auth gate accepts, for behavioural checks — never persisted."""
    from content_mcp import auth
    record = auth._live_session()
    if record:
        return record["token"]
    import os
    return os.environ.get("DELTA_DRILLS_CONTENT_PASSWORD") or None


def check_entry_points_are_executable():
    for name in ("dd-content", "dd-content-mcp"):
        path = os.path.join(HERE, "bin", name)
        assert os.path.exists(path), f"missing entry point {name}"
        assert os.access(path, os.X_OK), f"{name} is not executable"


if __name__ == "__main__":
    checks = [
        check_imports,
        check_every_op_is_well_formed,
        check_writes_are_password_gated,
        check_cli_covers_every_op,
        check_server_lists_every_op,
        check_initialize_negotiates_a_supported_version,
        check_password_is_stored_as_a_digest,
        check_backup_covers_the_content,
        check_a_graph_write_does_not_reformat_the_file,
        check_local_state_is_gitignored,
        check_restore_never_rotates_the_snapshot_first,
        check_a_failed_snapshot_blocks_the_write,
        check_a_rejected_drill_leaves_no_row_behind,
        check_unknown_drill_ids_are_refused,
        check_password_change_needs_the_current_password,
        check_entry_points_are_executable,
    ]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"PASS content-mcp ({len(checks)} checks)")
