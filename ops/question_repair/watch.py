"""watch.py — health checks for question_repair

This runner starts a Claude Code session with --dangerously-skip-permissions on
Seth's own machine. Two things stand between that and a bad day, and neither
fails loudly if it is quietly removed:

  the guard hook   is actually wired into the session that gets started
  the gates        run out here, in the runner, not inside the prompt

Everything below pins one of those. The behavioural end-to-end coverage (does a
bad answer get rejected, does rollback restore the bank) lives in
backend/app/practice/watch.py, which can import the backend package.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import ast
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(THIS, '..', '..'))


def _source(name):
    with open(os.path.join(THIS, name), encoding='utf-8') as fh:
        return fh.read()


def check_imports():
    """Both scripts parse, and the guard the runner names actually exists."""
    for name in ('run_repairs.py', 'history.py', 'sandbox_guard.py'):
        ast.parse(_source(name))

    runner = _source('run_repairs.py')
    assert 'GUARD = OPS_DIR / "sandbox_guard.py"' in runner, (
        "run_repairs.py no longer points at sandbox_guard.py by that name — if "
        "the path is wrong the hook silently never runs and the session is "
        "unsandboxed"
    )
    assert os.path.exists(os.path.join(THIS, 'sandbox_guard.py'))


def check_public_api():
    """The session must be started sandboxed, and read-only.

    --dangerously-skip-permissions without the settings hook is a session that
    can do anything to this machine. They belong to each other.
    """
    runner = _source('run_repairs.py')
    assert '"--dangerously-skip-permissions"' in runner, (
        "the runner no longer bypasses prompts — it runs unattended, so it "
        "would hang on the first tool call instead"
    )
    assert '"--settings", sandbox_settings()' in runner, (
        "the sandbox settings are no longer passed to the CLI, but "
        "--dangerously-skip-permissions still is — that is an unsandboxed "
        "session with permissions bypassed"
    )
    assert '"--tools", "Read,Grep,Glob"' in runner, (
        "the repair session's tool list changed — it is supposed to be able to "
        "read the repo and nothing else; the answer is applied by the runner"
    )
    assert '"--json-schema"' in runner, (
        "the session no longer answers in a validated schema — the runner would "
        "be parsing prose into the question bank"
    )


def check_invariants():
    """The guard denies by default, and the gates are not left to the model."""
    tree = ast.parse(_source('sandbox_guard.py'))
    allowed = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'ALLOWED_TOOLS' for t in node.targets
        ):
            allowed = {c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant)}
    assert allowed is not None, "sandbox_guard.py no longer defines ALLOWED_TOOLS"
    for writer in ('Write', 'Edit', 'NotebookEdit', 'Bash', 'Task', 'WebFetch', 'WebSearch'):
        assert writer not in allowed, (
            f"the repair sandbox now allows {writer} — this session runs with "
            "permissions bypassed, so ALLOWED_TOOLS is the only thing stopping it"
        )
    # Learned the hard way: this one is a TOOL, not a reply. Denying it makes
    # every repair come back empty while looking like a polite "no change".
    assert 'StructuredOutput' in allowed, (
        "StructuredOutput is denied — the session cannot return its answer at "
        "all, and every repair will silently look like a no-op"
    )

    # Tool-name allowlisting alone leaves Read pointed at ~/.ssh. Run the guard
    # for real rather than reading it — this is the boundary, and a boundary
    # nobody exercises is a boundary nobody knows is broken.
    import json
    import subprocess

    repo = os.path.dirname(os.path.dirname(THIS))
    cases = (
        ({'tool_name': 'Read', 'tool_input': {'file_path': os.path.join(repo, 'README.md')}}, 'allow'),
        ({'tool_name': 'Read', 'tool_input': {'file_path': 'This-Directory-Only/fly.toml'}}, 'allow'),
        ({'tool_name': 'Read', 'tool_input': {'file_path': os.path.expanduser('~/.ssh/id_rsa')}}, 'deny'),
        ({'tool_name': 'Read', 'tool_input': {'file_path': '../../etc/passwd'}}, 'deny'),
        ({'tool_name': 'Glob', 'tool_input': {'pattern': '**/*.py'}}, 'allow'),
        ({'tool_name': 'Glob', 'tool_input': {'pattern': '/home/**/*.pem'}}, 'deny'),
        ({'tool_name': 'Grep', 'tool_input': {'pattern': 'solve', 'path': os.path.expanduser('~')}}, 'deny'),
        ({'tool_name': 'Write', 'tool_input': {'file_path': os.path.join(repo, 'x')}}, 'deny'),
    )
    for event, expected in cases:
        result = subprocess.run(
            [sys.executable, os.path.join(THIS, 'sandbox_guard.py'), repo],
            input=json.dumps(event), capture_output=True, text=True,
        )
        decision = json.loads(result.stdout)['hookSpecificOutput']['permissionDecision']
        assert decision == expected, (
            f"sandbox {event['tool_name']} {event['tool_input']} -> {decision}, "
            f"expected {expected}; this session reads with permissions bypassed "
            "and returns free text, so an out-of-repo read is an exfiltration path"
        )

    runner = _source('run_repairs.py')
    assert 'shlex.quote(str(REPO_ROOT))' in runner, (
        "the guard is no longer told which repository it is guarding — with no "
        "root argument it denies everything, and the loop stops working"
    )
    assert 'improver.verify_answer_code' in runner, (
        "a rewritten reference answer is no longer run against the question's "
        "own test cases — a wrong answer here marks every future learner wrong"
    )
    assert 'improver.validated_changes' in runner, (
        "the runner stopped applying the shared gates before sending a repair"
    )
    assert 'def process_job' in runner and runner.count('write_run(record)') >= 3, (
        "a path through process_job no longer records what happened — an "
        "unrecorded run is a repair nobody can find, review, or roll back"
    )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
