#!/usr/bin/env python3
"""
PreToolUse hook: the sandbox the question-repair session runs inside.

The repair session is started with --dangerously-skip-permissions, because it
has to run unattended while Seth is doing something else. That flag turns off
the PROMPTS; it does not turn off hooks. This is what actually holds the line.

Policy: the session may look **inside this repository**, and nothing else.

  allowed   Read, Grep, Glob, TodoWrite  — reading the repo it was started in
            StructuredOutput             — how the CLI hands back the answer
  denied    everything else, by default, including every tool added to Claude
            Code after this file was written
  denied    any of the above aimed outside the repo root passed as argv[1]

Both halves matter. A tool-name allowlist alone still leaves `Read` pointed at
`~/.ssh/id_ed25519` or `~/.claude/.credentials.json`, and this session returns
free text that is written to a log — so "read-only" without a path check is
"can quietly copy any file this user owns into the repair rationale". The
learner note that seeds the prompt is attacker-supplied text, which is exactly
the shape of input that talks a read-only agent into reading the wrong file.

Deny-by-default is the whole design. An allowlist that has to be updated when a
new tool ships is an allowlist that is wrong the day it ships, and this process
runs with permissions bypassed. A repair session that genuinely needs a new
capability should have it added here deliberately.

The cost of that strictness, learned the hard way: StructuredOutput is a TOOL,
not a reply. Denying it produced sessions that reasoned correctly and then
returned nothing at all — the runner saw an empty verdict and every repair
looked like a no-op. If a whole run comes back blank, read
`history.py show <id>`: a denial listed there names whatever the guard just
locked out.

Reads a hook event on stdin, writes a hook response on stdout. Any failure in
this script denies the call: a guard that cannot decide must not wave the tool
through.
"""

from __future__ import annotations

import json
import os
import sys

ALLOWED_TOOLS = frozenset({"Read", "Grep", "Glob", "TodoWrite", "StructuredOutput"})

# Which fields of a tool call name a filesystem location. Anything not listed
# here is not inspected — which is safe only because the tools that reach this
# point are the read-only ones above.
PATH_FIELDS = ("file_path", "path", "notebook_path", "pattern", "glob")


def respond(decision: str, reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def escapes(value: str, root: str) -> bool:
    """True when `value` names somewhere outside `root`.

    Relative paths are resolved against the root, which is also the session's
    cwd. A glob pattern is checked the same way: `**/*.py` stays inside, while
    `/home/**/*.pem` and `../../.ssh/*` do not. `os.path.realpath` collapses
    both `..` and symlinks, so a link planted inside the repo cannot point out
    of it.
    """
    if not value:
        return False
    candidate = value if os.path.isabs(value) else os.path.join(root, value)
    resolved = os.path.realpath(candidate)
    return resolved != root and not resolved.startswith(root + os.sep)


def main() -> None:
    root = os.path.realpath(sys.argv[1]) if len(sys.argv) > 1 else ""
    try:
        event = json.load(sys.stdin)
    except Exception:
        respond("deny", "question-repair sandbox: unreadable hook event")
        return

    tool = str(event.get("tool_name") or "")
    if tool in ALLOWED_TOOLS:
        if not root:
            respond("deny", "question-repair sandbox: no repository root was configured")
        tool_input = event.get("tool_input") or {}
        for field in PATH_FIELDS:
            value = tool_input.get(field)
            if isinstance(value, str) and escapes(value, root):
                respond(
                    "deny",
                    f"question-repair sandbox: {field}={value!r} is outside the "
                    "repository. This session may only read Delta Drills.",
                )
        respond("allow", f"question-repair sandbox: {tool} is read-only, inside the repo")
    respond(
        "deny",
        f"question-repair sandbox: {tool or 'this tool'} is not available. "
        "This session is read-only — return the rewrite in the JSON answer and "
        "the runner will apply it.",
    )


if __name__ == "__main__":
    main()
