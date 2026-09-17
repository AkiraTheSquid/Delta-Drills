#!/usr/bin/env python3
"""
Read what the question-repair loop has been doing, in a form fit for a terminal.

Three logs describe one repair, and each answers a different question:

  runs/repair_runs.jsonl        what the RUNNER did — every job it picked up,
                                including the ones that changed nothing and the
                                ones that failed. Written by run_repairs.py.
  ai_feedback_revisions.jsonl   what reached the BANK — before/after for every
                                applied repair and every rollback. Written by
                                the backend (feedback_ai_layer).
  ~/.claude/projects/.../*.jsonl  the CONVERSATION — the actual session, stored
                                by Claude Code itself. Rendered by `transcript`.

    history.py list                  # every run, newest first
    history.py list --failed         # only the ones that broke
    history.py show q214             # one repair: note, rationale, diff
    history.py transcript q214       # the session that produced it
    history.py transcript q214 --thinking --tools
    history.py queue                 # what is still waiting
    history.py revisions             # what is live in the bank right now

Every command takes --json for the raw records.

Selectors are a job id, `q<question id>` (newest run for that question), or
`latest`. `claude --resume <session-id>` reopens any of these conversations
interactively; `show` prints the id.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

OPS_DIR = Path(__file__).resolve().parent
REPO_ROOT = OPS_DIR.parents[1]
BACKEND_DIR = REPO_ROOT / "This-Directory-Only" / "backend"
RUNS_LOG = Path(
    os.environ.get("DELTA_REPAIR_RUNS_DIR")
    or Path.home() / ".local" / "state" / "delta-drills" / "question-repair"
) / "repair_runs.jsonl"

DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

STATUS_COLOR = {
    "done": GREEN,
    "skipped": DIM,
    "dry-run": DIM,
    "failed": RED,
    "pending": YELLOW,
    "running": YELLOW,
}


def paint(text: str, colour: str) -> str:
    return text if not sys.stdout.isatty() else f"{colour}{text}{RESET}"


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def short_time(stamp: str) -> str:
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime("%m-%d %H:%M")
    except (TypeError, ValueError):
        return (stamp or "")[:16]


def load_runs() -> List[dict]:
    return read_jsonl(RUNS_LOG)


def select(runs: List[dict], selector: str) -> Optional[dict]:
    """job id, q<question id>, or `latest`. Newest match wins."""
    if not runs:
        return None
    if selector in ("latest", "last"):
        return runs[-1]
    if selector.lower().startswith("q") and selector[1:].isdigit():
        wanted = int(selector[1:])
        matches = [r for r in runs if r.get("question_id") == wanted]
        return matches[-1] if matches else None
    matches = [r for r in runs if str(r.get("job_id", "")).startswith(selector)]
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def cmd_list(args) -> None:
    runs = load_runs()
    if args.failed:
        runs = [r for r in runs if r.get("status") == "failed"]
    if args.applied:
        runs = [r for r in runs if r.get("applied_fields")]
    if args.question:
        runs = [r for r in runs if r.get("question_id") == args.question]
    runs = runs[-args.limit:] if args.limit else runs
    if args.json:
        print(json.dumps(runs, indent=2))
        return
    if not runs:
        print("No repair runs recorded yet.")
        return

    print(f"{BOLD}{'when':<12} {'q':>5}  {'tag':<11} {'status':<8} {'changed':<34} {'$':>6}{RESET}"
          if sys.stdout.isatty() else
          f"{'when':<12} {'q':>5}  {'tag':<11} {'status':<8} {'changed':<34} {'$':>6}")
    for run in reversed(runs):
        status = str(run.get("status", "?"))
        changed = ", ".join(run.get("applied_fields") or []) or "-"
        cost = run.get("cost_usd") or 0
        print(
            f"{short_time(run.get('started_at', '')):<12} "
            f"{str(run.get('question_id', '?')):>5}  "
            f"{str(run.get('tag', '')):<11} "
            f"{paint(f'{status:<8}', STATUS_COLOR.get(status, RESET))} "
            f"{changed[:34]:<34} "
            f"{cost:>6.2f}"
        )
    total = sum(r.get("cost_usd") or 0 for r in runs)
    applied = sum(1 for r in runs if r.get("applied_fields"))
    print(f"\n{len(runs)} run(s), {applied} changed the bank, ${total:.2f} total.")


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

def diff_block(label: str, before: str, after: str) -> None:
    import difflib

    print(f"\n{paint(label, BOLD)}")
    lines = difflib.unified_diff(
        (before or "").splitlines(), (after or "").splitlines(),
        lineterm="", n=2, fromfile="before", tofile="after",
    )
    for line in lines:
        if line.startswith("+"):
            print(paint(line, GREEN))
        elif line.startswith("-"):
            print(paint(line, RED))
        elif line.startswith("@"):
            print(paint(line, DIM))
        else:
            print(line)


def cmd_show(args) -> None:
    run = select(load_runs(), args.selector)
    if run is None:
        sys.exit(f"No run matches {args.selector!r}")
    if args.json:
        print(json.dumps(run, indent=2))
        return

    print(f"{paint('question', BOLD)}  {run.get('question_id')}   "
          f"{paint(str(run.get('status')), STATUS_COLOR.get(str(run.get('status')), RESET))}")
    print(f"{paint('flagged', BOLD)}   {run.get('tag')} — {run.get('note') or '(no note)'}")
    print(f"{paint('verdict', BOLD)}   {run.get('verdict') or '-'}")
    print(f"{paint('rationale', BOLD)} {run.get('rationale') or '-'}")
    print(f"{paint('applied', BOLD)}   {', '.join(run.get('applied_fields') or []) or 'nothing'}")
    if run.get("error"):
        print(f"{paint('error', BOLD)}     {paint(str(run['error']), RED)}")

    for attempt in run.get("attempts") or []:
        bits = [f"attempt {attempt.get('attempt')}", str(attempt.get("verdict") or "?")]
        if attempt.get("verification"):
            bits.append(str(attempt["verification"]))
        if attempt.get("permission_denials"):
            denied = ", ".join(sorted({d.get("tool_name", "?") for d in attempt["permission_denials"]}))
            bits.append(paint(f"sandbox denied: {denied}", YELLOW))
        print(f"  {DIM if sys.stdout.isatty() else ''}· {' | '.join(bits)}{RESET if sys.stdout.isatty() else ''}")

    before, after = run.get("before") or {}, run.get("after") or {}
    for field in sorted(after):
        diff_block(field, before.get(field, ""), after.get(field, ""))

    session = run.get("session_id")
    if session:
        print(f"\n{paint('session', BOLD)}   {session}")
        print(f"{paint('replay', BOLD)}    {Path(__file__).name} transcript q{run.get('question_id')}")
        print(f"{paint('reopen', BOLD)}    claude --resume {session}")


# ---------------------------------------------------------------------------
# transcript
# ---------------------------------------------------------------------------

def _blocks(entry: dict) -> Iterable[dict]:
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content if isinstance(content, list) else []


def render_transcript(path: Path, show_thinking: bool, show_tools: bool, width: int) -> None:
    for entry in read_jsonl(path):
        kind = entry.get("type")
        if kind not in ("user", "assistant"):
            continue
        for block in _blocks(entry):
            btype = block.get("type")
            if btype == "text" and (block.get("text") or "").strip():
                who = "you" if kind == "user" else "claude"
                print(f"\n{paint(who, BOLD)}")
                print(block["text"].strip())
            elif btype == "thinking" and show_thinking:
                print(f"\n{paint('thinking', DIM)}")
                print(paint((block.get("thinking") or "").strip(), DIM))
            elif btype == "tool_use" and show_tools:
                summary = json.dumps(block.get("input") or {}, ensure_ascii=False)
                print(f"\n{paint('· ' + str(block.get('name')), YELLOW)} {summary[:width]}")
            elif btype == "tool_result" and show_tools:
                body = block.get("content")
                if isinstance(body, list):
                    body = " ".join(str(p.get("text", "")) for p in body if isinstance(p, dict))
                text = str(body or "").strip().replace("\n", " ")
                print(f"  {paint(text[:width], DIM)}")


def cmd_transcript(args) -> None:
    run = select(load_runs(), args.selector)
    if run is None:
        sys.exit(f"No run matches {args.selector!r}")
    path = Path(run.get("transcript") or "")
    if not path.exists():
        sys.exit(
            f"No stored transcript for {args.selector} "
            f"(session {run.get('session_id') or 'unknown'}). "
            "Claude Code prunes old sessions."
        )
    if args.json:
        print(json.dumps(read_jsonl(path), indent=2))
        return
    print(f"{paint('transcript', BOLD)} q{run.get('question_id')} · {run.get('session_id')}")
    print(paint(str(path), DIM))
    render_transcript(path, args.thinking, args.tools, args.width)


# ---------------------------------------------------------------------------
# queue / revisions — the server side of the same story
# ---------------------------------------------------------------------------

def _backend():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))


def cmd_queue(args) -> None:
    _backend()
    from app import feedback_repair_queue

    jobs = feedback_repair_queue.load_jobs()
    if args.json:
        print(json.dumps(jobs, indent=2))
        return
    if not jobs:
        print("Queue is empty.")
        return
    for job in jobs:
        status = str(job.get("status", "?"))
        print(
            f"{short_time(job.get('created_at', '')):<12} "
            f"q{str(job.get('question_id', '?')):<5} "
            f"{str(job.get('tag', '')):<11} "
            f"{paint(f'{status:<8}', STATUS_COLOR.get(status, RESET))} "
            f"{str(job.get('note', ''))[:60]}"
        )
    counts = feedback_repair_queue.summary()
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def cmd_revisions(args) -> None:
    _backend()
    from app import feedback_ai_layer

    entries = feedback_ai_layer.load_revisions(args.question)
    if args.json:
        print(json.dumps(entries, indent=2))
        return
    if not entries:
        print("No repairs have been applied to the bank.")
        return
    for entry in entries:
        status = str(entry.get("status", "?"))
        fields = ", ".join(entry.get("fields") or []) or entry.get("actor", "")
        print(
            f"{short_time(entry.get('timestamp', '')):<12} "
            f"q{str(entry.get('question_id', '?')):<5} "
            f"{status:<12} {fields:<34} {str(entry.get('rationale', ''))[:60]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command")

    p_list = subs.add_parser("list", help="every repair run, newest first")
    p_list.add_argument("--limit", type=int, default=40)
    p_list.add_argument("--failed", action="store_true")
    p_list.add_argument("--applied", action="store_true", help="only runs that changed the bank")
    p_list.add_argument("--question", type=int)
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_show = subs.add_parser("show", help="one repair in full, with a diff")
    p_show.add_argument("selector", nargs="?", default="latest")
    p_show.add_argument("--json", action="store_true")
    p_show.set_defaults(func=cmd_show)

    p_tr = subs.add_parser("transcript", help="the conversation behind a repair")
    p_tr.add_argument("selector", nargs="?", default="latest")
    p_tr.add_argument("--thinking", action="store_true", help="include the model's reasoning")
    p_tr.add_argument("--tools", action="store_true", help="include tool calls and results")
    p_tr.add_argument("--width", type=int, default=160, help="truncate tool lines at N chars")
    p_tr.add_argument("--json", action="store_true")
    p_tr.set_defaults(func=cmd_transcript)

    p_q = subs.add_parser("queue", help="jobs waiting, running, or finished")
    p_q.add_argument("--json", action="store_true")
    p_q.set_defaults(func=cmd_queue)

    p_rev = subs.add_parser("revisions", help="repairs live in the question bank")
    p_rev.add_argument("--question", type=int)
    p_rev.add_argument("--json", action="store_true")
    p_rev.set_defaults(func=cmd_revisions)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
