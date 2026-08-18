#!/usr/bin/env python3
"""
Repair flagged Delta Drills questions with the local `claude` CLI.

A learner flags a question in the practice UI; the backend turns that into a job
(app/feedback_repair_queue.py) and stops. This script is the other half: it
picks the job up on Seth's machine, opens a read-only Claude Code session under
his own login, asks for a rewrite, verifies it, and writes it back into the
bank. Nothing here needs an API key — the session authenticates the same way
`claude` does when he types it himself.

    ops/question_repair/run_repairs.py --watch          # follow the queue
    ops/question_repair/run_repairs.py --once           # drain it and exit
    ops/question_repair/run_repairs.py --question 214 --tag unclear \
        --note "doesn't say what shape to return"       # repair one by hand
    ops/question_repair/history.py list                 # what it has done

Two sources, same code path:

  local   (default) the queue file and the bank on this machine. This is the
          right mode when the backend is running locally.
  remote  --api https://delta-drills-backend.fly.dev/api/practice, with an
          allowlisted user token. Production's queue, production's snapshot of
          the question, and the repair is applied to production's override
          layer on the Fly volume.

Why the session is read-only
----------------------------
It runs with --dangerously-skip-permissions so it never blocks on a prompt, and
sandbox_guard.py denies every tool except Read/Grep/Glob. The session's answer
is a JSON object validated against feedback_ai_improver.REPAIR_JSON_SCHEMA; THIS
script decides what to do with it. So the blast radius of the session is exactly
"it read some files and returned some text", no matter what it decides to try —
and the gates that protect the bank (narrow field set, compile check,
answer_code only on a `broken` flag, and a real grading-harness run) sit out here
where the model cannot reach them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OPS_DIR = Path(__file__).resolve().parent
REPO_ROOT = OPS_DIR.parents[1]
BACKEND_DIR = REPO_ROOT / "This-Directory-Only" / "backend"
VENV_PYTHON = BACKEND_DIR / ".venv" / "bin" / "python"
# Run history lives outside the repo on purpose: it is machine-local state, it
# grows without bound, and a data directory inside the tree collects tooling
# (READMEs, health checks, ignore rules) that has nothing to describe.
RUNS_DIR = Path(
    os.environ.get("DELTA_REPAIR_RUNS_DIR")
    or Path.home() / ".local" / "state" / "delta-drills" / "question-repair"
)
RUNS_LOG = RUNS_DIR / "repair_runs.jsonl"
GUARD = OPS_DIR / "sandbox_guard.py"

# One repair is a single CLI call plus a grading run. Ten minutes is generous;
# past that the session is stuck, not thinking.
CLI_TIMEOUT_SECONDS = 600
DEFAULT_INTERVAL_SECONDS = 60


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def ensure_backend_python() -> None:
    """Re-exec under the backend venv if we are not already there.

    Verification runs the rewritten answer through app.code_runner, which needs
    torch. System python does not have it, and a runner that silently skipped
    verification would be worse than one that refused to start.
    """
    # sys.prefix, not sys.executable: .venv/bin/python is a SYMLINK to the
    # system interpreter, so comparing resolved executable paths says "already
    # there" from any python on the machine and the re-exec silently never
    # happens. sys.prefix is what actually differs between the two.
    if Path(sys.prefix).resolve() == VENV_PYTHON.parents[1].resolve():
        return
    if not VENV_PYTHON.exists():
        sys.exit(
            f"Backend venv not found at {VENV_PYTHON}.\n"
            "The runner needs it to verify a rewritten answer against the grading harness."
        )
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


def load_backend_modules():
    """Import the shared prompt/gates/apply code out of the backend package."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from app import feedback_repair_queue  # noqa: WPS433 — deliberately late
    from app.practice import feedback_ai_improver

    return feedback_ai_improver, feedback_repair_queue


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Queue sources
# ---------------------------------------------------------------------------

class LocalQueue:
    """The queue and the bank on this machine, touched directly."""

    kind = "local"

    def __init__(self, improver, queue) -> None:
        self._improver = improver
        self._queue = queue

    def pending(self) -> List[dict]:
        sys.path.insert(0, str(BACKEND_DIR))
        from app import questions

        jobs = []
        for job in self._queue.pending_jobs():
            question = questions.get_question_by_id(int(job.get("question_id", -1)))
            if question is None:
                self._queue.finish(
                    job["job_id"], status=self._queue.SKIPPED,
                    error="question is no longer in the bank",
                )
                continue
            jobs.append({**job, "question": self._improver.question_snapshot(question)})
        return jobs

    def claim(self, job_id: str, runner: str) -> bool:
        return self._queue.claim(job_id, runner=runner) is not None

    def complete(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        job = self._queue.get_job(job_id) or {}
        if payload.get("error"):
            self._queue.finish(
                job_id, status=self._queue.FAILED, error=payload["error"],
                model=payload.get("model", ""), session_id=payload.get("session_id", ""),
            )
            return {"status": self._queue.FAILED, "applied_fields": []}

        revision = self._improver.apply_repair(
            int(job.get("question_id", -1)),
            payload,
            tag=str(job.get("tag", "")),
            trigger={
                "job_id": job_id,
                "user_email": job.get("user_email", ""),
                "tag": job.get("tag"),
                "note": job.get("note"),
                "correct": job.get("correct"),
                "flagged_at": job.get("created_at"),
                "completed_by": f"local-runner@{socket.gethostname()}",
            },
            model=payload.get("model", ""),
            session_id=payload.get("session_id", ""),
        )
        status = self._queue.DONE if revision else self._queue.SKIPPED
        self._queue.finish(
            job_id, status=status,
            rationale=payload.get("rationale", ""),
            model=payload.get("model", ""),
            session_id=payload.get("session_id", ""),
            applied_fields=(revision or {}).get("fields", []),
        )
        return {
            "status": status,
            "applied_fields": (revision or {}).get("fields", []),
            "revision": revision,
        }


class RemoteQueue:
    """Production's queue over HTTP, as an allowlisted user."""

    kind = "remote"

    def __init__(self, api: str, token: str) -> None:
        self._api = api.rstrip("/")
        self._token = token

    def _call(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        import urllib.error
        import urllib.request

        url = f"{self._api}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"{method} {path} -> {exc.code}: {detail}") from exc

    def pending(self) -> List[dict]:
        return list(self._call("GET", "/problem-feedback/repair-queue?status=pending").get("jobs") or [])

    def claim(self, job_id: str, runner: str) -> bool:
        try:
            self._call("POST", "/problem-feedback/repair-queue/claim", {"job_id": job_id, "runner": runner})
            return True
        except RuntimeError as exc:
            print(f"  claim refused: {exc}")
            return False

    def complete(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = {"job_id": job_id, **payload}
        return self._call("POST", "/problem-feedback/repair-queue/complete", body)


def build_source(args, improver, queue):
    if args.api:
        token = args.token or os.environ.get("DELTA_DRILLS_TOKEN", "").strip()
        if not token:
            token_file = Path.home() / ".config" / "delta-drills" / "token"
            if token_file.exists():
                token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            sys.exit(
                "Remote mode needs an allowlisted user token.\n"
                "Set DELTA_DRILLS_TOKEN, pass --token, or put it in ~/.config/delta-drills/token."
            )
        return RemoteQueue(args.api, token)
    return LocalQueue(improver, queue)


# ---------------------------------------------------------------------------
# The Claude Code session
# ---------------------------------------------------------------------------

def sandbox_settings() -> str:
    """--settings takes a JSON string, so the guard path can be computed here."""
    # The repo root is passed to the guard as an argument rather than read from
    # the environment: a hook inherits the session's env, and the session is the
    # thing being constrained.
    command = f'{shlex.quote(sys.executable)} {shlex.quote(str(GUARD))} {shlex.quote(str(REPO_ROOT))}'
    return json.dumps({
        "hooks": {
            "PreToolUse": [{
                "matcher": "*",
                "hooks": [{"type": "command", "command": command}],
            }],
        },
    })


def run_session(prompt: str, improver, model: str, verbose: bool) -> Dict[str, Any]:
    """One `claude -p` call. Returns the parsed result envelope.

    Read-only by construction: --tools names three read tools and the guard hook
    denies everything else regardless. --dangerously-skip-permissions only means
    "do not stop to ask" — with the guard in place there is nothing to ask about.
    """
    if not shutil.which("claude"):
        raise RuntimeError("`claude` is not on PATH — install Claude Code or fix PATH")

    command = [
        "claude", "-p",
        "--model", model,
        "--output-format", "json",
        "--json-schema", json.dumps(improver.REPAIR_JSON_SCHEMA),
        "--system-prompt", improver.SYSTEM_PROMPT,
        "--tools", "Read,Grep,Glob",
        "--settings", sandbox_settings(),
        "--dangerously-skip-permissions",
    ]
    if verbose:
        print(f"  $ claude -p --model {model} (read-only sandbox)")

    completed = subprocess.run(
        command,
        input=prompt,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=CLI_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"claude exited {completed.returncode}: {(completed.stderr or completed.stdout)[:500]}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"could not parse claude output: {completed.stdout[:500]}") from exc


def transcript_path(session_id: str) -> str:
    """Where Claude Code stored this session, for `history.py transcript`."""
    if not session_id:
        return ""
    slug = str(REPO_ROOT).replace("/", "-")
    candidate = Path.home() / ".claude" / "projects" / slug / f"{session_id}.jsonl"
    if candidate.exists():
        return str(candidate)
    matches = sorted((Path.home() / ".claude" / "projects").glob(f"*/{session_id}.jsonl"))
    return str(matches[0]) if matches else ""


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_answer(improver, answer_code: str, snapshot: Dict[str, Any]) -> Tuple[bool, str]:
    """Run a rewritten reference answer against the question's real test cases.

    Delegates to the backend's own gate rather than reimplementing it. The
    server runs the identical check before it writes, so a rule that lived only
    here would be a rule the endpoint did not have — and the endpoint is
    reachable without this runner.

    Running it here too is not redundant: a failure caught at this point can be
    fed back into a second attempt, which is the difference between a repair
    that lands and one that is thrown away.
    """
    return improver.verify_answer_code(answer_code, snapshot.get("test_cases"))


# ---------------------------------------------------------------------------
# One job
# ---------------------------------------------------------------------------

def process_job(job: dict, source, improver, args) -> dict:
    """Repair one flagged question end to end and record what happened."""
    snapshot = job.get("question") or {}
    question_id = job.get("question_id")
    tag = str(job.get("tag", ""))
    note = str(job.get("note", ""))
    record: Dict[str, Any] = {
        "job_id": job.get("job_id", ""),
        "question_id": question_id,
        "tag": tag,
        "note": note,
        "source": source.kind,
        "started_at": now(),
        "attempts": [],
    }
    print(f"\n[q{question_id}] {tag} — {note or '(no note)'}")

    prompt = improver.build_prompt(snapshot, tag, note, job.get("correct"))
    changes: Dict[str, str] = {}
    repair: Dict[str, Any] = {}
    envelope: Dict[str, Any] = {}

    try:
        for attempt in range(1, args.max_attempts + 1):
            envelope = run_session(prompt, improver, args.model, verbose=not args.quiet)
            repair = envelope.get("structured_output") or {}
            if not repair:
                # No structured answer at all. The session may have reasoned
                # perfectly and then had its reply blocked — StructuredOutput is
                # a tool, so the sandbox can deny it. Naming the denials here is
                # the difference between a five-minute fix and a loop that looks
                # like it politely declines every question forever.
                denied = ", ".join(sorted({
                    str(d.get("tool_name", "?"))
                    for d in envelope.get("permission_denials") or []
                })) or "none"
                raise RuntimeError(
                    "the session returned no structured answer "
                    f"(stop_reason={envelope.get('stop_reason')}, tools denied: {denied})"
                )
            record["attempts"].append({
                "attempt": attempt,
                "session_id": envelope.get("session_id", ""),
                "verdict": repair.get("verdict", ""),
                "rationale": repair.get("rationale", ""),
                "cost_usd": envelope.get("total_cost_usd"),
                "duration_ms": envelope.get("duration_ms"),
                "num_turns": envelope.get("num_turns"),
                "permission_denials": envelope.get("permission_denials") or [],
            })
            print(f"  verdict: {repair.get('verdict', '?')} — {repair.get('rationale', '').strip()}")

            changes = improver.validated_changes(repair, snapshot, tag)
            if "answer_code" not in changes:
                break

            ok, detail = verify_answer(improver, changes["answer_code"], snapshot)
            record["attempts"][-1]["verification"] = detail
            print(f"  verification: {'pass' if ok else 'FAIL'} — {detail}")
            if ok:
                break
            if attempt >= args.max_attempts:
                # Keep whatever else survived; a prose fix is still worth having.
                changes.pop("answer_code", None)
                print("  dropping answer_code — it never passed the question's own tests")
                break
            prompt = (
                f"{prompt}\n\nYour previous answer_code was rejected: {detail}\n"
                "Return a corrected repair. If you cannot make the reference answer pass "
                "the question's own test cases, set verdict to no_change."
            )
    except Exception as exc:
        record.update({"status": "failed", "error": str(exc), "finished_at": now()})
        print(f"  FAILED: {exc}")
        try:
            source.complete(job.get("job_id", ""), {
                "error": str(exc), "model": args.model,
                "session_id": (envelope or {}).get("session_id", ""),
            })
        except Exception as close_exc:
            # Reporting the failure failed too. The job stays claimed and is
            # handed back when the claim goes stale; the run log is the record.
            record["error"] = f"{exc} (could not close the job either: {close_exc})"
            print(f"  could not close the job: {close_exc}")
        write_run(record)
        return record

    session_id = str(envelope.get("session_id", ""))
    record.update({
        "model": args.model,
        "session_id": session_id,
        "transcript": transcript_path(session_id),
        "verdict": repair.get("verdict", ""),
        "rationale": repair.get("rationale", ""),
        "proposed_fields": sorted(changes),
        "before": {field: snapshot.get(field, "") for field in changes},
        "after": dict(changes),
        "cost_usd": sum(a.get("cost_usd") or 0 for a in record["attempts"]),
    })

    if args.dry_run:
        record.update({"status": "dry-run", "finished_at": now()})
        print(f"  dry run — would change {sorted(changes) or 'nothing'}")
        write_run(record)
        return record

    payload = {
        "verdict": repair.get("verdict", "no_change"),
        "rationale": repair.get("rationale", ""),
        "model": args.model,
        "session_id": session_id,
        # Only fields that survived every gate are offered. The server re-runs
        # the same gates on its own live copy of the question regardless.
        **{field: changes.get(field, "") for field in improver.EDITABLE_FIELDS},
    }
    try:
        result = source.complete(job.get("job_id", ""), payload)
    except Exception as exc:
        # A transport failure here is genuinely ambiguous — the server may have
        # applied the repair and lost the response. Say so rather than guessing,
        # keep the whole rewrite in the record so it can be replayed by hand,
        # and above all do not take the --watch loop down with it.
        record.update({
            "status": "failed", "finished_at": now(),
            "error": f"completion failed after the repair was produced: {exc}",
        })
        print(f"  FAILED to submit: {exc}")
        print("  the rewrite is kept in the run log — `history.py show` to replay it by hand")
        write_run(record)
        return record

    record.update({
        "status": result.get("status", "unknown"),
        "applied_fields": result.get("applied_fields", []),
        "finished_at": now(),
    })
    applied = result.get("applied_fields") or []
    print(f"  {record['status']}: {', '.join(applied) if applied else 'no change to the bank'}")
    write_run(record)
    return record


def write_run(record: dict) -> None:
    """Append to the runner's own history. Never rewritten; history.py reads it."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def manual_job(args, improver) -> dict:
    """--question: repair one question now, without waiting for a learner to flag it."""
    sys.path.insert(0, str(BACKEND_DIR))
    from app import questions

    question = questions.get_question_by_id(args.question)
    if question is None:
        sys.exit(f"Question {args.question} is not in the local bank")
    from app import feedback_repair_queue

    job = feedback_repair_queue.enqueue(
        question_id=args.question, tag=args.tag, note=args.note,
        correct=None, user_email=f"manual@{socket.gethostname()}",
    )
    return {**job, "question": improver.question_snapshot(question)}


def drain(source, improver, args) -> int:
    jobs = source.pending()
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        return 0
    runner = f"{socket.gethostname()}:{os.getpid()}"
    handled = 0
    for job in jobs:
        # A dry run must leave the queue exactly as it found it. Claiming would
        # hide the job from the real runner until the claim went stale, which is
        # a twenty-minute delay caused by a command that promised to change
        # nothing.
        if not args.dry_run and not source.claim(job.get("job_id", ""), runner):
            continue
        try:
            process_job(job, source, improver, args)
        except Exception as exc:
            # process_job handles its own failures; anything reaching here is a
            # surprise. One bad job must not end a --watch session.
            print(f"  unhandled error on q{job.get('question_id')}: {exc}")
        handled += 1
    return handled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="drain the queue and exit (default)")
    mode.add_argument("--watch", action="store_true", help="keep polling the queue")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS, help="seconds between polls in --watch")
    parser.add_argument("--api", default=os.environ.get("DELTA_DRILLS_API", ""), help="repair production instead: .../api/practice")
    parser.add_argument("--token", default="", help="allowlisted user token for --api")
    parser.add_argument("--model", default="opus", help="model alias or id for the repair session")
    parser.add_argument("--max-attempts", type=int, default=2, help="tries per job when the answer fails verification")
    parser.add_argument("--limit", type=int, default=0, help="stop after N jobs per pass")
    parser.add_argument("--dry-run", action="store_true", help="run the session, apply nothing")
    parser.add_argument("--quiet", action="store_true", help="less per-job chatter")
    parser.add_argument("--question", type=int, help="repair one question now instead of reading the queue")
    parser.add_argument("--tag", default="unclear", choices=("broken", "unclear", "wrong_image"), help="tag for --question")
    parser.add_argument("--note", default="", help="the complaint for --question")
    args = parser.parse_args()

    improver, queue = load_backend_modules()
    source = build_source(args, improver, queue)

    if args.question:
        if source.kind != "local":
            sys.exit("--question repairs the local bank; drop --api")
        job = manual_job(args, improver)
        if not args.dry_run:
            source.claim(job["job_id"], f"{socket.gethostname()}:{os.getpid()}")
        process_job(job, source, improver, args)
        return

    if not args.watch:
        handled = drain(source, improver, args)
        print(f"\n{handled} job(s) handled." if handled else "Queue is empty.")
        return

    print(f"Watching the {source.kind} repair queue every {args.interval}s. Ctrl-C to stop.")
    try:
        while True:
            handled = drain(source, improver, args)
            if handled:
                print(f"-- {handled} job(s) handled, back to watching")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    ensure_backend_python()
    main()
