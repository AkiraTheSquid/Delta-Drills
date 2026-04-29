#!/usr/bin/env python3
"""Rewrite einops question prompts that leak the einops pattern.

Many einops questions in einops_problems.csv embed the arrow pattern
(e.g. "(b c h w -> b c w h)") directly in the prompt, so the student
doesn't have to derive it. This script finds those rows, asks an OpenAI
chat model to rephrase the prompt in plain English, and writes overrides
to ``einops_prompt_rewrite_overrides.jsonl``.

The backend's ``_load_function_overrides`` merges this file with the
existing ``function_mode_overrides.jsonl``; ``question_text`` from the
prompt-rewrite file wins on conflict, other fields from the function-mode
quality fix are preserved.

Usage:
    python3 rewrite_einops_prompts.py [--dry-run] [--limit N]

Requires OPENAI_API_KEY in env or a sibling api_key.txt / .openai_key / .env
(same loader as chatgpt/ChatGPT.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

# Reuse the api-key + model loaders from the existing batch runner.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chatgpt"))
from ChatGPT_batch import get_configured_model, load_api_key  # type: ignore

from openai import OpenAI

# ── pipeline bootstrap ──
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from delta_paths import CSV_DIR, SHARED_DIR, get_chatgpt_runtime_dir

CHATGPT_DIR = SHARED_DIR / "chatgpt"
SYSTEM_PROMPT_PATH = SHARED_DIR / "prompts" / "einops_prompt_rewrite_system.txt"
EINOPS_CSV = CSV_DIR / "einops_problems.csv"
OUT_PATH = get_chatgpt_runtime_dir() / "einops_prompt_rewrite_overrides.jsonl"

ARROW_RX = re.compile(r"->")
EINOPS_PATTERN_RX = re.compile(r"\b[a-zA-Z](?:\s+[a-zA-Z0-9()*]+){1,}\s*->\s*[a-zA-Z0-9()* ]+\b")


def find_leaky_einops_rows() -> list[dict]:
    """Stream the einops CSV, return rows whose prompt contains an arrow pattern."""
    if not EINOPS_CSV.exists():
        raise FileNotFoundError(f"Einops CSV not found at {EINOPS_CSV}")
    rows: list[dict] = []
    # CSV id schema mirrors backend/app/questions.py:_load_csv_into:
    # numpy CSV consumes IDs 1..N (skip 2 header rows), einsum continues, einops follows.
    numpy_count = _count_csv_rows(CSV_DIR / "Export of numpy problems with outputs.csv", skip=2)
    einsum_count = _count_csv_rows(CSV_DIR / "einsum_problems.csv", skip=0)
    base_id = numpy_count + einsum_count + 1
    with EINOPS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for offset, row in enumerate(reader):
            qid = base_id + offset
            qtext = (row.get("Question") or "").strip()
            if not qtext or not ARROW_RX.search(qtext):
                continue
            rows.append({
                "id": qid,
                "topic": row.get("Topic", "").strip(),
                "subtopic": row.get("Subtopic", "").strip(),
                "question_text": qtext,
                "answer_code": (row.get("Answer") or "").strip(),
            })
    return rows


def _count_csv_rows(path: Path, skip: int) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        for _ in range(skip):
            next(f, None)
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def build_user_prompt(row: dict) -> str:
    return (
        f"id: {row['id']}\n"
        f"original question_text: {row['question_text']}\n"
        f"canonical answer (do not change, only for context): {row['answer_code']}\n"
    )


def parse_response(raw: str, expected_id: int) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    rid = record.get("id")
    qt = record.get("question_text")
    if not isinstance(rid, int) or not isinstance(qt, str) or not qt.strip():
        return None
    if rid != expected_id:
        return None
    if ARROW_RX.search(qt) or EINOPS_PATTERN_RX.search(qt):
        # Refuse rewrites that still leak.
        return None
    return {"id": rid, "question_text": qt.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="List the questions to rewrite and exit; no API calls.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N rows (useful for sampling).")
    ap.add_argument("--model", default=None, help="Override the configured chat model.")
    args = ap.parse_args()

    rows = find_leaky_einops_rows()
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("No leaky einops prompts found. Nothing to do.")
        return 0

    print(f"Found {len(rows)} einops prompts containing an arrow pattern.")
    if args.dry_run:
        for r in rows[:10]:
            print(f"  id={r['id']:>3} [{r['subtopic']}] {r['question_text'][:90]}")
        if len(rows) > 10:
            print(f"  ... and {len(rows) - 10} more")
        return 0

    # Try both the code-dir (Local_Deployed_Shared/chatgpt) and the runtime-dir
    # (This-Directory-Only/chatgpt). The user's actual api_key.txt lives in the
    # runtime dir alongside the AI repair runtime artifacts.
    api_key = (
        load_api_key(str(CHATGPT_DIR))
        or load_api_key(str(get_chatgpt_runtime_dir()))
    )
    if not api_key:
        print("ERROR: no OPENAI_API_KEY in env / api_key.txt / .openai_key / .env", file=sys.stderr)
        return 2
    model = args.model or get_configured_model(str(CHATGPT_DIR))
    print(f"Using model: {model}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    client = OpenAI(api_key=api_key)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    rejected = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": build_user_prompt(row)},
                    ],
                    temperature=0.4,
                    response_format={"type": "json_object"},
                )
                raw = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"  id={row['id']}: API error: {exc}", file=sys.stderr)
                rejected += 1
                continue
            parsed = parse_response(raw, row["id"])
            if parsed is None:
                print(f"  id={row['id']}: rejected — response did not validate")
                print(f"    raw: {raw[:200]!r}")
                rejected += 1
                continue
            out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            print(f"  [{i}/{len(rows)}] id={row['id']} ✓ {parsed['question_text'][:80]}")
            time.sleep(0.05)  # gentle rate-limit cushion

    print(f"\nWrote {written} overrides, rejected {rejected}, to {OUT_PATH}")
    print("Restart the backend (delta_drills_local) for changes to load.")
    return 0 if rejected == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
