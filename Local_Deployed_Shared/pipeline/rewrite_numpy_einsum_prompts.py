#!/usr/bin/env python3
"""Rewrite numpy + einsum prompts that name the canonical answer function.

Many numpy/einsum questions describe the task by naming the same numpy
function used in the canonical answer (e.g. "Row-wise argmax", "Tile X
three times", "Compute the cumulative sum"). That gives the answer away
before the student has to reason about which primitive applies.

This script:
  1. Loads ``questions_structured.json`` and selects numpy + einsum
     questions whose prompt contains a candidate leak token.
  2. Sends each (prompt, canonical answer) pair to OpenAI with the
     ``numpy_einsum_prompt_rewrite_system.txt`` system prompt.
  3. The model returns ``{id, leak, question_text?}``. Only records with
     ``leak=true`` and a non-leaky rewrite are written.
  4. Output: ``numpy_einsum_prompt_rewrite_overrides.jsonl`` in the
     chatgpt runtime dir. Backend ``_load_function_overrides()`` merges
     it after the function-mode rounds and the einops rewrite, layered
     last so its question_text wins on collision while preserving
     starter / test_cases from earlier rounds.

Usage:
    python3 rewrite_numpy_einsum_prompts.py [--dry-run] [--limit N]

Requires OPENAI_API_KEY in env or a sibling api_key.txt / .openai_key
(same loader as ``chatgpt/ChatGPT.py``). Honors the same model-config
and runtime-dir paths as ``rewrite_einops_prompts.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chatgpt"))
from ChatGPT_batch import get_configured_model, load_api_key  # type: ignore

from openai import OpenAI

# ── pipeline bootstrap ──
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from delta_paths import SHARED_DIR, get_chatgpt_runtime_dir

CHATGPT_DIR = SHARED_DIR / "chatgpt"
SYSTEM_PROMPT_PATH = SHARED_DIR / "prompts" / "numpy_einsum_prompt_rewrite_system.txt"
STRUCTURED_JSON = SHARED_DIR / "questions_structured.json"
OUT_PATH = get_chatgpt_runtime_dir() / "numpy_einsum_prompt_rewrite_overrides.jsonl"

# Tokens that, if present in the prompt, *might* indicate the prompt names
# the canonical answer function. We pre-filter on these to avoid sending
# the entire 296-question numpy+einsum bank to the API. The model is the
# final arbiter — it returns leak=false for setup-only mentions like
# "Given Z = np.arange(...)".
LEAK_TOKENS = {
    "argmax", "argmin", "argsort", "argpartition",
    "cumsum", "cumprod", "diff",
    "tile", "stack", "vstack", "hstack", "concatenate", "block",
    "where", "nonzero", "argwhere", "flatnonzero",
    "unique", "intersect1d", "setdiff1d", "union1d",
    "histogram", "bincount", "digitize",
    "meshgrid", "indices", "ix_",
    "cross", "dot", "matmul", "tensordot", "outer", "inner", "kron",
    "trace", "diag", "diagonal", "diagflat",
    "norm", "solve", "inv", "det", "eig", "svd",
    "rot90", "flip", "fliplr", "flipud", "transpose", "swapaxes", "moveaxis",
    "clip", "round", "floor", "ceil", "trunc", "sign",
    "minimum", "maximum",
    "any", "all", "isin", "isclose", "allclose",
    "pad", "roll",
    "cov", "corrcoef", "var", "std", "mean", "median", "average",
    "fft", "ifft", "rfft",
    "convolve", "correlate",
    "ravel", "reshape", "squeeze",
    "einsum",
    # task verbs that often name the canonical operation
    "tile", "repeat",
    "center", "centered", "centering",
    "broadcast",
    "swap",
    "negate",
}

LEAK_TOKEN_RX = re.compile(
    r"(?<![a-zA-Z])(" + "|".join(sorted(LEAK_TOKENS, key=len, reverse=True)) + r")(?![a-zA-Z])",
    flags=re.IGNORECASE,
)


def find_leak_candidates() -> list[dict]:
    structured = json.loads(STRUCTURED_JSON.read_text(encoding="utf-8"))
    candidates: list[dict] = []
    for q in structured:
        topic = (q.get("curriculum") or {}).get("subtopic_key", "")
        if "Einops" in topic and "Einsum" not in topic:
            continue
        ex = q.get("exercise") or {}
        qtxt = (ex.get("question_text") or "").strip()
        answer = (ex.get("canonical_solution") or "").strip()
        if not qtxt or not answer:
            continue
        if not LEAK_TOKEN_RX.search(qtxt):
            continue
        candidates.append({
            "id": q["id"],
            "subtopic": topic,
            "question_text": qtxt,
            "answer_code": answer,
        })
    return candidates


def build_user_prompt(row: dict) -> str:
    return (
        f"id: {row['id']}\n"
        f"subtopic: {row['subtopic']}\n"
        f"original prompt:\n{row['question_text']}\n\n"
        f"canonical answer (do not change, only for leak detection):\n{row['answer_code']}\n"
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
    leak = record.get("leak")
    if rid != expected_id or not isinstance(leak, bool):
        return None
    if not leak:
        return {"id": rid, "leak": False}
    qt = record.get("question_text")
    if not isinstance(qt, str) or not qt.strip():
        return None
    return {"id": rid, "leak": True, "question_text": qt.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="List candidate questions and exit; no API calls.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N candidates (useful for sampling).")
    ap.add_argument("--model", default=None, help="Override the configured chat model.")
    args = ap.parse_args()

    candidates = find_leak_candidates()
    if args.limit:
        candidates = candidates[: args.limit]
    if not candidates:
        print("No leak candidates found. Nothing to do.")
        return 0

    print(f"Found {len(candidates)} numpy+einsum questions matching a leak token.")
    if args.dry_run:
        for r in candidates[:15]:
            print(f"  id={r['id']:>3} [{r['subtopic']}] {r['question_text'][:90]}")
        if len(candidates) > 15:
            print(f"  ... and {len(candidates) - 15} more")
        return 0

    api_key = (
        load_api_key(str(CHATGPT_DIR))
        or load_api_key(str(get_chatgpt_runtime_dir()))
    )
    if not api_key:
        print("ERROR: no OPENAI_API_KEY in env / api_key.txt / .openai_key", file=sys.stderr)
        return 2
    model = args.model or get_configured_model(str(CHATGPT_DIR))
    print(f"Using model: {model}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    client = OpenAI(api_key=api_key)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_no_leak = 0
    rejected = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for i, row in enumerate(candidates, 1):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": build_user_prompt(row)},
                    ],
                    temperature=0.3,
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
            if not parsed["leak"]:
                skipped_no_leak += 1
                print(f"  [{i}/{len(candidates)}] id={row['id']} · no leak")
                continue
            record = {"id": parsed["id"], "question_text": parsed["question_text"]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            print(f"  [{i}/{len(candidates)}] id={row['id']} ✓ {record['question_text'][:80]}")
            time.sleep(0.05)

    print(f"\nWrote {written} rewrites, skipped {skipped_no_leak} (no leak), "
          f"rejected {rejected}, to {OUT_PATH}")
    print("Restart the backend (delta_drills_local) for changes to load.")
    return 0 if rejected == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
