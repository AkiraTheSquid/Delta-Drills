#!/usr/bin/env python3
"""Expand terse coding-drill prompts into comprehensive, self-contained problem statements.

Many bank questions are very short ("Merge heads: (b h t d) -> (b t (h*d)).") and
don't tell the student the input shape, the desired output shape, or the domain
meaning. This script:

  1. Loads ``questions_full.json`` (the layered export the backend actually
     serves) and selects questions whose ``question_text`` is shorter than
     ``--min-len`` chars (default 120).
  2. Sends each (prompt, canonical answer, setup_code, expected_expr) tuple
     to OpenAI with the ``prompt_expansion_system.txt`` system prompt.
  3. The model returns ``{id, question_text}``. The rewritten prompt is
     validated for length and anti-leakage (no einops arrow patterns) and
     written to ``prompt_expansion_overrides.jsonl`` in the chatgpt runtime
     dir.
  4. Backend ``_load_function_overrides()`` must be extended to layer this
     file LAST so its question_text wins on collision; runnable scaffolding
     from earlier rounds survives.

Usage:
    python3 expand_prompts.py --dry-run --limit 5
    python3 expand_prompts.py --limit 20

Requires OPENAI_API_KEY in env or sibling api_key.txt / .openai_key
(same loader as ``chatgpt/ChatGPT_batch.py``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ── pipeline bootstrap ──
import sys as _sys
from pathlib import Path as _Path
_SHARED = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_SHARED))                   # delta_paths
_sys.path.insert(0, str(_SHARED / "chatgpt"))       # ChatGPT_batch

from delta_paths import SHARED_DIR, get_chatgpt_runtime_dir
from ChatGPT_batch import get_configured_model, load_api_key  # type: ignore
from openai import OpenAI

CHATGPT_DIR = SHARED_DIR / "chatgpt"
SYSTEM_PROMPT_PATH = SHARED_DIR / "prompts" / "prompt_expansion_system.txt"
VERIFIER_PROMPT_PATH = SHARED_DIR / "prompts" / "prompt_expansion_verifier_system.txt"
QUESTIONS_FULL = SHARED_DIR.parent / "This-Directory-Only" / "questions_full.json"
OUT_PATH = get_chatgpt_runtime_dir() / "prompt_expansion_overrides.jsonl"
REJECT_PATH = get_chatgpt_runtime_dir() / "prompt_expansion_rejected.jsonl"

ARROW_RX = re.compile(r"->")
EINOPS_PATTERN_RX = re.compile(
    r"['\"][a-zA-Z][^'\"]*?\s*->\s*[^'\"]*?['\"]"
)
STAR_RX = re.compile(r"\s*\(★[★☆]+\)\s*")
PRINT_SUFFIX_RX = re.compile(r"\s*\n?Print the result\.\s*$", flags=re.IGNORECASE)


def _normalize_to_original_format(rewrite: str, original: str) -> str:
    """Deterministic post-process: strip stars and 'Print the result.' from rewrite
    if not present in original; re-add them at end if they ARE present in original
    but missing from rewrite.
    """
    out = rewrite.strip()
    orig_stars_match = STAR_RX.search(original)
    has_orig_stars = orig_stars_match is not None
    orig_stars = orig_stars_match.group(0).strip() if has_orig_stars else ""

    has_orig_print = bool(PRINT_SUFFIX_RX.search(original))

    # Strip stars from rewrite — we'll re-add at end if original had them
    out = STAR_RX.sub(" ", out)
    # Strip "Print the result." suffix — we'll re-add if original had it
    out = PRINT_SUFFIX_RX.sub("", out).rstrip()
    # Tidy doubled spaces
    out = re.sub(r"\s{2,}", " ", out).strip()
    # Ensure trailing period if no other terminal punctuation
    if out and out[-1] not in ".!?":
        out += "."
    # Re-attach
    if has_orig_stars:
        out = out + " " + orig_stars
    if has_orig_print:
        out = out + "\nPrint the result."
    return out


def load_candidates(min_len: int) -> list[dict]:
    if not QUESTIONS_FULL.exists():
        raise FileNotFoundError(f"questions_full.json not found at {QUESTIONS_FULL}")
    qs = json.loads(QUESTIONS_FULL.read_text(encoding="utf-8"))
    out: list[dict] = []
    for q in qs:
        qtxt = (q.get("question_text") or "").strip()
        if len(qtxt) >= min_len:
            continue
        tcs = q.get("test_cases") or []
        tc = tcs[0] if tcs else {}
        out.append({
            "id": q["id"],
            "topic": q.get("topic", ""),
            "subtopic": q.get("subtopic_key") or q.get("subtopic", ""),
            "question_text": qtxt,
            "answer_code": (q.get("answer_code") or "").strip(),
            "setup_code": (tc.get("setup_code") or "").strip(),
            "expected_expr": (tc.get("expected_expr") or "").strip(),
            "expected_artifact_type": q.get("expected_artifact_type", "stdout"),
        })
    return out


def build_user_prompt(row: dict) -> str:
    artifact = row.get("expected_artifact_type", "stdout")
    artifact_note = (
        "expected_artifact_type: stdout — original may end with 'Print the result.'; preserve only if literally present."
        if artifact == "stdout"
        else f"expected_artifact_type: {artifact} — this problem returns a value/image, NOT printed output. Do NOT add 'Print the result.'"
    )
    return (
        f"id: {row['id']}\n"
        f"topic: {row['topic']}\n"
        f"subtopic: {row['subtopic']}\n"
        f"{artifact_note}\n"
        f"original prompt:\n{row['question_text']}\n\n"
        f"canonical answer (do not name its functions in the rewrite):\n{row['answer_code']}\n\n"
        f"setup_code (input fixture — use to infer shapes):\n{row['setup_code']}\n\n"
        f"expected_expr (produces expected value):\n{row['expected_expr']}\n"
    )


def build_verifier_prompt(row: dict, rewrite_text: str) -> str:
    return (
        f"ORIGINAL prompt:\n{row['question_text']}\n\n"
        f"REWRITTEN prompt:\n{rewrite_text}\n\n"
        f"CANONICAL answer:\n{row['answer_code']}\n\n"
        f"SETUP_CODE:\n{row['setup_code']}\n"
    )


def parse_verifier_response(raw: str) -> tuple[str | None, list[str]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None, ["verifier-json-decode-failed"]
    if not isinstance(record, dict):
        return None, ["verifier-not-a-dict"]
    verdict = record.get("verdict")
    reasons = record.get("reasons") or []
    if verdict not in ("pass", "fail"):
        return None, ["verifier-bad-verdict"]
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return verdict, [str(r) for r in reasons]


def parse_response(raw: str, expected_id: int) -> tuple[dict | None, str | None]:
    """Return (record, reject_reason). record is None iff rejected."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None, "json-decode"
    if not isinstance(record, dict):
        return None, "not-a-dict"
    rid = record.get("id")
    qt = record.get("question_text")
    if not isinstance(rid, int) or rid != expected_id:
        return None, "id-mismatch"
    if not isinstance(qt, str) or not qt.strip():
        return None, "missing-question-text"
    qt_clean = qt.strip()
    if len(qt_clean) < 80:
        return None, f"too-short ({len(qt_clean)} chars)"
    if len(qt_clean) > 700:
        return None, f"too-long ({len(qt_clean)} chars)"
    word_count = len(qt_clean.split())
    if word_count > 110:  # 100-word cap + small grace margin
        return None, f"too-many-words ({word_count})"
    if EINOPS_PATTERN_RX.search(qt_clean):
        return None, "still-leaks-einops-pattern"
    return {"id": rid, "question_text": qt_clean}, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="List candidates and (with --sample) print model output for a few; never writes the override file.")
    ap.add_argument("--sample", action="store_true",
                    help="With --dry-run: actually call the API on --limit rows so you can preview the rewrites. Prints to stdout only.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N candidates.")
    ap.add_argument("--stratify", action="store_true",
                    help="When combined with --limit, evenly stride across the full candidate list so the sample spans all subtopics.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the verifier self-check pass (saves ~50%% of API spend).")
    ap.add_argument("--min-len", type=int, default=100,
                    help="Only expand prompts shorter than this many chars (default 100).")
    ap.add_argument("--model", default=None, help="Override the configured chat model.")
    ap.add_argument("--ids", type=str, default=None,
                    help="Comma-separated id allowlist (e.g. '18,65,384'). Overrides --min-len filter.")
    args = ap.parse_args()

    candidates = load_candidates(args.min_len)
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",") if x.strip()}
        # Re-load all and filter by id allowlist (bypass min-len)
        all_qs = json.loads(QUESTIONS_FULL.read_text(encoding="utf-8"))
        candidates = []
        for q in all_qs:
            if q["id"] not in wanted:
                continue
            tcs = q.get("test_cases") or []
            tc = tcs[0] if tcs else {}
            candidates.append({
                "id": q["id"],
                "topic": q.get("topic", ""),
                "subtopic": q.get("subtopic_key") or q.get("subtopic", ""),
                "question_text": (q.get("question_text") or "").strip(),
                "answer_code": (q.get("answer_code") or "").strip(),
                "setup_code": (tc.get("setup_code") or "").strip() if tc else "",
                "expected_expr": (tc.get("expected_expr") or "").strip() if tc else "",
                "expected_artifact_type": q.get("expected_artifact_type", "stdout"),
            })
    if args.limit:
        if args.stratify and args.limit < len(candidates):
            step = len(candidates) / args.limit
            picked_indices = sorted({int(i * step) for i in range(args.limit)})
            candidates = [candidates[i] for i in picked_indices if i < len(candidates)]
        else:
            candidates = candidates[: args.limit]
    if not candidates:
        print("No candidates. Nothing to do.")
        return 0

    print(f"Found {len(candidates)} candidate prompts (min-len={args.min_len}).")
    if args.dry_run and not args.sample:
        for r in candidates[:15]:
            print(f"  id={r['id']:>3} len={len(r['question_text']):>3} [{r['subtopic']}] {r['question_text'][:90]!r}")
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
    verifier_prompt = VERIFIER_PROMPT_PATH.read_text(encoding="utf-8") if not args.no_verify else None
    client = OpenAI(api_key=api_key)

    written = 0
    rejected = 0
    verifier_failures: list[dict] = []
    sample_records: list[dict] = []  # for --dry-run --sample mode

    def _verify(row: dict, rewrite_text: str) -> tuple[str, list[str]]:
        """Call verifier model. Returns (verdict, reasons). verdict is 'pass', 'fail', or 'error'."""
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": verifier_prompt},
                    {"role": "user", "content": build_verifier_prompt(row, rewrite_text)},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content or ""
        except Exception as exc:
            return "error", [f"verifier-api-error: {exc}"]
        verdict, reasons = parse_verifier_response(raw)
        if verdict is None:
            return "error", reasons
        return verdict, reasons

    def _process(row: dict) -> tuple[dict | None, str | None, str]:
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
            return None, f"api-error: {exc}", ""
        parsed, reject_reason = parse_response(raw, row["id"])
        if parsed is None:
            return None, reject_reason, raw
        # Deterministic post-process: enforce star + "Print the result." presence
        # matches original exactly. Catches the whole F7 hallucinated-suffix class
        # without spending a verifier call.
        parsed["question_text"] = _normalize_to_original_format(
            parsed["question_text"], row["question_text"]
        )
        if verifier_prompt is not None:
            verdict, reasons = _verify(row, parsed["question_text"])
            if verdict != "pass":
                verifier_failures.append({
                    "id": row["id"],
                    "verdict": verdict,
                    "reasons": reasons,
                    "rewrite": parsed["question_text"],
                })
                return None, f"verifier-{verdict}: {'; '.join(reasons[:2])}", raw
        return parsed, None, raw

    if args.dry_run and args.sample:
        # No file write — print model output to stdout only.
        for i, row in enumerate(candidates, 1):
            parsed, reason, raw = _process(row)
            print(f"\n[{i}/{len(candidates)}] id={row['id']} [{row['subtopic']}]")
            print(f"  ORIG ({len(row['question_text'])} chars): {row['question_text']!r}")
            if parsed is None:
                print(f"  REJECTED: {reason}")
                print(f"  RAW: {raw[:300]!r}")
                rejected += 1
            else:
                print(f"  NEW ({len(parsed['question_text'])} chars): {parsed['question_text']!r}")
                sample_records.append(parsed)
                written += 1
            time.sleep(0.05)
        print(f"\nDry-run sample: would-write {written}, rejected {rejected}")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for i, row in enumerate(candidates, 1):
            parsed, reason, raw = _process(row)
            if parsed is None:
                print(f"  id={row['id']}: rejected — {reason}")
                if reason and reason.startswith("api-error"):
                    print(f"    {reason}", file=sys.stderr)
                else:
                    print(f"    raw: {raw[:200]!r}")
                rejected += 1
                continue
            out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            print(f"  [{i}/{len(candidates)}] id={row['id']} ✓ ({len(parsed['question_text'])}c) {parsed['question_text'][:80]}")
            time.sleep(0.05)

    print(f"\nWrote {written} expansions, rejected {rejected}, to {OUT_PATH}")
    if verifier_failures:
        REJECT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REJECT_PATH.open("w", encoding="utf-8") as rf:
            for fail in verifier_failures:
                rf.write(json.dumps(fail, ensure_ascii=False) + "\n")
        print(f"Verifier rejected {len(verifier_failures)}; details in {REJECT_PATH}")
    print("Restart the backend (delta_drills_local) for changes to load.")
    return 0 if rejected == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
