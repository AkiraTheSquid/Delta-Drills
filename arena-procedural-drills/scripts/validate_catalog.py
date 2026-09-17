#!/usr/bin/env python3
"""Static validator for drills-catalog.js. Runs without a browser.

Checks:
  1. JS file parses as ES module (no syntax errors) — via Node if available,
     otherwise regex-based shape sanity.
  2. Every C() and E() entry references a notebook path that EXISTS on disk.
  3. Every entry's atom_ids/subtopics list is non-empty and length-aligned.
  4. Composite ids are unique across the whole catalog (composite:<part>:cx<idx>).
  5. Single-atom ids are unique across the whole catalog (drill:<atom>:ex<N>).
  6. Every C() entry has 8 positional args (part, idx, title, primary, atoms, subs, path, heading).

Run from repo root:
  arena-book/.venv/bin/python arena-procedural-drills/scripts/validate_catalog.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "Local_Deployed_Shared" / "practice" / "drills-catalog.js"


def _parse_with_node(src: str) -> bool:
    try:
        r = subprocess.run(
            ["node", "--check", "-"],
            input=src,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return r.returncode == 0, r.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, "node unavailable"


def main() -> int:
    src = CATALOG.read_text()
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Syntax (best-effort via node)
    ok, msg = _parse_with_node(src)
    if ok is False:
        errors.append(f"[syntax] node --check failed: {msg[:400]}")
    elif ok is None:
        warnings.append(f"[syntax] {msg} — falling back to regex shape sanity only")

    # 2. Extract E() and C() entries
    # C("part", idx, "title", "primary", [atoms], [subs], "path", "heading")
    # E("atom", idx, "title", "subtopic", "path", "heading")
    e_entries = []
    c_entries = []

    for m in re.finditer(r"^\s*E\((.*?)\),\s*$", src, flags=re.MULTILINE):
        e_entries.append((m.start(), m.group(1)))
    for m in re.finditer(r"^\s*C\((.*?)\),\s*$", src, flags=re.MULTILINE):
        c_entries.append((m.start(), m.group(1)))

    if not e_entries and not c_entries:
        errors.append("[shape] zero E() or C() entries matched — catalog block missing or shape changed")

    # 3. Validate each entry
    ids_seen: dict[str, int] = {}
    for pos, body in e_entries:
        # Quick split — single-line JSON-ish args
        # Try parsing args by eval-ish: replace JS true/false/null and parse
        try:
            args = json.loads(f"[{body}]".replace("true", "true").replace("false", "false"))
        except Exception:
            warnings.append(f"[E@{pos}] could not parse args (likely contains opts={{...}})")
            continue
        if len(args) < 6:
            errors.append(f"[E@{pos}] expected ≥6 args, got {len(args)}")
            continue
        atom, idx, title, sub, path, heading = args[:6]
        eid = f"drill:{atom}:ex{idx}"
        if eid in ids_seen:
            errors.append(f"[E@{pos}] duplicate id: {eid}")
        ids_seen[eid] = pos
        if not (REPO / path).exists():
            errors.append(f"[E@{pos}] notebook path missing: {path}")

    for pos, body in c_entries:
        try:
            args = json.loads(f"[{body}]")
        except Exception:
            errors.append(f"[C@{pos}] could not parse args; body[:120]={body[:120]}")
            continue
        if len(args) < 8:
            errors.append(f"[C@{pos}] expected ≥8 args, got {len(args)}")
            continue
        part, idx, title, primary, atoms, subs, path, heading = args[:8]
        cid = f"composite:{part}:cx{idx}"
        if cid in ids_seen:
            errors.append(f"[C@{pos}] duplicate id: {cid}")
        ids_seen[cid] = pos
        if not isinstance(atoms, list) or not atoms:
            errors.append(f"[C@{pos}] {cid}: atom_ids must be non-empty list")
        if not isinstance(subs, list) or not subs:
            errors.append(f"[C@{pos}] {cid}: subtopics must be non-empty list")
        if isinstance(atoms, list) and isinstance(subs, list) and len(atoms) != len(subs):
            errors.append(f"[C@{pos}] {cid}: atoms({len(atoms)}) != subtopics({len(subs)})")
        if not (REPO / path).exists():
            errors.append(f"[C@{pos}] notebook path missing: {path}")

    print(f"E() entries: {len(e_entries)}")
    print(f"C() entries: {len(c_entries)}")
    print(f"unique ids:  {len(ids_seen)}")

    for w in warnings:
        print(f"WARN {w}")
    for e in errors:
        print(f"ERR  {e}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
