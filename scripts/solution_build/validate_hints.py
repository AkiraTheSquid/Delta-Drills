#!/usr/bin/env python3
"""Validate authored drill hints. Each: non-empty, 15-400 chars, has a known
drill path. Usage: validate_hints.py <hints.jsonl>. Exit 0 iff all pass."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PATHS = {r["path"] for r in json.loads((HERE / "drill_hint_inputs.json").read_text())}
def check(o):
    p, h = o.get("path"), (o.get("hint") or "").strip()
    if p not in PATHS: return "unknown path"
    if not h: return "empty"
    if len(h) < 15: return "too short"
    if len(h) > 400: return "too long"
    return ""
src = Path(sys.argv[1])
items = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
bad = {o.get("path"): check(o) for o in items if check(o)}
print(json.dumps({"total": len(items), "pass": len(items)-len(bad), "fail": len(bad), "fails": bad}, indent=2))
sys.exit(0 if not bad else 1)
