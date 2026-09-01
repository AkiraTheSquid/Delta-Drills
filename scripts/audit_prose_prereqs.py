#!/usr/bin/env python3
"""The `a.T` rule for WORDS: no page uses a term before the page that defines it.

`audit_solution_prereqs.py` enforces this for symbols; nothing enforced it for
prose. A learner on py-0 who meets "broadcasting" in a sentence is in the same
place as the one who met `a.T` in a starter: the course has used something it
has not yet taught, and nothing failed.

The vocabulary is NOT declared here — `lessons/glossary.js` already maps every
course term (plus aliases) to the KC that teaches it, and `watch_jargon.py`
already guards that mapping's integrity. This audit only asks the ordering
question: for each term used in a page's PROSE (front matter and fenced code
stripped — code is the symbol audits' jurisdiction), is the teaching KC at or
before this page in registry order? Matching mirrors jargon.js: word
boundaries, case-insensitive, longest surface form first.

A term on its own teaching page is fine (that IS the definition). A term whose
KC the registry no longer has is skipped — watch_jargon reports those.

RATCHET: `prose_prereq_baseline.json` records today's violations; watch.py
fails only on new ones.

Usage
-----
    python3 scripts/audit_prose_prereqs.py             # full report
    python3 scripts/audit_prose_prereqs.py --summary
    python3 scripts/audit_prose_prereqs.py --new
    python3 scripts/audit_prose_prereqs.py --write-baseline
Exit 1 when anything is reported (with --new, when anything is NEW).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_lesson_syntax import LESSONS, page_symbols, lesson_order  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Local_Deployed_Shared"))
from watch_jargon import glossary_terms  # noqa: E402

GLOSSARY = ROOT / "Local_Deployed_Shared" / "lessons" / "glossary.js"
BASELINE = Path(__file__).resolve().parent / "prose_prereq_baseline.json"

_FENCE = re.compile(r"^(```|~~~).*?^\1", re.M | re.S)
_INLINE = re.compile(r"`[^`\n]+`")
_FRONT = re.compile(r"\A---\n.*?\n---\n", re.S)


def prose_of(path: Path) -> str:
    """Prose only: front matter, fenced blocks AND inline `code` spans go —
    jargon.js never decorates code, and neither should this audit ('sorted'
    inside backticks is the builtin, not the concept of sorting)."""
    text = _FRONT.sub("", path.read_text(encoding="utf-8"))
    return _INLINE.sub(" ", _FENCE.sub("", text))


def term_scanner() -> tuple[re.Pattern, dict[str, tuple[str, str]]]:
    """One regex over EVERY surface form, longest first, plus form -> (term, kc).

    A single global scan is what makes matching longest-first ACROSS terms,
    not just within one term's aliases: "singleton axis" must be one use of
    'singleton axis', not additionally a use of 'axis' (codex round 1 — the
    per-term patterns each matched independently). Mirrors jargon.js.
    """
    owner: dict[str, tuple[str, str]] = {}
    for term, kc, aliases in glossary_terms(GLOSSARY.read_text(encoding="utf-8")):
        for form in {term, *aliases}:
            owner.setdefault(form.lower(), (term, kc))
    forms = sorted(owner, key=len, reverse=True)
    pat = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(f) for f in forms),
                     re.IGNORECASE)
    return pat, owner


def find() -> list[dict]:
    kc_of_page: dict[str, str] = {}
    pages: dict[str, Path] = {}
    for path in sorted(LESSONS.rglob("kp-*.md")):
        fm, _used, _errs = page_symbols(path)
        if fm.get("kc"):
            kc_of_page[path.name] = fm["kc"]
            pages[path.name] = path
    rank = lesson_order(kc_of_page)
    pat, owner = term_scanner()

    findings: list[dict] = []
    for name, path in pages.items():
        my_rank = rank.get(kc_of_page[name])
        if my_rank is None:
            continue
        prose = prose_of(path)
        seen: set[str] = set()
        for m in pat.finditer(prose):
            term, kc = owner[m.group(0).lower()]
            owner_rank = rank.get(kc)
            if (term in seen or owner_rank is None or owner_rank <= my_rank):
                continue  # dup / retired KC (watch_jargon's) / already taught
            seen.add(term)
            line = prose.count("\n", 0, m.start()) + 1
            findings.append({
                "key": f"{name}|{term}",
                "detail": (f"{name} (rank {my_rank}) says {m.group(0)!r} "
                           f"(~line {line} of prose) but {kc} (rank "
                           f"{owner_rank}) is what defines '{term}' — "
                           "the page uses a word the course has not taught")})
    return findings


def load_baseline() -> set[str] | None:
    """None (absent) and empty are different answers — see the prereq audit."""
    if not BASELINE.exists():
        return None
    return set(json.loads(BASELINE.read_text(encoding="utf-8")).get("known") or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()

    findings = find()
    if args.write_baseline:
        BASELINE.write_text(json.dumps({
            "_": "Known prose use-before-definition. watch.py fails on anything "
                 "NOT listed here. Shrink it; re-record only with a reason.",
            "count": len(findings),
            "known": sorted(f["key"] for f in findings),
        }, indent=1) + "\n", encoding="utf-8")
        print(f"baseline written: {len(findings)} finding(s)")
        return 0

    known = load_baseline() or set()
    new = [f for f in findings if f["key"] not in known]
    shown = new if args.new else findings
    if not args.summary:
        for f in sorted(shown, key=lambda f: f["key"]):
            print(f["detail"])
    print(f"{len(findings)} finding(s); {len(new)} new vs baseline", file=sys.stderr)
    return 1 if (new if args.new else findings) else 0


if __name__ == "__main__":
    sys.exit(main())
