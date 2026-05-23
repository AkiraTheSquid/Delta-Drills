"""Diff our concept_edges + prereq_manual + prereqs against a blind second pass.

Usage:
    python concept-graph/scripts/compare_edges.py <other-dir>

`other-dir` should contain:
  - concept_edges.json
  - prereq_manual.json   (optional)
  - prereqs.json         (optional — recomputed if missing)

Outputs a Markdown report to concept-graph/COMPARE_REPORT.md plus prints summary.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def concept_edges_to_set(edges: list[dict]) -> tuple[set, set, dict]:
    """Return (full_triples, fromto_pairs, by_fromto_pair_to_kind)."""
    triples = set()
    pairs = set()
    by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for e in edges:
        f, t, k = e["from"], e["to"], e["kind"]
        triples.add((f, t, k))
        pairs.add((f, t))
        by_pair[(f, t)].append(k)
    return triples, pairs, dict(by_pair)


def prereq_edges_to_set(edges: list[dict]) -> set:
    return {(e["from"], e["to"]) for e in edges}


def diff_concept(mine: list[dict], theirs: list[dict]) -> dict:
    mine_t, mine_p, mine_k = concept_edges_to_set(mine)
    their_t, their_p, their_k = concept_edges_to_set(theirs)

    agree_full = mine_t & their_t
    agree_pair_diff_kind = (mine_p & their_p) - {(f, t) for (f, t, _) in agree_full}
    only_mine = mine_p - their_p
    only_theirs = their_p - mine_p

    flips = set()
    for f, t in only_mine:
        if (t, f) in only_theirs:
            flips.add(((f, t), (t, f)))

    return {
        "mine_total": len(mine),
        "theirs_total": len(theirs),
        "agree_full": sorted(agree_full),
        "agree_pair_diff_kind": sorted(
            (f, t, mine_k.get((f, t), []), their_k.get((f, t), []))
            for (f, t) in agree_pair_diff_kind
        ),
        "only_mine": sorted((f, t, mine_k[(f, t)]) for (f, t) in only_mine if (t, f) not in only_theirs),
        "only_theirs": sorted((f, t, their_k[(f, t)]) for (f, t) in only_theirs if (t, f) not in only_mine),
        "flips": sorted(flips),
    }


def diff_prereqs(mine: list[dict], theirs: list[dict]) -> dict:
    m = prereq_edges_to_set(mine)
    t = prereq_edges_to_set(theirs)

    agree = m & t
    only_mine = m - t
    only_theirs = t - m
    flips = {((f, x), (x, f)) for (f, x) in only_mine if (x, f) in only_theirs}

    only_mine_clean = sorted((f, x) for (f, x) in only_mine if (x, f) not in only_theirs)
    only_theirs_clean = sorted((f, x) for (f, x) in only_theirs if (x, f) not in only_mine)

    return {
        "mine_total": len(mine),
        "theirs_total": len(theirs),
        "agree": sorted(agree),
        "only_mine": only_mine_clean,
        "only_theirs": only_theirs_clean,
        "flips": sorted(flips),
    }


def render_md(c_diff: dict, p_diff: dict | None) -> str:
    out: list[str] = []
    out.append("# Concept graph & prereq DAG comparison\n")
    out.append("Comparison between our hand-authored graph and a blind second pass.\n")

    out.append("## Concept edges\n")
    mine = c_diff["mine_total"]
    theirs = c_diff["theirs_total"]
    af = len(c_diff["agree_full"])
    apk = len(c_diff["agree_pair_diff_kind"])
    om = len(c_diff["only_mine"])
    ot = len(c_diff["only_theirs"])
    fl = len(c_diff["flips"])
    out.append(f"- Mine: **{mine}** edges. Theirs: **{theirs}** edges.\n")
    out.append(f"- Full agreement (same `(from, to, kind)`): **{af}** ({100*af/max(mine,1):.0f}% of mine, {100*af/max(theirs,1):.0f}% of theirs)\n")
    out.append(f"- Same pair, different kind: **{apk}**\n")
    out.append(f"- Only mine: **{om}**\n")
    out.append(f"- Only theirs: **{ot}**\n")
    out.append(f"- Direction-flips (we both have a link but disagree on direction): **{fl}**\n")

    out.append("\n### Direction-flips (highest-signal disagreements)\n")
    if c_diff["flips"]:
        for (a, b) in c_diff["flips"]:
            out.append(f"- `{a[0]} → {a[1]}` (mine) vs `{b[0]} → {b[1]}` (theirs)\n")
    else:
        out.append("(none)\n")

    out.append("\n### Same pair, different kind\n")
    if c_diff["agree_pair_diff_kind"]:
        for (f, t, mk, tk) in c_diff["agree_pair_diff_kind"]:
            out.append(f"- `{f} → {t}` — mine: {mk}, theirs: {tk}\n")
    else:
        out.append("(none)\n")

    out.append("\n### Edges only in MINE (candidates for review — possible over-reach)\n")
    for (f, t, mk) in c_diff["only_mine"][:200]:
        out.append(f"- `{f} → {t}` [{','.join(mk)}]\n")
    if len(c_diff["only_mine"]) > 200:
        out.append(f"- … {len(c_diff['only_mine']) - 200} more\n")

    out.append("\n### Edges only in THEIRS (candidates I missed)\n")
    for (f, t, tk) in c_diff["only_theirs"][:200]:
        out.append(f"- `{f} → {t}` [{','.join(tk)}]\n")
    if len(c_diff["only_theirs"]) > 200:
        out.append(f"- … {len(c_diff['only_theirs']) - 200} more\n")

    if p_diff:
        out.append("\n---\n## Derived prereq DAG\n")
        pm = p_diff["mine_total"]
        pt = p_diff["theirs_total"]
        pa = len(p_diff["agree"])
        pom = len(p_diff["only_mine"])
        pot = len(p_diff["only_theirs"])
        pfl = len(p_diff["flips"])
        out.append(f"- Mine: **{pm}** edges. Theirs: **{pt}** edges.\n")
        out.append(f"- Agree (same `(from, to)`): **{pa}** ({100*pa/max(pm,1):.0f}% of mine, {100*pa/max(pt,1):.0f}% of theirs)\n")
        out.append(f"- Only mine: **{pom}**\n")
        out.append(f"- Only theirs: **{pot}**\n")
        out.append(f"- Direction-flips: **{pfl}**\n")

        out.append("\n### Prereq direction-flips\n")
        if p_diff["flips"]:
            for (a, b) in p_diff["flips"]:
                out.append(f"- `{a[0]} → {a[1]}` (mine) vs `{b[0]} → {b[1]}` (theirs)\n")
        else:
            out.append("(none)\n")

        out.append("\n### Prereqs only in MINE (sample)\n")
        for (f, t) in p_diff["only_mine"][:80]:
            out.append(f"- `{f} → {t}`\n")
        if len(p_diff["only_mine"]) > 80:
            out.append(f"- … {len(p_diff['only_mine']) - 80} more\n")

        out.append("\n### Prereqs only in THEIRS (sample)\n")
        for (f, t) in p_diff["only_theirs"][:80]:
            out.append(f"- `{f} → {t}`\n")
        if len(p_diff["only_theirs"]) > 80:
            out.append(f"- … {len(p_diff['only_theirs']) - 80} more\n")

    return "".join(out)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    other = Path(sys.argv[1]).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]

    mine_concept = load(repo_root / "vocab" / "concept_edges.json")["edges"]
    theirs_concept_path = other / "concept_edges.json"
    if not theirs_concept_path.exists():
        # try step1/ or step2/ subdirs
        if (other / "step2" / "concept_edges.json").exists():
            theirs_concept_path = other / "step2" / "concept_edges.json"
        elif (other / "step1" / "concept_edges.json").exists():
            theirs_concept_path = other / "step1" / "concept_edges.json"
        else:
            print(f"[ERROR] no concept_edges.json found under {other}", file=sys.stderr)
            return 1
    theirs_concept = load(theirs_concept_path)["edges"]

    c_diff = diff_concept(mine_concept, theirs_concept)

    p_diff = None
    mine_prereqs = load(repo_root / "vocab" / "prereqs.json")["edges"]
    theirs_prereqs_path = theirs_concept_path.parent / "prereqs.json"
    if theirs_prereqs_path.exists():
        theirs_prereqs = load(theirs_prereqs_path)["edges"]
        p_diff = diff_prereqs(mine_prereqs, theirs_prereqs)
    else:
        print(f"[note] {theirs_prereqs_path} missing — skipping prereq comparison")

    report = render_md(c_diff, p_diff)
    out_path = repo_root / "COMPARE_REPORT.md"
    out_path.write_text(report)

    print(f"compared against: {theirs_concept_path}")
    print(f"  full agreement on concept edges: {len(c_diff['agree_full'])} / mine={c_diff['mine_total']} / theirs={c_diff['theirs_total']}")
    print(f"  concept direction-flips: {len(c_diff['flips'])}")
    if p_diff:
        print(f"  prereq agreement: {len(p_diff['agree'])} / mine={p_diff['mine_total']} / theirs={p_diff['theirs_total']}")
        print(f"  prereq direction-flips: {len(p_diff['flips'])}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
