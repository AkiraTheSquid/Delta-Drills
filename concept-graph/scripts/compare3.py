"""3-way diff over concept_edges.json from three sources.

Usage:
    python compare3.py <name1>=<path1> <name2>=<path2> <name3>=<path3>

Example:
    python compare3.py mine=/path/to/concept_edges.json opus=/path/to/agent_opus/concept_edges.json codex=/path/to/agent_codex/concept_edges.json

Outputs a Markdown report to concept-graph/COMPARE3_REPORT.md with:
  - Per-source edge counts and kind distributions
  - Pairwise agreement matrix (full-triple agreement; pair-only agreement; direction-flips)
  - Three-way agreement: edges all three agree on (gold consensus)
  - Two-of-three agreement: edges with 2/3 support (likely-real)
  - One-only edges: per-source idiosyncratic edges (likely over-reach or genuine miss elsewhere)
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_edges(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)["edges"]


def index(edges: list[dict]) -> dict:
    triples = set()  # (from, to, kind)
    pairs = set()    # (from, to)
    by_pair_kind: dict[tuple[str, str], list[str]] = defaultdict(list)
    kind_counts: dict[str, int] = defaultdict(int)
    for e in edges:
        f, t, k = e["from"], e["to"], e["kind"]
        triples.add((f, t, k))
        pairs.add((f, t))
        by_pair_kind[(f, t)].append(k)
        kind_counts[k] += 1
    return {
        "edges": edges,
        "triples": triples,
        "pairs": pairs,
        "by_pair_kind": dict(by_pair_kind),
        "kind_counts": dict(kind_counts),
    }


def pairwise(a: dict, b: dict) -> dict:
    full = a["triples"] & b["triples"]
    pair_full = {(f, t) for (f, t, _) in full}
    pair_only_diff_kind = (a["pairs"] & b["pairs"]) - pair_full
    only_a = a["pairs"] - b["pairs"]
    only_b = b["pairs"] - a["pairs"]
    flips = {((f, t), (t, f)) for (f, t) in only_a if (t, f) in only_b}
    only_a_clean = sorted((f, t) for (f, t) in only_a if (t, f) not in only_b)
    only_b_clean = sorted((f, t) for (f, t) in only_b if (t, f) not in only_a)
    return {
        "agree_full": len(full),
        "agree_pair_diff_kind": len(pair_only_diff_kind),
        "only_a": len(only_a_clean),
        "only_b": len(only_b_clean),
        "flips": len(flips),
        "flips_list": sorted(flips),
    }


def three_way(srcs: list[tuple[str, dict]]) -> dict:
    pair_sets = [s["pairs"] for _, s in srcs]
    triple_sets = [s["triples"] for _, s in srcs]

    all3_pairs = pair_sets[0] & pair_sets[1] & pair_sets[2]
    all3_triples = triple_sets[0] & triple_sets[1] & triple_sets[2]

    twothree_pairs: set = set()
    for combo in itertools.combinations(range(3), 2):
        twothree_pairs |= (pair_sets[combo[0]] & pair_sets[combo[1]])
    only_2of3 = twothree_pairs - all3_pairs

    only_one = {}
    for i, (name, s) in enumerate(srcs):
        others = pair_sets[:i] + pair_sets[i+1:]
        union_others = others[0] | others[1]
        only_one[name] = s["pairs"] - union_others

    return {
        "all3_pairs": all3_pairs,
        "all3_triples": all3_triples,
        "only_2of3_pairs": only_2of3,
        "only_one_pairs": only_one,
    }


def render(srcs: list[tuple[str, dict]], three: dict) -> str:
    out: list[str] = []
    out.append("# 3-way concept-edges comparison\n\n")
    out.append("## Per-source summary\n\n")
    out.append("| source | total | is-a | refines | uses | part-of | alternative-to |\n")
    out.append("|---|--:|--:|--:|--:|--:|--:|\n")
    for name, s in srcs:
        kc = s["kind_counts"]
        out.append(f"| {name} | {len(s['edges'])} | {kc.get('is-a', 0)} | {kc.get('refines', 0)} | {kc.get('uses', 0)} | {kc.get('part-of', 0)} | {kc.get('alternative-to', 0)} |\n")

    out.append("\n## Pairwise agreement\n\n")
    out.append("| pair | full-triple agree | pair-only (diff kind) | only A | only B | direction-flips |\n")
    out.append("|---|--:|--:|--:|--:|--:|\n")
    for (na, sa), (nb, sb) in itertools.combinations(srcs, 2):
        pw = pairwise(sa, sb)
        out.append(f"| {na} ↔ {nb} | {pw['agree_full']} | {pw['agree_pair_diff_kind']} | {pw['only_a']} | {pw['only_b']} | {pw['flips']} |\n")

    out.append("\n## 3-way overlap\n\n")
    out.append(f"- **Edges all 3 agree on (same `(from, to)`):** {len(three['all3_pairs'])}\n")
    out.append(f"- **Edges all 3 agree on AND same `kind`:** {len(three['all3_triples'])}\n")
    out.append(f"- **Edges with 2-of-3 support:** {len(three['only_2of3_pairs'])}\n")
    for name, s in srcs:
        out.append(f"- **Only in {name}:** {len(three['only_one_pairs'][name])}\n")

    out.append("\n## All-3 consensus edges (same pair, same kind)\n")
    out.append("These are the high-confidence relations — promote to `accepted` without review.\n\n")
    for (f, t, k) in sorted(three["all3_triples"]):
        out.append(f"- `{f}` → `{t}` [{k}]\n")

    out.append("\n## Pair-agreement but kind disagreement\n\n")
    kind_map: dict[tuple[str, str], dict[str, str]] = {}
    for name, s in srcs:
        for (f, t), kinds in s["by_pair_kind"].items():
            kind_map.setdefault((f, t), {})[name] = ",".join(sorted(set(kinds)))
    for (f, t) in sorted(three["all3_pairs"]):
        kinds_per_source = kind_map[(f, t)]
        if len(set(kinds_per_source.values())) > 1:
            out.append(f"- `{f}` → `{t}` — " + ", ".join(f"{n}={k}" for n, k in kinds_per_source.items()) + "\n")

    out.append("\n## Pairwise direction-flips (both have the link, disagree on direction)\n\n")
    for (na, sa), (nb, sb) in itertools.combinations(srcs, 2):
        pw = pairwise(sa, sb)
        if pw["flips_list"]:
            out.append(f"### {na} ↔ {nb} flips ({len(pw['flips_list'])})\n\n")
            for (a_dir, b_dir) in pw["flips_list"][:50]:
                out.append(f"- `{a_dir[0]} → {a_dir[1]}` ({na}) vs `{b_dir[0]} → {b_dir[1]}` ({nb})\n")
            if len(pw["flips_list"]) > 50:
                out.append(f"- … {len(pw['flips_list']) - 50} more\n")
            out.append("\n")

    return "".join(out)


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2

    srcs: list[tuple[str, dict]] = []
    for arg in sys.argv[1:]:
        if "=" not in arg:
            print(f"bad arg: {arg}", file=sys.stderr)
            return 2
        name, path = arg.split("=", 1)
        srcs.append((name, index(load_edges(Path(path).expanduser().resolve()))))

    three = three_way(srcs)
    out = render(srcs, three)
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "COMPARE3_REPORT.md"
    out_path.write_text(out)

    print(f"wrote {out_path}")
    for name, s in srcs:
        print(f"  {name}: {len(s['edges'])} edges")
    print(f"  all-3 same pair: {len(three['all3_pairs'])}")
    print(f"  all-3 same pair+kind: {len(three['all3_triples'])}")
    print(f"  2-of-3 only: {len(three['only_2of3_pairs'])}")
    for name in (n for n, _ in srcs):
        print(f"  only-{name}: {len(three['only_one_pairs'][name])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
