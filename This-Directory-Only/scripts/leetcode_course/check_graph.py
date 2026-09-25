"""Structural checks for leetcode_course/patterns.json. Exit 1 on any finding.

- ids unique and `leetcode.`-prefixed; every prereq resolves; exactly one root;
- the prereq graph is acyclic and every node is reachable from the root;
- `encompassing` keys are a subset of `prereqs`, weights in (0, 1];
- every non-root node encompasses at least one prerequisite (FIRe credit has
  somewhere to flow);
- a prereq that is ALSO reachable through another prereq is redundant for
  gating, so it is allowed only when it carries an encompassing weight (the
  direct edge is how implicit credit reaches a non-parent ancestor).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GRAPH = Path(__file__).resolve().parents[2] / "leetcode_course" / "patterns.json"


def ancestors(kc: str, parents: dict[str, list[str]]) -> set[str]:
    seen, stack = set(), list(parents[kc])
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack.extend(parents[p])
    return seen


def findings(kcs: list[dict]) -> list[str]:
    out = []
    ids = [k["id"] for k in kcs]
    if len(set(ids)) != len(ids):
        out.append("duplicate ids")
    parents = {k["id"]: k.get("prereqs", []) for k in kcs}
    for k in kcs:
        kid, pre, enc = k["id"], k.get("prereqs", []), k.get("encompassing", {})
        if not kid.startswith("leetcode."):
            out.append(f"{kid}: id must start with leetcode.")
        for p in pre:
            if p not in parents:
                out.append(f"{kid}: unknown prereq {p}")
        for p, w in enc.items():
            if p not in pre:
                out.append(f"{kid}: encompasses {p} which is not a prereq")
            if not 0 < w <= 1:
                out.append(f"{kid}: encompassing weight {w} outside (0, 1]")
        if pre and not enc:
            out.append(f"{kid}: no encompassing edge")
        if not k.get("describe"):
            out.append(f"{kid}: missing describe")
    if out:
        return out
    roots = [k for k, p in parents.items() if not p]
    if len(roots) != 1:
        out.append(f"expected one root, found {roots}")
    for kid in parents:
        if kid in ancestors(kid, parents):
            out.append(f"{kid}: cycle")
    if out:
        return out
    for k in kcs:
        pre, enc = k.get("prereqs", []), k.get("encompassing", {})
        for p in pre:
            via_other = any(p in ancestors(q, parents) for q in pre if q != p)
            if via_other and p not in enc:
                out.append(f"{k['id']}: prereq {p} is redundant (reachable via another prereq) and carries no encompassing weight")
        if roots and k["id"] != roots[0] and roots[0] not in ancestors(k["id"], parents):
            out.append(f"{k['id']}: not reachable from root {roots[0]}")
    return out


def main() -> int:
    kcs = json.loads(GRAPH.read_text())["kcs"]
    bad = findings(kcs)
    for f in bad:
        print(f)
    print(f"{len(kcs)} concepts, {sum(len(k.get('prereqs', [])) for k in kcs)} prereq edges, "
          f"{sum(len(k.get('encompassing', {})) for k in kcs)} encompassing edges; {len(bad)} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
