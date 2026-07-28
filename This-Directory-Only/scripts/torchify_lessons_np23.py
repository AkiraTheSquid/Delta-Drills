#!/usr/bin/env python3
"""Translate the code fences of the np-2 + np-3 KP pages to PyTorch.

Only fenced code is touched.  Prose is left exactly as written and reported at
the end for hand review, because the prose makes claims — about dtypes, about
what numpy does differently, about views vs copies — that a regex has no
business rewriting.

The page set is read from the KC registry rather than hardcoded, so a KC moving
between lessons cannot silently leave a page behind in the other dialect.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from torchify_np23 import translate  # noqa: E402

REPO = Path("/home/stellar-thread/Applications/Delta-Drills-Local")
LESSON_DIR = REPO / "Local_Deployed_Shared/lessons/numpy"
REGISTRY = REPO / "Local_Deployed_Shared/lessons/kc_registry.json"
LESSONS = {"np-2", "np-3"}

FENCE = re.compile(r"(```python[^\n]*\n)(.*?)(```)", re.S)


def pages() -> list[Path]:
    registry = json.loads(REGISTRY.read_text())
    names = [kc["id"].split(".", 1)[1] for kc in registry["kcs"]
             if kc["lesson"] in LESSONS]
    found, missing = [], []
    for name in sorted(names):
        path = LESSON_DIR / f"kp-{name}.md"
        (found if path.exists() else missing).append(path)
    if missing:
        print(f"WARNING: {len(missing)} KC(s) have no page: "
              f"{[p.name for p in missing]}")
    return found


def main() -> int:
    targets = pages()
    print(f"{len(targets)} page(s) in {sorted(LESSONS)}")
    changed = 0
    for path in targets:
        src = path.read_text()
        out = FENCE.sub(lambda m: m.group(1) + translate(m.group(2)) + m.group(3), src)
        if out != src:
            path.write_text(out)
            changed += 1
            print(f"  fences translated: {path.name}")
    print(f"{changed} file(s) changed")

    print("\nprose still mentioning numpy (hand-review):")
    for path in targets:
        body = FENCE.sub("", path.read_text())
        hits = [ln.strip() for ln in body.splitlines()
                if re.search(r"\bnumpy\b|\bNumPy\b|\bnp\.|\bndarray\b", ln)]
        if hits:
            print(f"\n{path.name} ({len(hits)}):")
            for h in hits:
                print(f"   {h[:160]}")

    # A fence that binds `t` shadows the torch alias for every fence below it.
    print("\nfences binding the name `t` (would shadow the torch alias):")
    for path in targets:
        for m in FENCE.finditer(path.read_text()):
            for ln in m.group(2).splitlines():
                if re.match(r"\s*t\s*=|.*\bdef \w+\([^)]*\bt\b", ln):
                    print(f"   {path.name}: {ln.strip()[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
