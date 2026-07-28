#!/usr/bin/env python3
"""Translate the code fences of the einops + einsum KP pages to PyTorch.

Only fenced code is touched — prose is left exactly as written, because the
prose makes claims (about dtypes, about what NumPy does differently) that a
regex has no business rewriting.  Those get read and edited by hand.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from torchify_einops_einsum import translate  # noqa: E402

REPO = Path("/home/stellar-thread/Applications/Delta-Drills-Local")
DIRS = [REPO / "Local_Deployed_Shared/lessons/einops",
        REPO / "Local_Deployed_Shared/lessons/einsum"]

FENCE = re.compile(r"(```python[^\n]*\n)(.*?)(```)", re.S)

changed = 0
for d in DIRS:
    for path in sorted(d.glob("kp-*.md")):
        src = path.read_text()
        out = FENCE.sub(lambda m: m.group(1) + translate(m.group(2)) + m.group(3), src)
        if out != src:
            path.write_text(out)
            changed += 1
            print(f"  fences translated: {path.relative_to(REPO)}")

print(f"{changed} file(s) changed")

# Report what prose still says, so nothing gets converted silently.
print("\nprose still mentioning numpy (hand-review):")
for d in DIRS:
    for path in sorted(d.glob("kp-*.md")):
        body = FENCE.sub("", path.read_text())
        hits = [ln.strip() for ln in body.splitlines()
                if re.search(r"\bnumpy\b|\bNumPy\b|\bnp\.", ln)]
        if hits:
            print(f"\n{path.relative_to(REPO)} ({len(hits)}):")
            for h in hits:
                print(f"   {h[:150]}")
