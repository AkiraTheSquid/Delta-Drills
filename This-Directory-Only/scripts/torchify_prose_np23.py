#!/usr/bin/env python3
"""Mechanical half of the np-2/np-3 prose conversion.

Renames ONLY the numpy symbols that torch spells identically, outside code
fences and outside the frontmatter's KC keys (`kc:`, `supporting:` and friends
are registry identifiers — renaming them would unhook the page from the graph).

Everything torch spells differently, or does differently, is deliberately NOT
in this table: those sentences make claims that have to be rewritten by hand,
not renamed.  They are printed at the end as the hand-review list.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from torchify_lessons_np23 import FENCE, pages  # noqa: E402

# Verified present in torch 2.12 under the same name, same meaning.
SAME_NAME = [
    "where", "unique", "diff", "tile", "argsort", "argmin", "argmax", "argwhere",
    "count_nonzero", "bincount", "isnan", "isinf", "isfinite", "nan_to_num",
    "nanmean", "nansum", "tril", "triu", "zeros_like", "ones_like", "abs",
    "sign", "ceil", "floor", "round", "exp", "log", "sqrt", "maximum",
    "minimum", "cumsum", "cumprod", "sort", "column_stack", "vstack", "hstack",
    "ravel", "trace", "clip", "empty", "eye", "arange", "ones", "zeros",
    "stack", "diag", "diagflat", "result_type", "quantile", "isin", "outer",
    "dot", "einsum", "flatten", "squeeze", "cat", "median", "topk", "kthvalue",
    "cummax", "cummin", "promote_types", "randperm",
]

# Spelled differently but unambiguous in prose.
RENAMES = [
    (r"\bnp\.array\(", "t.tensor("),
    (r"`np\.array`", "`t.tensor`"),
    (r"\bnp\.linalg\.", "t.linalg."),
    (r"\bnp\.nan\b", "t.nan"),
    (r"\bnp\.inf\b", "t.inf"),
    (r"\bnp\.int64\b", "t.int64"),
    (r"\bnp\.float64\b", "t.float64"),
    (r"\bnp\.float32\b", "t.float32"),
]


def convert(text: str) -> str:
    for name in SAME_NAME:
        text = re.sub(rf"\bnp\.{name}\b", f"t.{name}", text)
    for pat, rep in RENAMES:
        text = re.sub(pat, rep, text)
    return text


def split_frontmatter(src: str):
    if not src.startswith("---\n"):
        return "", src
    end = src.find("\n---\n", 4)
    return (src[:end + 5], src[end + 5:]) if end != -1 else ("", src)


def main() -> int:
    changed = 0
    for path in pages():
        src = path.read_text()
        front, body = split_frontmatter(src)
        # In the frontmatter only `title:` is prose; the rest are KC ids.
        front = re.sub(r"^(title:.*)$", lambda m: convert(m.group(1)), front, flags=re.M)

        out, last = [], 0
        for m in FENCE.finditer(body):
            out.append(convert(body[last:m.start()]))
            out.append(m.group(0))          # fences already translated
            last = m.end()
        out.append(convert(body[last:]))
        new = front + "".join(out)

        if new != src:
            path.write_text(new)
            changed += 1
    print(f"{changed} page(s) had prose symbols renamed")

    print("\nHAND-REVIEW — prose still naming numpy or a numpy-only API:")
    for path in pages():
        body = FENCE.sub("", split_frontmatter(path.read_text())[1])
        hits = [ln.strip() for ln in body.splitlines()
                if re.search(r"\bnumpy\b|\bNumPy\b|\bnp\.|\bndarray\b", ln)]
        if hits:
            print(f"\n{path.name} ({len(hits)}):")
            for h in hits:
                print(f"   {h[:170]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
