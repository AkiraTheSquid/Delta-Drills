"""Generate statement, solutions and test inputs for core problems that have none.

Two gaps in the core bank have no dataset row:
- core LeetCode problems absent from LeetCodeDataset (mostly design problems);
- Striver A2Z problems hosted on GfG / TUF (title + link only), skipping the
  "Beginner Problems" step and anything flagged Top 100 Liked.

Codex (`codex exec`, not Claude) writes each problem's statement, starter code,
an optimal reference solution, an independent brute-force solution and ~25
test inputs. It never writes outputs: harness.py runs BOTH solutions on every
input, and a problem is kept only when they agree on every case. The agreed
outputs become the expected answers.

Writes RAW_DIR/generated_raw.jsonl (codex output, append-only, keyed by
`key` so a rerun skips done problems) and RAW_DIR/generated.jsonl (kept rows).
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import harness
from build_pool import RAW_DIR, load_dataset, sheet_tiers

BATCH = int(os.environ.get("GEN_BATCH", "6"))
WORKERS = int(os.environ.get("GEN_WORKERS", "2"))  # max two model processes
MODEL = os.environ.get("GEN_MODEL", "gpt-5.6-sol")
EFFORT = os.environ.get("GEN_EFFORT", "high")
RAW_OUT = RAW_DIR / "generated_raw.jsonl"

PROMPT = """You are authoring coding-interview practice problems for an app that runs
Python solutions against test inputs. For EACH problem below, produce one JSON
object. Output ONLY a JSON array of these objects, no prose, no code fences.

Fields:
- "key": copy the given key exactly.
- "title": the problem's usual name.
- "statement": the full problem statement in Markdown as the platform states it:
  task, 2-3 worked examples (Input / Output / Explanation), and Constraints.
  Write it for a Python function with the signature in starter_code.
- "shape": "function" for a single function, "design" for a class with methods
  (LeetCode design problems, iterators, caches).
- "starter_code": a Python stub with the exact signature and `pass` body. Plain
  top-level function for "function" (NOT a `class Solution`); the class with
  __init__ and every method for "design".
- "entry_point": the function name, or the class name for "design".
- "arg_types": for "function", a list naming each argument's type ONLY where it is
  "list_node" (linked list head), "list_node_cycle" (linked list that may contain
  a cycle, passed as [values, pos] where pos is the index the tail links back to,
  -1 for none) or "tree_node" (binary tree root); use "plain"
  otherwise. Classes ListNode(val, next) and TreeNode(val, left, right) already
  exist; tests pass linked lists / trees as LeetCode-style plain lists.
- "return_type": "list_node", "tree_node" or "plain".
- "compare": "exact", "unordered" (any order of the returned list's elements is
  accepted; each element's own contents must match exactly), "unordered_deep"
  (the inner lists' order does not matter either, e.g. 3Sum triplets or anagram
  groups) or "float".
- "solution": a correct, efficient reference solution (full code, same signature).
- "brute_force": an INDEPENDENT, obviously-correct slow solution (different
  approach, same signature). It is run on every input to cross-check "solution".
- "inputs": 25 test inputs covering the examples, edge cases (empty, single
  element, duplicates, negatives, boundaries) and a few medium cases. Keep every
  input small enough for brute_force to finish within a second. "function":
  each input is a list of positional arguments. "design": each input is
  {{"ops": ["ClassName", "method", ...], "args": [[ctor args], [method args], ...]}}.
  Do NOT include expected outputs.
- If the problem is not well defined enough to test automatically (e.g. it
  prints patterns, needs interactive I/O, or has many valid answers that
  "unordered" can't capture), return {{"key": ..., "skip": "<reason>"}} instead.

Use only the Python standard library. No input(), no print-based answers.

Problems:
{items}
"""


def tasks() -> list[dict]:
    ds = load_dataset()
    core, top = sheet_tiers()
    out = []
    for r in csv.DictReader(open(RAW_DIR / "neetcode_250.csv")):
        lc = r["LC #"].strip()
        if lc and lc in core and lc not in ds:
            out.append({"key": f"lc{lc}", "title": r["Title"], "platform": "LeetCode", "url": r["LeetCode URL"],
                        "topic": r["Topic"]})
    have = {t["key"] for t in out}
    for r in csv.DictReader(open(RAW_DIR / "striver_a2z_dsa_sheet.csv")):
        lc = r["LC #"].strip()
        if r["In Top 100 Liked"] == "Yes":
            continue
        if lc and lc in core and lc not in ds and f"lc{lc}" not in have:
            out.append({"key": f"lc{lc}", "title": r["Title"], "platform": "LeetCode", "url": r["Link"], "topic": r["Subtopic"]})
            have.add(f"lc{lc}")
        elif not lc and r["Step"] != "Beginner Problems":
            slug = re.sub(r"[^a-z0-9]+", "-", r["Title"].lower()).strip("-")
            out.append({"key": f"st{r['#']}-{slug}"[:60], "title": r["Title"], "platform": r["Platform"], "url": r["Link"],
                        "topic": f"{r['Step']} / {r['Subtopic']}"})
    return out


def ask_codex(batch: list[dict]) -> list[dict]:
    items = "\n".join(json.dumps(t) for t in batch)
    with tempfile.TemporaryDirectory() as tmp:
        last = os.path.join(tmp, "last.txt")
        cmd = ["nice", "codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "-C", tmp,
               "-m", MODEL, "-c", f"model_reasoning_effort={EFFORT}", "-o", last, "-"]
        res = subprocess.run(cmd, input=PROMPT.format(items=items), capture_output=True, text=True, timeout=1800)
        text = open(last).read() if os.path.exists(last) else ""
    m = re.search(r"\[.*\]", text, re.S)
    if res.returncode != 0 or not m:
        print(f"codex batch failed ({res.returncode}): {res.stderr[-300:]}", file=sys.stderr)
        return []
    try:
        rows = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        print(f"codex batch unparseable: {exc}", file=sys.stderr)
        return []
    with open(RAW_OUT, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


def check(row: dict) -> dict:
    """Run solution and brute force; keep only if they agree on every input."""
    if row.get("skip"):
        return {"key": row["key"], "ok": False, "error": f"skip: {row['skip']}"}
    cases = row.get("inputs") or []
    if len(cases) < harness.MIN_CASES:
        return {"key": row["key"], "ok": False, "error": f"only {len(cases)} inputs"}
    if row.get("shape") == "design" and any(len(c.get("ops", [])) != len(c.get("args", [])) for c in cases):
        return {"key": row["key"], "ok": False, "error": "design case with ops/args length mismatch"}
    got, err = harness.run(row, row["solution"])
    if got is None:
        return {"key": row["key"], "ok": False, "error": f"solution: {err}"}
    ref, err = harness.run(row, row["brute_force"])
    if ref is None:
        return {"key": row["key"], "ok": False, "error": f"brute_force: {err}"}
    bad = [i for i, (a, b) in enumerate(zip(got, ref)) if not harness.same(a, b, row.get("compare", "exact"))]
    if bad or len(got) != len(ref):
        return {"key": row["key"], "ok": False, "error": f"disagree on cases {bad[:5]}"}
    return {**row, "ok": True, "outputs": got}


def main() -> None:
    todo = tasks()
    done = {}
    if RAW_OUT.exists():
        for line in RAW_OUT.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    pending = [t for t in todo if t["key"] not in done]
    print(f"{len(todo)} problems to generate, {len(pending)} not yet asked", file=sys.stderr)
    if "--dry" in sys.argv:
        for t in todo:
            print(t["key"], "|", t["title"], "|", t["topic"])
        return
    batches = [pending[i:i + BATCH] for i in range(0, len(pending), BATCH)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for rows in ex.map(ask_codex, batches):
            for r in rows:
                done[r["key"]] = r
    meta = {t["key"]: t for t in todo}
    results = [check(done[k]) | {"meta": meta[k]} for k in meta if k in done]
    with open(RAW_DIR / "generated.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    kept = sum(r["ok"] for r in results)
    print(f"kept {kept}/{len(results)} (asked {len(done)}/{len(todo)})")
    for r in results:
        if not r["ok"]:
            print(" ", r["key"], r["error"][:120])


if __name__ == "__main__":
    main()
