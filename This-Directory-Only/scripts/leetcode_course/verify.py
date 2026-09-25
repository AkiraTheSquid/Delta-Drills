"""Run every pool problem's reference solution against its own tests.

A problem survives only if its solution passes, so no learner is ever graded
against a broken case. LeetCode rows run the dataset's `check(candidate)`
asserts; Codewars rows call `fn_name(*inputs[i])` and compare to
`outputs[i]`. APPS-style outputs are sometimes wrapped in a one-element list,
so both readings are tried and the one that passes is recorded as `unwrap`.

Each problem runs in its own `python3` subprocess with a timeout.
Writes RAW_DIR/verified.jsonl: {id, ok, unwrap?, error?}.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from build_pool import RAW_DIR
from harness import sandbox_env

TIMEOUT = int(os.environ.get("VERIFY_TIMEOUT", "20"))
WORKERS = int(os.environ.get("VERIFY_WORKERS", "6"))

CW_HARNESS = """
import json as _json
_cases = _json.loads({cases!r})
def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            return False
    if isinstance(a, tuple):
        a = list(a)
    return a == b
_fn = {fn}
_unwrap = None
for _inp, _out in zip(_cases["inputs"], _cases["outputs"]):
    _got = _fn(*_inp)
    _ok_raw = _same(_got, _out)
    _ok_unw = isinstance(_out, list) and len(_out) == 1 and _same(_got, _out[0])
    if _unwrap is None:
        _unwrap = (not _ok_raw) and _ok_unw
    if not (_ok_unw if _unwrap else _ok_raw):
        raise AssertionError(f"case {{_inp!r}}: got {{_got!r}} want {{_out!r}}")
print("UNWRAP" if _unwrap else "RAW")
"""


def program(p: dict) -> str:
    if p["tests"]["kind"] == "assert":
        return "\n".join([p["preamble"], p["solution"], p["tests"]["check"], f"check({p['entry_point']})", "print('RAW')"])
    cases = json.dumps({"inputs": p["tests"]["inputs"], "outputs": p["tests"]["outputs"]})
    return p["solution"] + "\n" + CW_HARNESS.format(cases=cases, fn=p["entry_point"])


def run(p: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.py")
        with open(path, "w") as fh:
            fh.write(program(p))
        try:
            res = subprocess.run(["nice", sys.executable, "-I", path], capture_output=True, text=True,
                                 timeout=TIMEOUT, cwd=tmp, env=sandbox_env(tmp))
        except subprocess.TimeoutExpired:
            return {"id": p["id"], "ok": False, "error": "timeout"}
    if res.returncode != 0:
        tail = (res.stderr.strip().splitlines() or ["?"])[-1][:200]
        return {"id": p["id"], "ok": False, "error": tail}
    return {"id": p["id"], "ok": True, "unwrap": res.stdout.strip().endswith("UNWRAP")}


def main() -> None:
    pool = [json.loads(line) for line in open(RAW_DIR / "pool.jsonl")]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(run, pool))
    with open(RAW_DIR / "verified.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    tiers = {p["id"]: (p["source"], p["tier"]) for p in pool}
    summary: dict = {}
    for r in results:
        key = tiers[r["id"]] + (r["ok"],)
        summary[key] = summary.get(key, 0) + 1
    print(json.dumps({"/".join(map(str, k)): v for k, v in sorted(summary.items())}, indent=1))


if __name__ == "__main__":
    main()
