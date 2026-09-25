"""Run a generated problem's solution over its inputs and return the outputs.

Generated problems (gen_missing.py) carry inputs but no outputs: the expected
output of every case is what the reference solution returns, cross-checked
against an independent brute-force solution. This module is that runner.

Shapes:
- "function": `inputs` is a list of positional-arg lists for `entry_point`.
  `arg_types` / `return_type` name "list_node" or "tree_node" where the
  function takes or returns a linked list / binary tree; cases stay plain
  JSON lists (LeetCode's serialisation) and are converted at the boundary.
- "design": each case is {"ops": [...], "args": [[...], ...]} in LeetCode's
  format; ops[0] constructs `entry_point`, the rest are method calls; the
  output is the list of return values (None for the constructor).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HELPERS = '''
import random, functools, collections, string, math, datetime, heapq, bisect
from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from math import *

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def list_node(values):
    head = cur = None
    for v in values:
        node = ListNode(v)
        if head is None:
            head = cur = node
        else:
            cur.next = node
            cur = node
    return head

def tree_node(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue, i = deque([root]), 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i]); queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i]); queue.append(node.right)
        i += 1
    return root

def list_node_cycle(spec):
    """[values, pos]: LeetCode's cycle encoding; pos = index the tail links to, -1 = none."""
    values, pos = spec
    head = list_node(values)
    if head is not None and pos >= 0:
        tail, target, i, cur = None, None, 0, head
        while cur is not None:
            if i == pos:
                target = cur
            tail, cur, i = cur, cur.next, i + 1
        tail.next = target
    return head

def _from_list_node(node):
    out = []
    while node is not None and len(out) < 100000:
        out.append(node.val); node = node.next
    return out

def _from_tree_node(root):
    out, queue = [], deque([root])
    while queue:
        node = queue.popleft()
        if node is None:
            out.append(None); continue
        out.append(node.val); queue.append(node.left); queue.append(node.right)
    while out and out[-1] is None:
        out.pop()
    return out
'''

DRIVER = '''
import json as _json, sys as _sys
_spec = _json.load(open(_sys.argv[1]))
_IN = {"list_node": list_node, "tree_node": tree_node, "list_node_cycle": list_node_cycle}
_OUT = {"list_node": _from_list_node, "tree_node": _from_tree_node}
def _plain(x):
    if isinstance(x, tuple):
        return [_plain(v) for v in x]
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, (ListNode, TreeNode)):
        return _from_list_node(x) if isinstance(x, ListNode) else _from_tree_node(x)
    return x
_res = []
for _case in _spec["inputs"]:
    if _spec["shape"] == "design":
        _obj, _outs = None, []
        for _op, _a in zip(_case["ops"], _case["args"]):
            if _obj is None:
                _obj = globals()[_op](*_a); _outs.append(None)
            else:
                _outs.append(_plain(getattr(_obj, _op)(*_a)))
        _res.append(_outs)
    else:
        _types = _spec.get("arg_types") or []
        _args = [(_IN[_types[i]](a) if i < len(_types) and _types[i] in _IN else a) for i, a in enumerate(_case)]
        _got = eval(_spec["entry_point"])(*_args)
        _rt = _spec.get("return_type")
        _res.append(_OUT[_rt](_got) if _rt in _OUT else _plain(_got))
print("\\n@@OUT@@" + _json.dumps(_res))
'''

TIMEOUT = int(os.environ.get("HARNESS_TIMEOUT", "30"))
MIN_CASES = 10


def sandbox_env(home: str) -> dict:
    """Environment for running untrusted solution code: no inherited secrets
    (API keys, tokens), HOME pointed at a throwaway dir. Not a security boundary
    (no network or filesystem isolation); it keeps keys out of reach of dataset
    and model-written code, which is the realistic leak in this offline pipeline."""
    return {"PATH": "/usr/bin:/bin", "HOME": home, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}


def run(problem: dict, code: str) -> tuple[list | None, str]:
    """(outputs, error). `code` is the solution source to run."""
    spec = {k: problem.get(k) for k in ("inputs", "shape", "entry_point", "arg_types", "return_type")}
    with tempfile.TemporaryDirectory() as tmp:
        prog, data = os.path.join(tmp, "p.py"), os.path.join(tmp, "spec.json")
        open(prog, "w").write(HELPERS + "\n" + code + "\n" + DRIVER)
        json.dump(spec, open(data, "w"))
        try:
            res = subprocess.run(["nice", sys.executable, "-I", prog, data], capture_output=True, text=True,
                                 timeout=TIMEOUT, cwd=tmp, env=sandbox_env(tmp))
        except subprocess.TimeoutExpired:
            return None, "timeout"
    if res.returncode != 0 or "@@OUT@@" not in res.stdout:
        return None, (res.stderr.strip().splitlines() or ["no output"])[-1][:300]
    return json.loads(res.stdout.rsplit("@@OUT@@", 1)[1]), ""


def same(a, b, compare: str) -> bool:
    """Compare two outputs under the problem's declared comparison."""
    if compare in ("unordered", "unordered_deep") and isinstance(a, list) and isinstance(b, list):
        deep = compare == "unordered_deep"
        key = lambda v: json.dumps(sorted(v, key=json.dumps) if deep and isinstance(v, list) else v, sort_keys=True)
        return sorted(map(key, a)) == sorted(map(key, b))
    if compare == "float":
        try:
            return abs(float(a) - float(b)) <= 1e-5 * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            return a == b
    return a == b
