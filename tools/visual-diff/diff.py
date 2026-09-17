#!/usr/bin/env python3
"""
DIFF — two measured designs in, one ranked list of differences out.

    ./diff.py mockup ours            # how far our page is from the reference
    ./diff.py lw mockup              # is the reference itself faithful?
    ./diff.py mockup ours --json

Every comparison is between the SAME ROLE on both sides — a paragraph against a
paragraph, a ToC label against a ToC label — which is what the role vocabulary
in targets.json exists to make possible.

🔴 RANKED, NOT ALPHABETICAL. A 2px difference in body text is a different design;
a 2px difference in a bottom margin is nothing. Each field carries a weight, the
report multiplies it by how far off the value is, and what comes out at the top
of the list is what a person would notice first standing back from the screen.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# field -> (weight, tolerance for "same", tolerance for "close")
NUMERIC = {
    "font.size": (3.0, 0.5, 2.0),
    "font.lineHeight": (2.0, 0.5, 2.0),
    "font.ratio": (2.5, 0.03, 0.12),
    "font.letterSpacing": (1.0, 0.1, 0.5),
    "column.width": (3.0, 4.0, 24.0),
    "column.left": (2.0, 4.0, 24.0),
    "box.w": (1.5, 3.0, 16.0),
    "box.x": (1.5, 3.0, 16.0),
    "box.h": (0.8, 3.0, 16.0),
    "margin.top": (1.0, 1.0, 6.0),
    "margin.bottom": (1.0, 1.0, 6.0),
    "padding.left": (0.8, 1.0, 6.0),
    "opacity": (1.2, 0.02, 0.2),
    "border.width": (0.8, 0.5, 1.5),
    "border.radius": (0.6, 1.0, 4.0),
    "series.gap": (2.2, 1.0, 6.0),
    "series.height": (1.4, 1.0, 6.0),
    "series.left": (1.4, 2.0, 12.0),
    "series.count": (2.0, 0.0, 0.0),
}
EXACT = {"font.weight": 1.6, "font.family": 1.0, "font.transform": 0.8}
COLOR = {"color": 2.0, "background": 1.6, "border.color": 0.8}


def dig(obj, path):
    for key in path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def color_distance(a, b):
    if not a or not b or not a.startswith("#") or not b.startswith("#"):
        return None
    pair = [tuple(int(h[i : i + 2], 16) for i in (1, 3, 5)) for h in (a, b)]
    return sum(abs(x - y) for x, y in zip(*pair)) / 3.0


def grade(kind, delta, near, off):
    if kind == "exact":
        return "same" if delta == 0 else "OFF"
    if delta <= near:
        return "same"
    if delta <= off:
        return "close"
    return "OFF"


class Finding:
    def __init__(self, role, field, left, right, verdict, score, unit=""):
        self.role, self.field = role, field
        self.left, self.right = left, right
        self.verdict, self.score, self.unit = verdict, score, unit

    def line(self):
        mark = {"OFF": "✗", "close": "~", "same": "·"}[self.verdict]
        return f"  {mark} {self.field:<18} {self.left}{self.unit}  →  {self.right}{self.unit}"


def compare_node(role, left, right, findings, prefix=""):
    for field, (weight, near, off) in NUMERIC.items():
        if prefix and not field.startswith(prefix):
            continue
        a, b = dig(left, field), dig(right, field)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        delta = abs(a - b)
        verdict = grade("num", delta, near, off)
        scale = max(abs(a), abs(b), 1)
        findings.append(Finding(role, field, a, b, verdict, weight * (delta / scale), "px" if "ratio" not in field and "count" not in field and "opacity" not in field else ""))
    for field, weight in EXACT.items():
        a, b = dig(left, field), dig(right, field)
        if a is None or b is None:
            continue
        verdict = grade("exact", 0 if str(a) == str(b) else 1, 0, 0)
        findings.append(Finding(role, field, a, b, verdict, weight if verdict == "OFF" else 0.0))
    for field, weight in COLOR.items():
        # 🔴 A FULLY TRANSPARENT COLOUR IS NOT BLACK. getComputedStyle answers
        # rgba(0,0,0,0) for "no background", which parses to #000000 and would
        # otherwise report every unpainted element as a black-vs-theme
        # difference — noise loud enough to bury the real findings.
        a = "transparent" if dig(left, f"{field}.a") == 0 else dig(left, f"{field}.hex")
        b = "transparent" if dig(right, f"{field}.a") == 0 else dig(right, f"{field}.hex")
        if a is None or b is None:
            continue
        if a == "transparent" or b == "transparent":
            verdict = "same" if a == b else "OFF"
            findings.append(Finding(role, field, a, b, verdict, weight if verdict == "OFF" else 0.0))
            continue
        distance = color_distance(a, b)
        if distance is None:
            continue
        verdict = grade("num", distance, 6, 40)
        findings.append(Finding(role, field, a, b, verdict, weight * (distance / 255.0)))


def margins(node):
    box = node.get("margin") or [None] * 4
    pad = node.get("padding") or [None] * 4
    return {
        "margin": {"top": box[0], "bottom": box[2]},
        "padding": {"left": pad[3]},
    }


def diff(left, right, skip_color=False):
    """`unmeasurable` roles are skipped on BOTH sides. A role is unmeasurable on
    a target when that page has no element whose box IS the thing — LessWrong
    paints its collapsed tick on the row itself, so measuring `toc_dot` there
    compares a 3px mark against a 256px wrapper and reports a difference that
    does not exist. What could not be measured is named in the report rather
    than quietly dropped."""
    findings = []
    skip = set(left.get("unmeasurable") or []) | set(right.get("unmeasurable") or [])
    lroles, rroles = left.get("roles", {}), right.get("roles", {})
    for role in sorted(set(lroles) | set(rroles)):
        if role in skip:
            continue
        a, b = lroles.get(role, {}), rroles.get(role, {})
        if not a.get("found") or not b.get("found"):
            findings.append(Finding(role, "PRESENT", a.get("found", False), b.get("found", False), "OFF", 2.5))
            continue
        compare_node(role, {**a, **margins(a)}, {**b, **margins(b)}, findings)

    lcol, rcol = left.get("column"), right.get("column")
    if lcol and rcol:
        compare_node("column", {"column": lcol}, {"column": rcol}, findings, prefix="column.")

    for name in sorted(set(left.get("series", {})) | set(right.get("series", {}))):
        a, b = left["series"].get(name, {}), right["series"].get(name, {})
        if not a.get("found") or not b.get("found"):
            findings.append(Finding(f"series:{name}", "PRESENT", a.get("found", False), b.get("found", False), "OFF", 2.0))
            continue
        compare_node(f"series:{name}", {"series": a}, {"series": b}, findings, prefix="series.")

    if skip_color:
        findings = [f for f in findings if f.field not in COLOR]
    return findings, sorted(skip)


def report(left, right, findings, skipped=(), show_same=False):
    off = [f for f in findings if f.verdict == "OFF"]
    close = [f for f in findings if f.verdict == "close"]
    same = [f for f in findings if f.verdict == "same"]
    out = [
        f"design diff: {left['target']} (reference)  →  {right['target']}",
        f"  {left.get('url','')}",
        f"  {right.get('url','')}",
        "",
        f"{len(off)} different · {len(close)} close · {len(same)} matching"
        f"   (viewport {left['viewport']['w']}x{left['viewport']['h']})",
        "",
    ]
    if skipped:
        out.append(f"not comparable on these targets: {', '.join(skipped)}")
        out.append("")
    lp = (left.get("series", {}).get("paras") or {}).get("count")
    rp = (right.get("series", {}).get("paras") or {}).get("count")
    if lp and rp and lp != rp:
        # Two different documents. Counts, heights and total page length are
        # differences of CONTENT; only the per-element numbers mean anything.
        out.append(f"⚠ different content ({lp} vs {rp} paragraphs) — counts and heights are not design findings")
        out.append("")
    shown = off + close + (same if show_same else [])
    by_role = {}
    for finding in shown:
        by_role.setdefault(finding.role, []).append(finding)
    order = sorted(by_role, key=lambda r: -max(f.score for f in by_role[r]))
    for role in order:
        rows = sorted(by_role[role], key=lambda f: -f.score)
        if not rows:
            continue
        out.append(f"{role}")
        out.extend(row.line() for row in rows)
        out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Diff two captured design specs.")
    parser.add_argument("reference")
    parser.add_argument("subject")
    parser.add_argument("--out", default=os.path.join(HERE, "out"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--all", action="store_true", help="include matching fields")
    parser.add_argument("--no-color", action="store_true", help="ignore colours (dark vs light theme)")
    args = parser.parse_args()

    specs = []
    for name in (args.reference, args.subject):
        path = os.path.join(args.out, f"{name}.json")
        if not os.path.exists(path):
            sys.exit(f"no capture at {path} — run ./capture.py {name} first")
        with open(path) as handle:
            specs.append(json.load(handle))

    findings, skipped = diff(specs[0], specs[1], skip_color=args.no_color)
    if args.json:
        print(json.dumps([
            {"role": f.role, "field": f.field, "reference": f.left, "subject": f.right,
             "verdict": f.verdict, "score": round(f.score, 3)}
            for f in sorted(findings, key=lambda f: -f.score)
        ], indent=2))
    else:
        print(report(specs[0], specs[1], findings, skipped=skipped, show_same=args.all))


if __name__ == "__main__":
    main()
