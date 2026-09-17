#!/usr/bin/env python3
"""
DOM CLONE — their rail's rendered markup and its RESOLVED css, as files.

    ./dom_clone.py lw            # capture, and write reference/lw-toc.html + .css
    ./dom_clone.py ours          # capture ours
    ./dom_clone.py --diff        # lw vs ours, property by property, per role

Seth, 2026-09-02: "Doesn't that other repository utilize HTML and CSS? Can't we
just duplicate the HTML and CSS and then replace everything else with the rest
of the stack that we're using?"

Half yes, and the half that is no is the reason this file exists.

🔴 FORUMMAGNUM SHIPS NO HTML AND NO CSS. Their rail is a .tsx component whose
styling is a TypeScript object literal handed to `defineStyles`; the class names
in the page (`FixedPositionToC-rowDot-a3f`) are minted by JSS at runtime. There
is no stylesheet in that repository to copy, which is why source_diff.py had to
parse object literals to say anything at all.

But the BROWSER has an HTML-and-CSS version of it — that is what it renders.
So this asks the browser for it: walk the rail's subtree, read every node's
computed style, and write the result out as an ordinary .html file and an
ordinary .css file. That artefact is the "duplicate the HTML and CSS" half of
the question, and it is strictly better than reading their source, because a
computed style has already resolved their theme, their breakpoints, their
inherited typography and their cascade.

And it is what catches the failure source_diff.py cannot: source_diff compares
declarations THEY WROTE against declarations WE WROTE, so a property neither
file mentions — because a parent, a theme or a MUI baseline supplies it — is
invisible to it. It reported 32 identical properties on a rail that plainly did
not match. This reads the finished result on both sides, so nothing hides.

LICENCE. ForumMagnum is GPL-3.0. What lands in `reference/` is a local
measurement artefact, gitignored alongside the screenshots, used the same way
the screenshots are: to compare against. It is not vendored, not served, and
not linked from the app — `tools/watch.py` fails if anything under tools/ ever
becomes reachable from index.html.
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cdp  # noqa: E402

OUT = os.path.join(HERE, "out")
REFERENCE = os.path.join(HERE, "reference")

# Properties whose value cannot mean anything across a light theme and a dark
# one. Reported as "set on both, values not comparable" rather than as drift —
# the same convention source_diff.py uses for `theme.palette.*`.
THEMED = {
    "color", "background-color", "background-image", "box-shadow", "filter",
    "border-top-color", "border-right-color", "border-bottom-color", "border-left-color",
    "font-family",
}

# Properties that say nothing about the design: they are consequences of the
# page around the element, not choices anyone made about the rail.
#
# `scrollbar-width` is here because the app sets `thin` on everything from a
# global rule, so it shows up on all eleven roles at once and says the same
# thing eleven times. `list-style-type` because our rows are <li> and theirs are
# <div>: `none` and `disc` both draw nothing on a non-list-item box.
IGNORED = {
    "transition-property", "transition-duration", "transition-timing-function",
    "grid-template-columns", "grid-template-rows", "grid-column", "grid-row",
    "scrollbar-width", "list-style-type",
}

# Places the port differs ON PURPOSE, with the reason. Same convention as
# source_diff.py's table, and for the same reason: a decision that reads as
# drift gets re-decided every time someone runs the tool.
#
# 🔴 THE WHOLE ROOT IS ONE DECISION. Their rail is a sticky block inside a
# centred CSS grid (MultiToCLayout); ours is `position: fixed` in a page that
# has no such grid, so every offset, the width and the stacking context differ
# by construction. What has to match is what is INSIDE it.
DEVIATIONS = {
    ("toc", "position"): "theirs is sticky inside a centred grid; ours is fixed, having no grid",
    ("toc", "top"): "ours starts under the app topbar (--dd-topbar-h)",
    ("toc", "right"): "a consequence of being fixed rather than laid out",
    ("toc", "bottom"): "ours is pinned to the bottom of the window; theirs ends with its block",
    ("toc", "left"): "a consequence of being fixed rather than laid out",
    ("toc", "width"): "ours carries their 17px inset as padding, because nothing centres it for us",
    ("toc", "height"): "the window is a different height than their sticky block",
    ("toc", "max-height"): "theirs caps at the viewport; ours is already pinned top and bottom",
    ("toc", "padding-bottom"): "ours clears the app's bottom chrome",
    ("toc", "font-size"): "the app's root type, inherited; every rail size below is set explicitly",
    ("toc", "line-height"): "as above",
    ("toc", "z-index"): "ours has to clear the app topbar",
    ("toc", "padding-left"): "their 17px inset, which their centred grid supplies and ours must ask for",
    # Their scroller is an ANCESTOR of the rail (MultiToCLayout's
    # stickyBlockScroller), so their own rows box never scrolls. We have no such
    # ancestor, so the rows box is the scroller — same behaviour, one element up.
    ("toc_rows", "overflow-x"): "ours is the scroller itself; theirs is scrolled by an ancestor we do not have",
    ("toc_rows", "overflow-y"): "as above",
    ("toc_rows", "overscroll-behavior-x"): "ours must not hand a finished list's scroll to the page",
    ("toc_rows", "overscroll-behavior-y"): "as above",
    ("toc_rows", "min-height"): "a flex scroller needs a 0 floor or it grows to its content instead of scrolling",
    ("toc_rows", "flex-basis"): "the `flex: 1` that goes with that floor",
    ("toc_rows", "height"): "the rail is as tall as the window, which is not their block's height",
    ("toc_progress", "height"): "the rail is as tall as the window, which is not their block's height",
    ("toc_fill", "height"): "how far down the notebook the capture was taken",
    ("toc_fill", "flex-basis"): "as above — this IS the read percentage",
    ("toc_row", "height"): "one section's share of a different document",
    ("toc_row", "width"): "the rows column is the same 256px; a row is as wide as its own label",
    ("toc_line", "height"): "one row's content height, which follows its heading level",
    ("toc_line", "width"): "as above",
    ("toc_label", "width"): "the labels are different words",
    ("toc_title", "width"): "the titles are different words",
    ("toc_fade", "width"): "as above",
    ("toc_current", "width"): "as above",
    ("toc_current", "height"): "whichever heading is current is a different level on each page",
    ("toc_current", "font-size"): "as above",
    ("toc_current", "line-height"): "as above",
    ("toc_row", "flex-grow"): "one section's share of a different document",
    ("toc_fill", "::after height"): "the window marker: how much of a different document a window shows",
}

# The roles worth lining up. `page`/`column`/prose roles belong to probe.js;
# this tool is pointed at the rail.
RAIL_ROLES = [
    "toc", "toc_progress", "toc_fill", "toc_rows", "toc_row", "toc_line",
    "toc_dot", "toc_fade", "toc_label", "toc_title", "toc_current",
]


def load_targets(path):
    with open(path) as handle:
        return json.load(handle)


def root_selector(spec):
    """The rail itself, taken from the target's own role map so this tool needs
    no second copy of anyone's class names."""
    toc = spec.get("roles", {}).get("toc") or []
    if not toc:
        raise SystemExit("target has no `toc` role — nothing to clone")
    return toc[0]


def capture(name, spec, keep=False):
    width, height = spec.get("viewport", cdp.DEFAULT_VIEWPORT)
    tab = cdp.open_tab(url="about:blank")
    try:
        tab.set_viewport(width, height)
        tab.navigate(spec["url"], ready=spec.get("ready"), settle=spec.get("settle", 1.4))
        for step in spec.get("pre", []):
            tab.evaluate(step)
            time.sleep(0.25)

        # 🔴 THE RAIL MUST BE OPEN OR HALF THE STYLES ARE THE CLOSED ONES.
        # Both designs fade their labels in on pointer position, and a capture
        # taken without the hover records `opacity: 0` as the label's style.
        hover = spec.get("hover")
        if hover:
            tab.send("Input.dispatchMouseEvent", type="mouseMoved", x=hover[0] + 40, y=hover[1] + 40)
            time.sleep(0.15)
            tab.send("Input.dispatchMouseEvent", type="mouseMoved", x=hover[0], y=hover[1])
            time.sleep(spec.get("hoverSettle", 0.6))

        with open(os.path.join(HERE, "dom_clone.js")) as handle:
            script = handle.read().strip().rstrip(";")
        payload = {"root": root_selector(spec), "roles": spec.get("roles", {})}
        result = tab.evaluate(f"({script})({json.dumps(payload)})")
        if not isinstance(result, dict):
            raise cdp.CdpError(f"dom_clone.js returned {result!r}")
        if result.get("error"):
            raise cdp.CdpError(result["error"])
        result["target"] = name
        # Same shape capture.py records, and for the same reason: a capture
        # that does not say what window it was taken in cannot be compared with
        # one taken in another. out/watch.py holds every capture to it.
        result["capturedWith"] = {"viewport": [width, height], "hover": hover}

        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, f"{name}.dom.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=1)
        return result, path
    finally:
        if not keep:
            tab.close()


# ---- emitting the static clone -------------------------------------------


def authored(node, baselines):
    """The declarations that are actually this element's design: everything its
    computed style holds that a bare element of the same tag on the same page
    does not. Without this subtraction every rule carries ninety lines of
    inherited body typography and browser defaults."""
    base = baselines.get(node["tag"], {})
    return {
        prop: value
        for prop, value in node["style"].items()
        if value and value != base.get(prop) and prop not in IGNORED
    }


def declarations(style):
    return "".join(f"  {prop}: {value};\n" for prop, value in sorted(style.items()))


def emit(result, stem):
    """Write the rail out as a plain .html file and a plain .css file.

    Class names are renumbered (`dc-12`) on purpose: their real ones are JSS
    output with a content hash in them, they change between deploys, and a file
    full of `FixedPositionToC-rowDot-a3f` invites exactly the copy-paste this
    port is not doing."""
    nodes = result["nodes"]
    baselines = result["baselines"]
    role_of = {}
    for role, hit in (result.get("roles") or {}).items():
        if hit and hit.get("node") is not None:
            role_of.setdefault(hit["node"], role)

    rules = []
    for node in nodes:
        style = authored(node, baselines)
        note = f"  /* role: {role_of[node['i']]} */\n" if node["i"] in role_of else ""
        if style:
            rules.append(f".dc-{node['i']} {{\n{note}{declarations(style)}}}\n")
        for pseudo, pstyle in (node.get("pseudo") or {}).items():
            trimmed = {
                prop: value
                for prop, value in pstyle.items()
                if value and value != baselines.get(node["tag"], {}).get(prop) and prop not in IGNORED
            }
            trimmed.setdefault("content", pstyle.get("content", '""'))
            rules.append(f".dc-{node['i']}{pseudo} {{\n{declarations(trimmed)}}}\n")

    # Nest the markup back up from the recorded paths.
    html = []
    stack = []  # (depth, closing tag)
    for node in nodes:
        while stack and stack[-1][0] >= node["depth"]:
            depth, tag = stack.pop()
            html.append(f"{'  ' * depth}</{tag}>")
        pad = "  " * node["depth"]
        html.append(f'{pad}<{node["tag"]} class="dc-{node["i"]}">')
        if node["text"]:
            html.append(f"{pad}  {escape(node['text'])}")
        stack.append((node["depth"], node["tag"]))
    while stack:
        depth, tag = stack.pop()
        html.append(f"{'  ' * depth}</{tag}>")

    os.makedirs(REFERENCE, exist_ok=True)
    css_path = os.path.join(REFERENCE, f"{stem}.css")
    html_path = os.path.join(REFERENCE, f"{stem}.html")
    header = (
        f"/* GENERATED by tools/visual-diff/dom_clone.py from {result['url']}\n"
        f"   at {result['viewport']['w']}x{result['viewport']['h']}, root font "
        f"{result['rootFontSize']}px, page background {result['pageBackground']}.\n"
        "   These are COMPUTED styles read back out of the browser, with every\n"
        "   property a bare element of the same tag already had subtracted. Do not\n"
        "   edit: re-run the tool. Do not serve: this is a measurement, and the\n"
        "   source it measures is GPL-3.0. */\n\n"
    )
    with open(css_path, "w") as handle:
        handle.write(header + "\n".join(rules))
    with open(html_path, "w") as handle:
        handle.write(
            f'<!doctype html>\n<meta charset="utf-8">\n<title>{stem}</title>\n'
            f'<link rel="stylesheet" href="{stem}.css">\n'
            f'<body style="background: {result["pageBackground"]}; margin: 0">\n'
            + "\n".join(html)
            + "\n</body>\n"
        )
    return html_path, css_path


def escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---- the diff -------------------------------------------------------------


def px(value):
    """`26px` and `26.0004px` are the same decision. Sub-pixel noise is what
    subpixel layout leaves behind, not a difference anyone wrote."""
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)px", (value or "").strip())
    return round(float(match.group(1)), 1) if match else None


ALIASES = {"start": "left", "end": "right"}


def same(a, b):
    a, b = ALIASES.get(a, a), ALIASES.get(b, b)
    if a == b:
        return True
    left, right = px(a), px(b)
    if left is not None and right is not None:
        return abs(left - right) <= 0.6
    return False


def node_for(result, role):
    hit = (result.get("roles") or {}).get(role)
    if not hit or hit.get("node") is None:
        return None
    return result["nodes"][hit["node"]]


def diff(left, right, show_themed=False):
    lines = []
    totals = {"same": 0, "differ": 0, "themed": 0, "missing": 0, "deliberate": 0}

    lines.append(f"structure   theirs {len(left['nodes'])} nodes  ·  ours {len(right['nodes'])} nodes")
    lines.append("")

    for role in RAIL_ROLES:
        a, b = node_for(left, role), node_for(right, role)
        if not a or not b:
            which = "theirs" if not a else "ours"
            lines.append(f"── {role}: NOT FOUND in {which}")
            totals["missing"] += 1
            continue

        head = (
            f"── {role}\n"
            f"   theirs  <{a['tag']}>  {a['box']['w']}x{a['box']['h']} at x={a['box']['x']}\n"
            f"   ours    <{b['tag']}>  {b['box']['w']}x{b['box']['h']} at x={b['box']['x']}"
        )
        body = []
        for prop in left["props"]:
            if prop in IGNORED:
                continue
            theirs, mine = a["style"].get(prop, ""), b["style"].get(prop, "")
            if prop in THEMED:
                if theirs and mine:
                    totals["themed"] += 1
                    if show_themed:
                        body.append(f"   ~ {prop:<26} theirs {theirs}   ours {mine}")
                continue
            if same(theirs, mine):
                totals["same"] += 1
                continue
            reason = DEVIATIONS.get((role, prop))
            if reason:
                totals["deliberate"] += 1
                if show_themed:
                    body.append(f"   = {prop:<26} {reason}")
                continue
            totals["differ"] += 1
            body.append(f"   ✗ {prop:<26} theirs {theirs or '—':<22} ours {mine or '—'}")

        # The marks both rails draw are pseudo-elements; a missing one is the
        # single most visible way a port can be wrong and the easiest to miss.
        for pseudo in ("::before", "::after"):
            has_a, has_b = pseudo in (a.get("pseudo") or {}), pseudo in (b.get("pseudo") or {})
            if has_a != has_b:
                totals["differ"] += 1
                body.append(f"   ✗ {pseudo:<26} theirs {'present' if has_a else 'absent':<22} ours {'present' if has_b else 'absent'}")
            elif has_a:
                for prop in ("content", "width", "height", "background-color", "font-size", "line-height"):
                    theirs = a["pseudo"][pseudo].get(prop, "")
                    mine = b["pseudo"][pseudo].get(prop, "")
                    if prop in THEMED or same(theirs, mine):
                        continue
                    reason = DEVIATIONS.get((role, f"{pseudo} {prop}"))
                    if reason:
                        totals["deliberate"] += 1
                        if show_themed:
                            body.append(f"   = {pseudo} {prop:<19} {reason}")
                        continue
                    totals["differ"] += 1
                    body.append(f"   ✗ {pseudo} {prop:<19} theirs {theirs or '—':<22} ours {mine or '—'}")

        lines.append(head)
        lines.extend(body or ["   (every compared property matches)"])
        lines.append("")

    return lines, totals


def load(name):
    path = os.path.join(OUT, f"{name}.dom.json")
    if not os.path.exists(path):
        raise SystemExit(f"no capture at {path} — run `./dom_clone.py {name}` first")
    with open(path) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Clone a rail's rendered markup and computed styling.")
    parser.add_argument("target", nargs="?", help="a name from targets.json")
    parser.add_argument("--targets", default=os.path.join(HERE, "targets.json"))
    parser.add_argument("--diff", action="store_true", help="compare two existing captures")
    parser.add_argument("--left", default="lw")
    parser.add_argument("--right", default="ours")
    parser.add_argument("--themed", action="store_true", help="also print theme-dependent values")
    parser.add_argument("--keep", action="store_true", help="leave the tab open")
    parser.add_argument("--no-emit", action="store_true", help="capture without writing reference/")
    args = parser.parse_args()

    if args.diff:
        lines, totals = diff(load(args.left), load(args.right), show_themed=args.themed)
        print(f"{args.left} vs {args.right}\n")
        print("\n".join(lines))
        print(
            f"{totals['same']} identical · {totals['differ']} different · "
            f"{totals['deliberate']} deliberate · {totals['themed']} theme-dependent · "
            f"{totals['missing']} role(s) unmatched"
        )
        return 1 if totals["differ"] or totals["missing"] else 0

    if not args.target:
        parser.error("give a target to capture, or --diff")
    if not cdp.browser_is_up():
        sys.exit("no Chrome on :9222 — start one with `chrome_dev about:blank`")

    targets = load_targets(args.targets)
    if args.target not in targets:
        sys.exit(f"unknown target {args.target!r}; have: {', '.join(sorted(targets))}")

    result, path = capture(args.target, targets[args.target], keep=args.keep)
    found = sum(1 for hit in result["roles"].values() if hit)
    print(f"{args.target}: {len(result['nodes'])} nodes, {found}/{len(result['roles'])} roles -> {path}")
    if not args.no_emit:
        html_path, css_path = emit(result, f"{args.target}-toc")
        print(f"  clone -> {os.path.relpath(html_path, HERE)} + {os.path.relpath(css_path, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
