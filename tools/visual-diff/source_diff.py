#!/usr/bin/env python3
"""
SOURCE DIFF — their stylesheet against ours, as text rather than as pixels.

    ./source_diff.py                       # every mapped element
    ./source_diff.py --element rowDot
    ./source_diff.py --repo ~/src/ForumMagnum

LessWrong's design does not live in a .css file. It lives in JSS: TypeScript
object literals inside `defineStyles(...)`, compiled to class names at runtime.
So this reads those literals out of the checkout, normalises them into ordinary
CSS declarations (`fontSize: 9` becomes `font-size: 9px`), and lines each one
up against the rule in our stylesheet that plays the same part.

What it answers is the question the rendered page cannot: not "do these look
alike" but "which declarations did they write that we never wrote". A property
they set and we do not is the most common way a port drifts — nothing looks
broken, one behaviour is simply missing.

🔴 VALUES THAT COME FROM THEIR THEME ARE NOT COMPARABLE. `theme.palette.grey[700]`
is a light-theme grey and ours is a dark-theme one; the report says the
property was checked for PRESENCE and leaves the value alone. Same for their
breakpoints. Everything numeric is compared for real.

LICENCE: ForumMagnum is GPL-3.0. This reads their source to REPORT on it; it
copies nothing into the app, and the port it checks was written by hand from
the values it prints.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOC = "packages/lesswrong/components/posts/TableOfContents"

# their element  ->  (source file, JSS key, our selector)
MAP = [
    ("root",        f"{TOC}/FixedPositionToC.tsx",  "root",                ".anb-toc"),
    ("rows",        f"{TOC}/FixedPositionToC.tsx",  "rows",                ".anb-toc-rows"),
    ("rowWrapper",  f"{TOC}/FixedPositionToC.tsx",  "rowWrapper",          ".anb-toc-row"),
    ("rowDot",      f"{TOC}/FixedPositionToC.tsx",  "rowDot",              ".anb-toc-dot"),
    ("progressBar", f"{TOC}/FixedPositionToC.tsx",  "progressBarContainer", ".anb-toc-progress"),
    ("tocTitle",    f"{TOC}/FixedPositionToC.tsx",  "tocTitle",            ".anb-toc-title"),
    ("link",        f"{TOC}/TableOfContentsRow.tsx", "link",               ".anb-toc-label"),
    # 🔴 THE LEVEL LIVES ON ITS OWN ELEMENT, AS IT DOES IN THEIRS. Their level
    # rules sit on TableOfContentsRow-root, the element BETWEEN the fader and
    # the link, so the indent stays out of the opacity transition. Ours is
    # .anb-toc-level, and the type size is the one part of the rule that has to
    # reach the link itself.
    ("level2",      f"{TOC}/TableOfContentsRow.tsx", "level2",             '.anb-toc-row[data-level="2"] .anb-toc-level'),
    ("level3",      f"{TOC}/TableOfContentsRow.tsx", "level3",             '.anb-toc-row[data-level="3"] .anb-toc-level'),
    ("level4",      f"{TOC}/TableOfContentsRow.tsx", "level4",             '.anb-toc-row[data-level="4"] .anb-toc-level'),
    ("rowWrapper2", f"{TOC}/FixedPositionToC.tsx",  "rowDotContainer",     ".anb-toc-line"),
    ("scroller",    f"{TOC}/MultiToCLayout.tsx",     "stickyBlockScroller", ".anb-toc-rows"),
]

OUR_CSS = os.path.normpath(os.path.join(HERE, "..", "..", "Local_Deployed_Shared", "styles", "practice", "arena-notebook-nav.css"))
UNITLESS = {"z-index", "opacity", "flex", "flex-grow", "flex-shrink", "font-weight", "line-height", "order"}

# Their root font-size, so `1.1rem` can be compared with the 14.3px we wrote.
LW_ROOT_PX = 13.0

# Places our port differs ON PURPOSE. Listed here so the report can tell a
# decision apart from drift — an unlisted difference is something to look at,
# and this table is the whole record of what was decided otherwise.
DEVIATIONS = {
    ("root", "top"): "our app has a fixed topbar; the rail starts under it",
    ("root", "flex-direction"): "their root is a column wrapping a row; ours is that row",
    ("root", "height"): "ours is pinned top-and-bottom instead of sized",
    ("root", "max-height"): "ours is pinned top-and-bottom instead of sized",
    ("root", "transition"): "ours fades the panel, theirs fades the whole ToC in",
    ("rowWrapper", "flex-direction"): "their row wraps a dot container; our row IS it",
    ("progressBar", "display"): "ours positions the fill absolutely rather than by flex",
    ("progressBar", "flex-direction"): "ours positions the fill absolutely rather than by flex",
    ("progressBar", "justify-content"): "ours positions the fill absolutely rather than by flex",
    ("progressBar", "margin-bottom"): "ours positions the fill absolutely rather than by flex",
    ("progressBar", "--scroll-amount"): "ours writes the fill height directly, no custom property",
    ("scroller", "position"): "their ToC column is sticky in a grid; ours is fixed",
    ("scroller", "top"): "their sticky offset; ours is pinned by the fixed rail around it",
    ("scroller", "font-size"): "ours sets type per row, not on the scroller",
    ("scroller", "line-height"): "ours sets type per row, not on the scroller",
    ("scroller", "text-align"): "inherited from the app's own left-aligned page",
    ("scroller", "transition"): "ours does not animate the block's top",
    ("scroller", "margin-left"): "our rail starts at the viewport edge",
    ("scroller", "padding-left"): "our rail starts at the viewport edge",
    ("scroller", "height"): "ours is pinned top-and-bottom instead of sized",
    ("scroller", "max-height"): "ours is pinned top-and-bottom instead of sized",
}


def repo_candidates():
    return [
        os.environ.get("FORUMMAGNUM", ""),
        os.path.join(HERE, "reference", "ForumMagnum"),
        os.path.expanduser("~/src/ForumMagnum"),
        "/tmp/claude-1000/-home-stellar-thread/250bbcaa-d765-412a-aeab-3003921c60c7/scratchpad/ForumMagnum",
    ]


def find_repo(explicit=None):
    for path in ([explicit] if explicit else []) + repo_candidates():
        if path and os.path.isdir(os.path.join(path, "packages/lesswrong")):
            return path
    return None


def kebab(name):
    return re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name).lower()


def _skip_literal(source, i):
    """Advance past a comment, string or template literal starting at `i`.

    🔴 A BRACE INSIDE A COMMENT OR A STRING IS NOT A BRACE. The scan below is
    counting delimiters to find where one JSS object ends, and their files are
    full of `// { ... }` and `` `calc(100vh - ${x}px)` `` — one unbalanced
    character in either and the block ends in the wrong place, which reports
    another element's properties as this one's. Returns the index just past the
    literal, or `i` when there is no literal here. Found by codex, 2026-09-02.
    """
    two = source[i : i + 2]
    if two == "//":
        end = source.find("\n", i)
        return len(source) if end == -1 else end
    if two == "/*":
        end = source.find("*/", i + 2)
        return len(source) if end == -1 else end + 2
    quote = source[i]
    if quote not in "\"'`":
        return i
    j = i + 1
    while j < len(source):
        if source[j] == "\\":
            j += 2
            continue
        if quote == "`" and source[j : j + 2] == "${":
            # A template substitution is code again, and it nests.
            j += 2
            depth = 1
            while j < len(source) and depth:
                j = _skip_literal(source, j) if source[j] in "\"'`/" else j
                if j < len(source):
                    if source[j] == "{":
                        depth += 1
                    elif source[j] == "}":
                        depth -= 1
                    j += 1
            continue
        if source[j] == quote:
            return j + 1
        j += 1
    return len(source)


def jss_block(source, key):
    """The direct declarations of one JSS key, nested objects skipped.

    A hand-rolled brace scan rather than a parser: the file is TypeScript with
    template literals, spreads and theme calls in it, and everything this needs
    is the flat `prop: value,` pairs at the top level of one object."""
    match = re.search(rf"(?m)^\s{{2}}{re.escape(key)}:\s*\{{", source)
    if not match:
        return None
    i = match.end()
    depth = 1
    body_start = i
    while i < len(source) and depth:
        skipped = _skip_literal(source, i)
        if skipped != i:
            i = skipped
            continue
        char = source[i]
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth -= 1
        i += 1
    body = source[body_start : i - 1]

    declarations = {}
    depth = 0
    line_start = 0
    pos = 0
    while pos < len(body):
        skipped = _skip_literal(body, pos)
        if skipped != pos:
            pos = skipped
            continue
        char = body[pos]
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth -= 1
        elif char == "," and depth == 0:
            _declaration(body[line_start:pos], declarations)
            line_start = pos + 1
        pos += 1
    _declaration(body[line_start:], declarations)
    return declarations


def _declaration(chunk, into):
    chunk = re.sub(r"//.*", "", chunk).strip()
    if not chunk or chunk.startswith("...") or chunk.startswith("["):
        return
    if ":" not in chunk:
        return
    name, _, value = chunk.partition(":")
    name = name.strip().strip("'\"")
    value = value.strip().rstrip(",").strip()
    # `&:hover`, `&::-webkit-scrollbar` and friends are nested RULES, not
    # declarations; the scan sees the head of one and would file it as a
    # property called "&".
    if not name or name.startswith("&") or "\n" in name or value.startswith("{"):
        return
    into[kebab(name)] = value


def normalise(prop, value):
    """Their value in CSS terms, or None when it depends on their theme."""
    value = value.strip()
    if "theme." in value or "var(" in value or "getClassName" in value:
        return None
    value = value.strip("'\"")
    rem = re.fullmatch(r"(-?\d+(?:\.\d+)?)rem", value)
    if rem:
        # Their rem is 13px at the root, which is why their 1.1rem rows measure
        # 14.3px on the rendered page.
        return f"{float(rem.group(1)) * LW_ROOT_PX:g}px"
    if re.fullmatch(r"-?\d+(\.\d+)?", value) and prop not in UNITLESS:
        return f"{float(value):g}px"
    if re.fullmatch(r"-?\d+(\.\d+)?", value):
        return f"{float(value):g}"
    return value


SIDES = ("top", "right", "bottom", "left")


def expand(prop, value):
    """`padding: 6px 0` sets padding-top and padding-bottom, and a report that
    does not know that says we never set either of them."""
    out = {prop: value}
    parts = value.split()
    if prop in ("padding", "margin") and 1 <= len(parts) <= 4:
        if len(parts) == 1:
            parts = parts * 4
        elif len(parts) == 2:
            parts = [parts[0], parts[1], parts[0], parts[1]]
        elif len(parts) == 3:
            parts = [parts[0], parts[1], parts[2], parts[1]]
        for side, part in zip(SIDES, parts):
            out[f"{prop}-{side}"] = part
    if prop == "flex":
        grow = parts[0] if parts else "0"
        out["flex-grow"] = grow
        if len(parts) > 1:
            out["flex-shrink"] = parts[1]
        if len(parts) > 2:
            out["flex-basis"] = parts[2]
    return out


def our_rules(path):
    with open(path) as handle:
        css = re.sub(r"/\*.*?\*/", "", handle.read(), flags=re.S)
    rules = {}
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        declarations = {}
        for line in body.split(";"):
            if ":" not in line:
                continue
            name, _, value = line.partition(":")
            declarations.update(expand(name.strip(), value.strip()))
        for selector in selectors.split(","):
            selector = " ".join(selector.split())
            if not selector or selector.startswith("@"):
                continue
            rules.setdefault(selector, {}).update(declarations)
    return rules


def _same(have, want):
    tidy = lambda v: re.sub(r"\b0px\b", "0", " ".join(v.split())).replace(" ", "")
    return tidy(have) == tidy(want)


def compare(repo, rules, only=None):
    lines = []
    totals = {"same": 0, "different": 0, "missing": 0, "theme": 0, "deviation": 0}
    for name, rel, key, selector in MAP:
        if only and only != name:
            continue
        path = os.path.join(repo, rel)
        if not os.path.exists(path):
            lines.append(f"{name}: {rel} not in the checkout")
            continue
        with open(path) as handle:
            block = jss_block(handle.read(), key)
        if block is None:
            lines.append(f"{name}: no `{key}` block in {os.path.basename(rel)}")
            continue
        ours = rules.get(selector, {})
        rows = []
        for prop, raw in sorted(block.items()):
            want = normalise(prop, raw)
            have = ours.get(prop)
            if want is None:
                verdict, shown = ("theme", f"{raw}  →  {have or '—'}")
                if have is None:
                    verdict = "missing"
            elif have is None:
                verdict, shown = "missing", f"{want}  →  —"
            elif _same(have, want):
                verdict, shown = "same", want
            else:
                verdict, shown = "different", f"{want}  →  {have}"
            totals[verdict] += 1
            rows.append((verdict, prop, shown))
        for index, (verdict, prop, shown) in enumerate(rows):
            if verdict in ("different", "missing") and (name, prop) in DEVIATIONS:
                totals[verdict] -= 1
                totals["deviation"] = totals.get("deviation", 0) + 1
                rows[index] = ("deviation", prop, f"{shown}   — {DEVIATIONS[(name, prop)]}")
        interesting = [r for r in rows if r[0] not in ("same", "deviation")]
        deviations = [r for r in rows if r[0] == "deviation"]
        mark = {"missing": "✗", "different": "~", "theme": "·", "deviation": "="}
        lines.append(f"{name}  ({key} → {selector})   {len(rows) - len(interesting)}/{len(rows)} identical")
        for verdict, prop, shown in interesting:
            lines.append(f"  {mark[verdict]} {prop:<18} {shown}")
        for verdict, prop, shown in deviations:
            lines.append(f"  {mark[verdict]} {prop:<18} {shown}")
        lines.append("")
    return lines, totals


def main():
    parser = argparse.ArgumentParser(description="Diff LessWrong's JSS against our CSS.")
    parser.add_argument("--repo", default=None, help="path to a ForumMagnum checkout")
    parser.add_argument("--css", default=OUR_CSS)
    parser.add_argument("--element", default=None, help="only this mapped element")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo = find_repo(args.repo)
    if not repo:
        sys.exit(
            "no ForumMagnum checkout found. Clone it and point at it:\n"
            "  gh repo clone ForumMagnum/ForumMagnum ~/src/ForumMagnum\n"
            "  FORUMMAGNUM=~/src/ForumMagnum ./source_diff.py"
        )

    rules = our_rules(args.css)
    lines, totals = compare(repo, rules, args.element)
    if args.json:
        print(json.dumps({"repo": repo, "totals": totals, "report": lines}, indent=2))
        return
    print(f"their JSS ({repo})\n  vs {args.css}\n")
    print(
        f"{totals['same']} identical · {totals['different']} different · "
        f"{totals['missing']} we never set · {totals['deviation']} deliberate · "
        f"{totals['theme']} theme-dependent (presence only)\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
