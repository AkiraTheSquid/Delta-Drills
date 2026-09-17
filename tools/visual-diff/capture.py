#!/usr/bin/env python3
"""
CAPTURE — load one target at a pinned viewport, measure it, photograph it.

    ./capture.py ours
    ./capture.py lw --discover        # dump candidate selectors instead
    ./capture.py mockup --keep        # leave the tab open to look at

A target is one entry in targets.json: a URL, the expression that means "the
thing I am about to measure has rendered", an optional list of scripted setup
steps, an optional real mouse hover, and the selectors that map this page's
markup onto the SHARED ROLE VOCABULARY the diff speaks.

🔴 THE HOVER IS A REAL MOUSE EVENT, not a class the script adds. Both rails
under comparison open on pointer position — LessWrong's through CSS :hover,
ours through mouseenter on a gutter strip — and neither reacts to anything a
`classList.add` can do from the outside. Input.dispatchMouseEvent is the only
way to measure the open state as the reader actually gets it.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cdp  # noqa: E402

DISCOVER_JS = r"""
(() => {
  const out = { fixedLeft: [], paragraphs: [], headings: [], code: [] };
  const named = (el) => {
    const cls = (el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className) || "";
    return { tag: el.tagName.toLowerCase(), cls: String(cls).slice(0, 160), id: el.id || null };
  };
  /* Anything pinned near the left edge and tall enough to be a contents rail. */
  document.querySelectorAll("body *").forEach((el) => {
    const cs = getComputedStyle(el);
    if (cs.position !== "fixed" && cs.position !== "sticky") return;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height < 200 || r.left > 420) return;
    out.fixedLeft.push(Object.assign(named(el), { x: Math.round(r.left), w: Math.round(r.width), h: Math.round(r.height), pos: cs.position }));
  });
  const tally = (sel, bucket) => {
    const counts = new Map();
    document.querySelectorAll(sel).forEach((el) => {
      const parent = el.parentElement;
      if (!parent) return;
      const key = named(parent).cls.split(/\s+/)[0] || parent.tagName.toLowerCase();
      counts.set(key, (counts.get(key) || 0) + 1);
    });
    [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6).forEach(([k, n]) => bucket.push({ parent: k, n }));
  };
  tally("p", out.paragraphs);
  tally("h1,h2,h3", out.headings);
  tally("pre,code", out.code);
  return out;
})()
"""


def load_targets(path):
    with open(path) as handle:
        return json.load(handle)


def capture(name, spec, out_dir, keep=False, discover=False, url_override=None, no_hover=False):
    """`no_hover` measures the rail as it sits when nobody is pointing at it —
    the collapsed state, which is the one a reader looks at most of the time
    and the one the two designs are hardest to tell apart in."""
    width, height = spec.get("viewport", cdp.DEFAULT_VIEWPORT)
    if no_hover:
        spec = {**spec, "hover": None}
        name = f"{name}.collapsed"
    url = url_override or spec["url"]
    tab = cdp.open_tab(url="about:blank")
    try:
        tab.set_viewport(width, height)
        tab.navigate(url, ready=spec.get("ready"), settle=spec.get("settle", 1.4))

        for step in spec.get("pre", []):
            tab.evaluate(step)
            time.sleep(0.25)

        hover = spec.get("hover")
        if hover:
            # Two moves: some hover handlers only fire on a CHANGE of position.
            tab.send("Input.dispatchMouseEvent", type="mouseMoved", x=hover[0] + 40, y=hover[1] + 40)
            time.sleep(0.15)
            tab.send("Input.dispatchMouseEvent", type="mouseMoved", x=hover[0], y=hover[1])
            time.sleep(spec.get("hoverSettle", 0.6))

        if discover:
            found = tab.evaluate(DISCOVER_JS)
            print(json.dumps(found, indent=2))
            return None

        with open(os.path.join(HERE, "probe.js")) as handle:
            # probe.js is a parenthesised function EXPRESSION and ends in a
            # statement terminator; calling it needs the `;` gone or the whole
            # thing parses as a statement followed by a stray call.
            probe = handle.read().strip().rstrip(";")
        result = tab.evaluate(f"({probe})({json.dumps(spec)})")
        if not isinstance(result, dict):
            raise cdp.CdpError(f"probe returned {result!r}")
        result["target"] = name
        result["capturedWith"] = {"viewport": [width, height], "hover": hover}
        # Carried into the capture so the diff knows what this page cannot
        # answer for, without needing targets.json to read a spec back.
        result["unmeasurable"] = spec.get("unmeasurable", [])

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        tab.screenshot(os.path.join(out_dir, f"{name}.png"))
        # The rail on its own, because that is the part being copied and a
        # full-page shot of two different documents diffs into pure noise.
        # 🔴 THE CLIP IS IN DOCUMENT COORDINATES, NOT VIEWPORT ONES. captureBeyondViewport
        # photographs the whole page, so y=0 is the top of the document — which
        # on a sticky rail is the one place the rail is not yet stuck, and the
        # crop comes back empty. The clip has to start at the scroll position
        # the measurement was taken at.
        top = tab.evaluate("window.scrollY") or 0
        tab.screenshot(
            os.path.join(out_dir, f"{name}.rail.png"),
            clip={"x": 0, "y": top, "width": spec.get("railWidth", 340), "height": height},
        )
        return result
    finally:
        if not keep:
            tab.close()


def main():
    parser = argparse.ArgumentParser(description="Measure one target's design.")
    parser.add_argument("target")
    parser.add_argument("--targets", default=os.path.join(HERE, "targets.json"))
    parser.add_argument("--out", default=os.path.join(HERE, "out"))
    parser.add_argument("--url", default=None, help="override the target's URL")
    parser.add_argument("--keep", action="store_true", help="leave the tab open")
    parser.add_argument("--discover", action="store_true", help="dump candidate selectors")
    parser.add_argument("--no-hover", action="store_true", help="measure the rail closed")
    args = parser.parse_args()

    if not cdp.browser_is_up():
        sys.exit("no Chrome on :9222 — start one with `chrome_dev about:blank`")

    targets = load_targets(args.targets)
    if args.target not in targets:
        sys.exit(f"unknown target {args.target!r}; have: {', '.join(sorted(targets))}")

    result = capture(
        args.target, targets[args.target], args.out,
        keep=args.keep, discover=args.discover, url_override=args.url, no_hover=args.no_hover,
    )
    if result:
        roles = result["roles"]
        missing = sorted(name for name, role in roles.items() if not role.get("found"))
        print(f"{result['target']}: {len(roles) - len(missing)}/{len(roles)} roles measured -> {args.out}/{result['target']}.json")
        if missing:
            print(f"  MISSING (selectors need work): {', '.join(missing)}")


if __name__ == "__main__":
    main()
