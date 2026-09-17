#!/usr/bin/env python3
"""
PIXELS — the part of the comparison that does not care what the CSS says.

    ./pixels.py mockup ours                  # hovered rails
    ./pixels.py mockup.collapsed ours.collapsed
    ./pixels.py mockup ours --page           # whole viewport instead of the rail

Two things come out. A composite image (reference | subject | difference) to
look at, and an INK PROFILE of each rail: where the marks actually are down the
strip, how many, how far apart. The profile is the honest measure of whether
one rail is a map of the same document as the other, because it is computed
from what was painted rather than from what the stylesheet intended.

🔴 THE PROFILE IS BACKGROUND-RELATIVE. One of these pages is dark and one is
light; "ink" here means "differs from this image's own dominant colour", not
"is dark". Comparing absolute pixel values across two themes measures the theme
and nothing else.
"""

import argparse
import os
import sys
from collections import Counter

from PIL import Image, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))


def background_of(img):
    counts = Counter(img.getdata())
    return counts.most_common(1)[0][0]


def ink_bands(img, x_from=0, x_to=None, tol=18):
    """Runs of rows carrying a mark, with the horizontal extent of each."""
    width, height = img.size
    x_to = x_to or width
    pixels = img.load()
    bg = background_of(img)
    rows = []
    for y in range(height):
        xs = [
            x
            for x in range(x_from, x_to)
            if sum(abs(a - b) for a, b in zip(pixels[x, y], bg)) > tol * 3
        ]
        rows.append(xs)
    bands = []
    run = None
    for y, xs in enumerate(rows):
        if xs:
            if run is None:
                run = {"top": y, "bottom": y, "x0": min(xs), "x1": max(xs)}
            else:
                run["bottom"] = y
                run["x0"] = min(run["x0"], min(xs))
                run["x1"] = max(run["x1"], max(xs))
        elif run is not None:
            bands.append(run)
            run = None
    if run is not None:
        bands.append(run)
    return bg, bands


def median(values):
    if not values:
        return None
    values = sorted(values)
    mid = len(values) >> 1
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def profile(path, x_from, x_to):
    img = Image.open(path).convert("RGB")
    bg, bands = ink_bands(img, x_from, x_to)
    tops = [b["top"] for b in bands]
    gaps = [b - a for a, b in zip(tops, tops[1:])]
    return {
        "image": os.path.basename(path),
        "background": "#%02x%02x%02x" % bg,
        "marks": len(bands),
        "gap": median(gaps),
        "height": median([b["bottom"] - b["top"] + 1 for b in bands]),
        "width": median([b["x1"] - b["x0"] + 1 for b in bands]),
        "x": (min([b["x0"] for b in bands], default=None), max([b["x1"] for b in bands], default=None)),
    }


def composite(left_path, right_path, out_path, width):
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    height = min(left.height, right.height)
    left = left.crop((0, 0, min(width, left.width), height))
    right = right.crop((0, 0, min(width, right.width), height))
    # The difference panel is amplified: a design difference of a few grey
    # levels is invisible at 1x and is exactly what is being looked for.
    delta = ImageChops.difference(left, right).point(lambda v: min(255, v * 4))
    gap = 10
    sheet = Image.new("RGB", (left.width + right.width + delta.width + gap * 2, height), (110, 110, 110))
    x = 0
    for panel in (left, right, delta):
        sheet.paste(panel, (x, 0))
        x += panel.width + gap
    sheet.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compare two captures as pixels.")
    parser.add_argument("reference")
    parser.add_argument("subject")
    parser.add_argument("--out", default=os.path.join(HERE, "out"))
    parser.add_argument("--page", action="store_true", help="whole viewport, not the rail")
    parser.add_argument("--width", type=int, default=210, help="rail width to compare")
    args = parser.parse_args()

    suffix = ".png" if args.page else ".rail.png"
    paths = [os.path.join(args.out, f"{name}{suffix}") for name in (args.reference, args.subject)]
    for path in paths:
        if not os.path.exists(path):
            sys.exit(f"no screenshot at {path} — run ./capture.py first")

    out_path = os.path.join(args.out, f"{args.reference}-vs-{args.subject}{'' if args.page else '.rail'}.png")
    composite(paths[0], paths[1], out_path, 10_000 if args.page else args.width)

    print(f"composite (reference | subject | difference x4): {out_path}\n")
    if args.page:
        return
    print(f"{'':<26} {'marks':>6} {'gap':>7} {'height':>7} {'width':>7}  x-range   background")
    for path in paths:
        p = profile(path, 0, args.width)
        print(
            f"{p['image']:<26} {p['marks']:>6} {str(p['gap']):>7} {str(p['height']):>7} "
            f"{str(p['width']):>7}  {p['x'][0]}-{p['x'][1]:<5}  {p['background']}"
        )


if __name__ == "__main__":
    main()
