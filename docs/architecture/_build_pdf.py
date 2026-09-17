#!/usr/bin/env python3
"""Render the EWMA model markdown to HTML (with KaTeX) and produce a PDF.

Pipeline:
  1. Read Delta-Drills-Current-Model-EWMA.md
  2. Convert to HTML via the `markdown` python package (extensions: tables, fenced_code)
  3. Splice into _render.html template at __CONTENT__ marker
  4. Run google-chrome --headless --print-to-pdf

The KaTeX math passes through the markdown conversion verbatim (because the
markdown package doesn't touch $...$ or $$...$$), and KaTeX auto-render
takes care of it on page load. Tables and fenced code blocks render via
markdown extensions.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Accept the stem (filename without extension) as an optional CLI arg so the
# same builder handles every markdown doc in this folder.
STEM = sys.argv[1] if len(sys.argv) > 1 else "Delta-Drills-Current-Model-EWMA"
MD = HERE / f"{STEM}.md"
TPL = HERE / "_render.html"
HTML = HERE / f"_{STEM}.staged.html"
PDF = HERE / f"{STEM}.pdf"

def md_to_html(md_src: str) -> str:
    """Use Python's `markdown` package to convert. We DO need to protect
    math from markdown's emphasis (single-_), so we mask $...$ before
    conversion and unmask after."""
    import markdown

    # Protect math regions from markdown mangling
    inline_math = re.findall(r"\$[^\$\n]+?\$", md_src)
    display_math = re.findall(r"\$\$[\s\S]+?\$\$", md_src)
    placeholders = {}
    for i, m in enumerate(display_math):
        key = f"@@DISPLAYMATH{i}@@"
        placeholders[key] = m
        md_src = md_src.replace(m, key, 1)
    for i, m in enumerate(inline_math):
        key = f"@@INLINEMATH{i}@@"
        placeholders[key] = m
        md_src = md_src.replace(m, key, 1)

    html = markdown.markdown(md_src, extensions=["tables", "fenced_code", "sane_lists"])

    # Unmask
    for key, math in placeholders.items():
        html = html.replace(key, math)
    return html

def main() -> int:
    md_src = MD.read_text(encoding="utf-8")
    content_html = md_to_html(md_src)
    tpl = TPL.read_text(encoding="utf-8")
    staged = tpl.replace("__CONTENT__", content_html)
    HTML.write_text(staged, encoding="utf-8")
    print(f"[stage] wrote {HTML}")

    # Headless chrome print-to-pdf. Use a profile dir off /tmp to avoid
    # polluting the user's real chrome state. --no-sandbox is needed in
    # some hardened environments but not here; --virtual-time-budget gives
    # KaTeX time to finish auto-rendering.
    out_url = HTML.resolve().as_uri()
    cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=8000",
        f"--print-to-pdf={PDF}",
        f"--user-data-dir=/tmp/dd-arch-pdf-{HERE.name}",
        out_url,
    ]
    print(f"[chrome] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f"[chrome] returncode={r.returncode}")
        print(f"[chrome] stdout:\n{r.stdout}")
        print(f"[chrome] stderr:\n{r.stderr}")
        return 1
    if not PDF.exists() or PDF.stat().st_size < 1024:
        print(f"[chrome] PDF missing or too small: {PDF}")
        return 1
    print(f"[ok] PDF: {PDF}  ({PDF.stat().st_size} bytes)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
