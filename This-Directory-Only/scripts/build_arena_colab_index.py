#!/usr/bin/env python3
"""Emit a single static HTML index of every split ARENA exercise notebook.

Walks arena-book-colab/ARENA_5.0/, builds the Colab GitHub-redirect URL for
each .ipynb, and writes arena-book-colab/INDEX.html. One file you can open
locally and click through every exercise without any brain power.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
COLAB_ROOT = REPO_DIR / "arena-book-colab"
ARENA_DIR = COLAB_ROOT / "ARENA_5.0"
OUT_PATH = COLAB_ROOT / "INDEX.html"

GITHUB_USER = "AkiraTheSquid"
GITHUB_REPO = "Delta-Drills"
GITHUB_BRANCH = "main"

NAME_RE = re.compile(r"^(\d+)_(\d+(?:_\d+)?)_(\d+)_(.+)\.ipynb$")


def colab_url_for(rel_path: Path) -> str:
    rel_str = "/".join(rel_path.parts)
    return (
        f"https://colab.research.google.com/github/{GITHUB_USER}/{GITHUB_REPO}"
        f"/blob/{GITHUB_BRANCH}/arena-book-colab/{rel_str}"
    )


def github_url_for(rel_path: Path) -> str:
    rel_str = "/".join(rel_path.parts)
    return (
        f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}"
        f"/arena-book-colab/{rel_str}"
    )


def main() -> None:
    if not ARENA_DIR.exists():
        raise SystemExit(f"missing {ARENA_DIR} — run split_arena_exercises.py first")

    chapters = {}
    for nb in sorted(ARENA_DIR.rglob("*.ipynb")):
        rel = nb.relative_to(COLAB_ROOT)  # ARENA_5.0/<chapter>/<section>/<file>.ipynb
        if len(rel.parts) < 4:
            continue
        _, chapter_dir, section_dir, fname = rel.parts
        m = NAME_RE.match(fname)
        if not m:
            continue
        ex_id = f"{m.group(2).replace('_', '.')}.{m.group(3)}"
        title = m.group(4).replace("-", " ")
        chapters.setdefault(chapter_dir, {}).setdefault(section_dir, []).append(
            (ex_id, title, rel)
        )

    total = sum(len(exs) for ch in chapters.values() for exs in ch.values())

    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>ARENA Colab Index</title>",
        "<style>",
        "  body { font: 14px/1.5 -apple-system, system-ui, sans-serif;",
        "         max-width: 1100px; margin: 24px auto; padding: 0 20px;",
        "         background: #0f172a; color: #e2e8f0; }",
        "  h1 { color: #f1f5f9; }",
        "  h2 { color: #93c5fd; margin-top: 32px;",
        "       border-bottom: 1px solid #334155; padding-bottom: 6px; }",
        "  h3 { color: #cbd5e1; margin-top: 18px; font-size: 1.05em; }",
        "  .row { display: grid; grid-template-columns: 64px 1fr auto auto;",
        "         gap: 12px; align-items: center; padding: 4px 0;",
        "         border-bottom: 1px solid rgba(148,163,184,0.12); }",
        "  .id { color: #64748b; font-variant-numeric: tabular-nums; }",
        "  .title { color: #e2e8f0; }",
        "  a.pill { padding: 3px 10px; border-radius: 999px; font-size: 12px;",
        "           font-weight: 600; text-decoration: none;",
        "           border: 1px solid rgba(96,165,250,0.32); color: #93c5fd;",
        "           background: rgba(96,165,250,0.18); }",
        "  a.pill.colab { color: #fdba74;",
        "                 background: rgba(251,146,60,0.18);",
        "                 border-color: rgba(251,146,60,0.32); }",
        "  a.pill:hover { filter: brightness(1.2); }",
        "  .meta { color: #64748b; font-size: 12px; margin-top: 6px; }",
        "</style></head><body>",
        f"<h1>ARENA 5.0 — split exercise notebooks ({total} total)</h1>",
        '<div class="meta">Colab links resolve via GitHub redirect. '
        f'They will 404 until <code>arena-book-colab/</code> is pushed to '
        f'<code>{GITHUB_USER}/{GITHUB_REPO}@{GITHUB_BRANCH}</code>.</div>',
    ]

    for chapter in sorted(chapters):
        parts.append(f"<h2>{html.escape(chapter)}</h2>")
        for section in sorted(chapters[chapter]):
            parts.append(f"<h3>{html.escape(section)}</h3>")
            for ex_id, title, rel in sorted(chapters[chapter][section], key=lambda x: x[0]):
                colab = colab_url_for(rel)
                gh = github_url_for(rel)
                parts.append(
                    f'<div class="row"><span class="id">{html.escape(ex_id)}</span>'
                    f'<span class="title">{html.escape(title)}</span>'
                    f'<a class="pill" href="{html.escape(gh)}" target="_blank" rel="noreferrer">View on GitHub</a>'
                    f'<a class="pill colab" href="{html.escape(colab)}" target="_blank" rel="noreferrer">Open in Colab</a>'
                    f"</div>"
                )

    parts.append("</body></html>")
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote index for {total} exercises → {OUT_PATH}")
    print(f"Open with: xdg-open {OUT_PATH}")


if __name__ == "__main__":
    main()
