"""watch.py — health checks for notebooks

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.

This folder holds BUILD OUTPUT, not source: `scripts/compile_web_notebooks.py`
compiles each lesson with the very same `build_notebook` that writes the Colab
`.ipynb` files, so the web notebook and the Colab notebook cannot describe
different lessons. That guarantee is only worth something while the two are
actually in step — a `.ipynb` regenerated without re-running the web compiler
would leave this folder describing last week's lesson, and nothing on the page
would look wrong. So the main check here is a cell-for-cell comparison against
the published notebooks.

Regenerate with:

    python3 scripts/compile_web_notebooks.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
IPYNB_DIR = os.path.join(REPO, "arena-book-colab", "ARENA_5.0", "ch-1-foundations")

# Roles are minted by `compile_web_notebooks.cell_role`; the renderer
# (`practice/notebook-view.js::_cellNode`) dispatches on them and draws nothing
# for a role it does not know. A new role must be taught to both.
ROLES = {"setup", "checker", "solution", "check", "hints", "problem", "code", "prose"}


def _manifest():
    with open(os.path.join(HERE, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def _notebook(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def check_imports():
    """Every compiled notebook is listed, and every listed notebook exists."""
    manifest = _manifest()
    listed = {entry["file"] for entry in manifest["lessons"]}
    assert listed, "manifest.json lists no lessons — did the compiler write anything?"

    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(HERE, "*.json"))}
    on_disk.discard("manifest.json")
    assert listed == on_disk, (
        "manifest.json and the folder disagree about which notebooks exist: "
        f"listed-not-present={sorted(listed - on_disk)}, "
        f"present-not-listed={sorted(on_disk - listed)} — the index the "
        "Notebooks tab reads is built from the manifest, so an unlisted file is "
        "a lesson nobody can open"
    )


def check_public_api():
    """The shape `practice/notebook-view.js` reads, on every notebook.

    The renderer walks `cells` once and trusts each entry to carry `t`, `id`,
    `role` and `src`; it groups problems by `q` and jumps to a concept through
    `segments`. Anything missing here is a silent hole in the page — a cell that
    renders blank, or a table-of-contents entry that scrolls nowhere.
    """
    for entry in _manifest()["lessons"]:
        nb = _notebook(entry["file"])
        where = entry["file"]
        for key in ("id", "title", "topic", "subtopic_key", "segments", "cells"):
            assert key in nb, f"{where} is missing top-level key {key!r}"
        assert nb["id"] == entry["id"], f"{where} says id={nb['id']!r}, manifest says {entry['id']!r}"
        assert nb["cells"], f"{where} has no cells"
        assert entry["cells"] == len(nb["cells"]), (
            f"{where} has {len(nb['cells'])} cells but the manifest advertises "
            f"{entry['cells']}"
        )

        ids = set()
        for i, cell in enumerate(nb["cells"]):
            for key in ("t", "id", "role", "src"):
                assert key in cell, f"{where} cell {i} is missing {key!r}"
            assert cell["t"] in ("code", "md"), f"{where} cell {i} has type {cell['t']!r}"
            assert cell["role"] in ROLES, (
                f"{where} cell {i} has role {cell['role']!r}, which "
                "notebook-view.js does not know how to draw"
            )
            assert cell["id"] not in ids, f"{where} has two cells with id {cell['id']!r}"
            ids.add(cell["id"])
            if cell["id"].startswith("dd-q"):
                assert "q" in cell, (
                    f"{where} cell {cell['id']!r} belongs to a problem but carries "
                    "no `q` — the renderer groups a problem by that number"
                )

        for key, anchor in nb["segments"].items():
            assert anchor in ids, (
                f"{where} segment {key!r} points at cell {anchor!r}, which is not "
                "in the notebook — the concept jump would scroll nowhere"
            )

        # A problem the manifest advertises must have somewhere to be graded.
        for q in entry["questions"]:
            assert f"dd-q{q}-check" in ids, (
                f"{where} advertises problem {q} but has no dd-q{q}-check cell — "
                "there would be nothing to press to be graded"
            )


def check_invariants():
    """The web notebook and the Colab notebook are the same notebook.

    Compared cell for cell: order, id, type and source. Trimmed fields (`role`,
    `q`) are derived from the id, so they cannot disagree without the id
    disagreeing first.
    """
    if not os.path.isdir(IPYNB_DIR):
        return  # the Colab notebooks are not checked out beside this folder

    for entry in _manifest()["lessons"]:
        matches = sorted(glob.glob(os.path.join(IPYNB_DIR, f"{entry['id']}-*.ipynb")))
        assert len(matches) == 1, (
            f"expected exactly one published notebook for {entry['id']}, found "
            f"{[os.path.basename(p) for p in matches]}"
        )
        with open(matches[0], encoding="utf-8") as f:
            published = json.load(f)["cells"]
        web = _notebook(entry["file"])["cells"]
        name = os.path.basename(matches[0])

        assert len(web) == len(published), (
            f"{entry['file']} has {len(web)} cells but {name} has "
            f"{len(published)} — re-run scripts/compile_web_notebooks.py"
        )
        for i, (w, p) in enumerate(zip(web, published)):
            assert w["id"] == p["id"], (
                f"{entry['file']} cell {i} is {w['id']!r} but {name} cell {i} is "
                f"{p['id']!r} — the two notebooks are no longer the same notebook"
            )
            expected = "code" if p["cell_type"] == "code" else "md"
            assert w["t"] == expected, (
                f"{entry['file']} cell {w['id']!r} is a {w['t']} cell but {name} "
                f"has it as {p['cell_type']}"
            )
            assert w["src"] == p["source"], (
                f"{entry['file']} cell {w['id']!r} does not match {name} — the "
                "web edition would teach or grade something the Colab edition "
                "does not. Re-run scripts/compile_web_notebooks.py"
            )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
