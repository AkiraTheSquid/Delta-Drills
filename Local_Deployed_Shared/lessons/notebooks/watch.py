"""watch.py — health checks for notebooks

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.

This folder holds BUILD OUTPUT, not source, from TWO compilers with two
different guarantees.

`<lesson-id>.json` + `manifest.json` — the Delta Drills lessons.
`scripts/compile_web_notebooks.py` compiles each lesson with the very same
`build_notebook` that writes the Colab `.ipynb` files, so the web notebook and
the Colab notebook cannot describe different lessons. That guarantee is only
worth something while the two are actually in step — a `.ipynb` regenerated
without re-running the web compiler would leave this folder describing last
week's lesson, and nothing on the page would look wrong. So the main check for
that set is a cell-for-cell comparison against the published notebooks.

`arena-<slug>.json` + `arena-index.json` — the ARENA curriculum, since
2026-09-01. `scripts/compile_arena_notebooks.py` rewrites Callum McDougall's
upstream `.ipynb` into the same cell shape so the Courses tab can open a
section IN the app instead of at Google Colab. There is no shared-compiler
guarantee for these, because the source is not ours — the invariant that
replaces it is that the set of notebooks equals the set of sections the Courses
tab links to. A section with no notebook is a row that opens an error.

🔴 THE TWO SETS SHARE A FOLDER AND EACH COMPILER SWEEPS ONLY ITS OWN PREFIX.
Both delete files their own index does not name; a sweep over a bare `*.json`
glob would delete the other compiler's whole output.

Regenerate with:

    python3 scripts/compile_web_notebooks.py
    python3 scripts/compile_arena_notebooks.py
"""
import glob
import json
import os
import re
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
IPYNB_DIR = os.path.join(REPO, "arena-book-colab", "ARENA_5.0", "ch-1-foundations")

# Roles are minted by `compile_web_notebooks.cell_role`; the renderer
# (`practice/notebook-view.js::_cellNode`) dispatches on them and draws nothing
# for a role it does not know. A new role must be taught to both.
ROLES = {"setup", "checker", "solution", "check", "hints", "problem", "code", "prose"}

# The ARENA set has its own, much smaller grammar, minted by
# `compile_arena_notebooks.py::_cells` and drawn by
# `practice/arena-notebook.js::_cellNode`. `magic` is the one that is not
# obvious: a cell carrying `%pip install …` is a SyntaxError to the kernel, so
# it is rendered read-only with a reason instead of with a Run button.
ARENA_PREFIX = "arena-"
ARENA_INDEX = "arena-index.json"
ARENA_ROLES = {"prose", "details", "code", "magic"}


def _manifest():
    with open(os.path.join(HERE, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def _arena_index():
    with open(os.path.join(HERE, ARENA_INDEX), encoding="utf-8") as f:
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
    # The ARENA notebooks share this folder and answer to their own index —
    # see the module docstring. Compared against that index in
    # check_the_arena_curriculum_is_openable below, never against this one.
    on_disk = {name for name in on_disk if not name.startswith(ARENA_PREFIX)}
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


def check_the_arena_curriculum_is_openable():
    """Every ARENA section the Courses tab links to opens a notebook.

    The Courses tab renders its section rows from the list inside
    `Local_Deployed_Shared/courses.js`, and each row now opens
    `arena-<slug>.json` in the app rather than a Colab tab. Nothing at click
    time checks that the file is there: the view fetches it, gets a 404 and
    says "No notebook for …". So the section list, the index and the folder
    have to agree here, before a learner finds out.

    🔴 THE SECTION LIST IS THE AUTHORITY, NOT THE INDEX. Both are generated
    from courses.js, so comparing the index to the folder alone would pass
    happily on a curriculum that gained a section and was never recompiled —
    which is the whole failure this check exists for.
    """
    index = _arena_index()
    listed = {entry["file"] for entry in index["sections"]}
    assert listed, "arena-index.json lists no sections — did the compiler write anything?"

    on_disk = {
        os.path.basename(p)
        for p in glob.glob(os.path.join(HERE, f"{ARENA_PREFIX}*.json"))
    }
    on_disk.discard(ARENA_INDEX)
    assert listed == on_disk, (
        "arena-index.json and the folder disagree about which ARENA notebooks "
        f"exist: listed-not-present={sorted(listed - on_disk)}, "
        f"present-not-listed={sorted(on_disk - listed)} — re-run "
        "scripts/compile_arena_notebooks.py"
    )

    courses = os.path.join(REPO, "Local_Deployed_Shared", "courses.js")
    with open(courses, encoding="utf-8") as f:
        source = f.read()
    linked = re.findall(r'url:\s*"(/arena-book/[^"]+)"', source)
    compiled = {entry.get("notebook_path") for entry in index["sections"]}
    for url in linked:
        rel = urllib.parse.unquote(url[len("/arena-book/"):]).replace(".html", ".ipynb")
        assert rel in compiled, (
            f"courses.js links section {rel!r}, which has no compiled notebook — "
            "the row would open an error. Re-run scripts/compile_arena_notebooks.py"
        )

    for entry in index["sections"]:
        nb = _notebook(entry["file"])
        where = entry["file"]
        for key in ("id", "title", "chapter", "notebook_path", "cells"):
            assert key in nb, f"{where} is missing top-level key {key!r}"
        assert nb["id"] == entry["id"], f"{where} says id={nb['id']!r}, index says {entry['id']!r}"
        assert nb["cells"], f"{where} has no cells"
        assert entry["cells"] == len(nb["cells"]), (
            f"{where} has {len(nb['cells'])} cells but the index advertises {entry['cells']}"
        )
        ids = set()
        for i, cell in enumerate(nb["cells"]):
            for key in ("t", "id", "role", "src"):
                assert key in cell, f"{where} cell {i} is missing {key!r}"
            assert cell["t"] in ("code", "md"), f"{where} cell {i} has type {cell['t']!r}"
            assert cell["role"] in ARENA_ROLES, (
                f"{where} cell {i} has role {cell['role']!r}, which "
                "arena-notebook.js does not know how to draw"
            )
            assert cell["id"] not in ids, f"{where} has two cells with id {cell['id']!r}"
            ids.add(cell["id"])


def check_the_arena_notebooks_are_not_swept_away_by_the_lesson_compiler():
    """`compile_web_notebooks.py`'s stale sweep must skip the ARENA prefix.

    🔴 That sweep deletes every `*.json` in this folder its own manifest does
    not name, and it runs on every deploy. The ARENA notebooks are in this
    folder and are not in that manifest, so without the prefix guard a deploy
    silently deletes all 31 of them and every Courses section starts opening
    "No notebook for …" — with nothing in the deploy log to say why.
    """
    path = os.path.join(REPO, "scripts", "compile_web_notebooks.py")
    with open(path, encoding="utf-8") as f:
        source = f.read()
    sweep = source.split('for stale in sorted(args.out.glob("*.json")):', 1)
    assert len(sweep) == 2, (
        "compile_web_notebooks.py no longer sweeps with `args.out.glob(\"*.json\")` — "
        "re-point this check at whatever replaced it, and make sure the ARENA "
        "notebooks are still exempt"
    )
    guard = sweep[1][:400]
    assert f'startswith("{ARENA_PREFIX}")' in guard, (
        "compile_web_notebooks.py's stale sweep does not exempt the ARENA "
        "notebooks any more — the next deploy would delete every arena-*.json "
        "in this folder"
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
    checks = [
        check_imports,
        check_public_api,
        check_the_arena_curriculum_is_openable,
        check_the_arena_notebooks_are_not_swept_away_by_the_lesson_compiler,
        check_invariants,
    ]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
