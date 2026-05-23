"""Watch script for arena-procedural-drills/.

Verifies the procedural-drill folder structure is consistent with the
template: each child topic folder has at least one .ipynb, scripts/ has
a matching builder, and every notebook has the required metadata block.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check_imports() -> None:
    # No Python package to import — this folder ships data + notebooks.
    # Still verify nbformat / json work on the .ipynb files we ship.
    import json as _json  # noqa: F401


VALID_BLOOM = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}


def check_public_api() -> None:
    # The "public API" of this folder is the set of .ipynb files + their
    # delta_drills metadata block. Verify each notebook is well-formed.
    notebooks = list(HERE.rglob("*.ipynb"))
    assert notebooks, "no .ipynb files in arena-procedural-drills/"
    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text())
        meta = nb.get("metadata", {}).get("delta_drills", {})
        assert "atom_id" in meta, f"{nb_path}: missing metadata.delta_drills.atom_id"
        assert "subtopic" in meta, f"{nb_path}: missing metadata.delta_drills.subtopic"
        assert meta.get("drill_kind") == "procedural", f"{nb_path}: drill_kind != procedural"
        assert nb_path.stem == meta["atom_id"], (
            f"{nb_path}: filename stem '{nb_path.stem}' != atom_id '{meta['atom_id']}'"
        )
        # v0.2+ schema fields: kc_decomposition + exercises[] arrays. Skip
        # for older notebooks (template_version < v0.2) so the invariant
        # is forward-compatible during the rollout window.
        version = meta.get("template_version", "v0.1")
        if version >= "v0.2":
            kcs = meta.get("kc_decomposition")
            assert isinstance(kcs, list) and kcs, (
                f"{nb_path}: v0.2 schema requires non-empty kc_decomposition[]"
            )
            for kc in kcs:
                assert kc.get("kind") in {"component-skill", "integrative-skill"}, (
                    f"{nb_path}: kc {kc.get('id')} has invalid kind"
                )
            exercises = meta.get("exercises")
            assert isinstance(exercises, list) and exercises, (
                f"{nb_path}: v0.2 schema requires non-empty exercises[]"
            )
            known_kcs = {kc["id"] for kc in kcs}
            for ex in exercises:
                for field in ("id", "title", "bloom_level", "difficulty", "kcs", "lo"):
                    assert field in ex, f"{nb_path}: exercise missing field '{field}'"
                assert ex["bloom_level"] in VALID_BLOOM, (
                    f"{nb_path}: exercise {ex['id']} bloom_level "
                    f"'{ex['bloom_level']}' not in {sorted(VALID_BLOOM)}"
                )
                unknown = set(ex["kcs"]) - known_kcs
                assert not unknown, (
                    f"{nb_path}: exercise {ex['id']} references unknown KCs: {sorted(unknown)}"
                )


def check_invariants() -> None:
    # Each topic folder must contain at least one .ipynb.
    for child in HERE.iterdir():
        if not child.is_dir() or child.name in {"scripts", "__pycache__"}:
            continue
        ipynbs = list(child.glob("*.ipynb"))
        assert ipynbs, f"topic folder {child} contains no .ipynb files"
    # Every notebook should have a matching builder in scripts/.
    scripts_dir = HERE / "scripts"
    if scripts_dir.exists():
        builder_atoms = {
            p.stem.replace("build_", "").replace("_", "-")
            for p in scripts_dir.glob("build_*.py")
        }
        notebook_atoms = {p.stem for p in HERE.rglob("*.ipynb")}
        orphans = notebook_atoms - builder_atoms
        assert not orphans, f"notebook(s) without builder script: {sorted(orphans)}"


if __name__ == "__main__":
    try:
        check_imports()
        check_public_api()
        check_invariants()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print("PASS")
