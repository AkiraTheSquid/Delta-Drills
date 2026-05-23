"""Watch script for arena-procedural-drills/scripts/.

Verifies every builder script can produce a valid notebook structure
without actually re-emitting (build is idempotent + cheap so just run it
in a dry-run mode by importing).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check_imports() -> None:
    # Each builder is stdlib-only (json, pathlib). Verify they import clean.
    for py in HERE.glob("build_*.py"):
        spec = importlib.util.spec_from_file_location(py.stem, py)
        assert spec and spec.loader, f"cannot load {py}"
        # Importing executes the module top-level, which writes the .ipynb.
        # That's the explicit contract: builders are scripts, not libraries.
        # We don't want side effects in `mod watch`, so just compile-check.
        with open(py, "rb") as f:
            compile(f.read(), str(py), "exec")


def check_public_api() -> None:
    builders = list(HERE.glob("build_*.py"))
    assert builders, "no build_*.py scripts present"
    for py in builders:
        text = py.read_text()
        # Each builder must declare its target atom + subtopic + title.
        assert "ATOM_ID" in text, f"{py.name}: missing ATOM_ID constant"
        assert "SUBTOPIC" in text, f"{py.name}: missing SUBTOPIC constant"
        assert "TITLE" in text, f"{py.name}: missing TITLE constant"
        # Each builder must write its output to ../prereqs_*/<atom>.ipynb.
        assert "OUT.write_text" in text or "OUT.write_bytes" in text, (
            f"{py.name}: no OUT.write_* call"
        )


def check_invariants() -> None:
    # Builder filename → atom-id naming: build_<atom_id_underscored>.py.
    for py in HERE.glob("build_*.py"):
        underscored = py.stem.replace("build_", "")
        # The atom-id in the script should be the kebab-case form of this.
        kebab = underscored.replace("_", "-")
        text = py.read_text()
        # Must declare ATOM_ID = "<kebab>" somewhere.
        assert f'ATOM_ID = "{kebab}"' in text, (
            f"{py.name}: ATOM_ID does not match filename — expected '{kebab}'"
        )


if __name__ == "__main__":
    try:
        check_imports()
        check_public_api()
        check_invariants()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print("PASS")
