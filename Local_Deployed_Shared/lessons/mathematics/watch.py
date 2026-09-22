"""watch.py — health checks for lessons/mathematics (the math MC lane)

The folder's own contract (README.md): every `kp-*.md` here is a `kind: math`
page with a `kp-*.problems.json` beside it, and every problem in those files
is proven by scripts/validate_math.py (SymPy). The deep checks live in that
script, which lessons/watch.py already runs on the whole tree; this watch
is the folder-local half — the shape rules that make a page reach it.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import subprocess
from pathlib import Path
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(_DIR, ".."))


def check_imports():
    import math_bank  # the shared loader, stdlib-only

    assert callable(math_bank.load_math_rows)


def check_public_api():
    """Every page here is a math page with its problems beside it, and every
    problems file has a page — a lone half is a page the picker can serve
    with nothing to grade, or problems no lesson introduces."""
    import math_bank

    pages = {f[:-3] for f in os.listdir(_DIR) if f.startswith("kp-") and f.endswith(".md")}
    files = {f[: -len(".problems.json")] for f in os.listdir(_DIR) if f.endswith(".problems.json")}
    assert pages == files, (
        f"page/problems mismatch — pages without problems: {sorted(pages - files)}, "
        f"problems without a page: {sorted(files - pages)}"
    )
    for slug in sorted(pages):
        with open(os.path.join(_DIR, slug + ".md"), encoding="utf-8") as fh:
            head = fh.read(2000)
        assert "\nkind: math\n" in head, f"{slug}.md is in lessons/mathematics but is not `kind: math`"
        data = math_bank.read_problem_file(Path(_DIR) / (slug + ".problems.json"))
        assert data["kc"].startswith("math."), f"{slug}.problems.json kc {data['kc']!r} is not a math.* id"


def check_invariants():
    """Every problem proven. Delegates to scripts/validate_math.py under the
    backend venv (sympy lives there); passes on an empty folder."""
    venv = os.path.join(_REPO, "This-Directory-Only", "backend", ".venv", "bin", "python")
    python = venv if os.path.exists(venv) else sys.executable
    result = subprocess.run(
        [python, os.path.join(_REPO, "scripts", "validate_math.py")],
        capture_output=True, text=True, cwd=os.path.join(_REPO, "scripts"),
    )
    assert result.returncode == 0, (result.stdout + result.stderr).strip()[-2000:]


if __name__ == "__main__":
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
