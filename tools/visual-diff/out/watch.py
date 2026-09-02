"""watch.py — health checks for visual-diff/out

Capture artefacts. The checks here are about what must NOT happen to this
folder: it must stay disposable, and it must stay out of git.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def check_imports():
    assert os.path.isdir(HERE)


def check_public_api():
    """Whatever is here has to be readable as a capture: a JSON that names the
    viewport it was taken at, because a comparison across viewports measures the
    window instead of the design."""
    for name in sorted(os.listdir(HERE)):
        if not name.endswith(".json"):
            continue
        data = json.load(open(os.path.join(HERE, name), encoding="utf-8"))
        assert "roles" in data, f"{name} is not a capture"
        assert data.get("capturedWith", {}).get("viewport"), (
            f"{name} does not record the viewport it was taken at"
        )


def check_invariants():
    """🔴 NOTHING HERE IS COMMITTED. A capture is a photograph of a live
    third-party page at one moment; in the repo it becomes something a future
    session trusts."""
    tracked = subprocess.run(
        ["git", "ls-files", "--", HERE], cwd=REPO, capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        return  # not a git checkout; nothing to hold
    stray = [
        line for line in tracked.stdout.split()
        if not line.endswith(("out/README.md", "out/watch.py"))
    ]
    assert not stray, f"capture artefacts are tracked in git: {stray[:5]}"


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"PASS visual-diff/out ({len(checks)} checks)")
