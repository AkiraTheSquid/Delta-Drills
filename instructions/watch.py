"""watch.py — health checks for instructions

Specs are prose, so nothing here imports. What can rot instead is the evidence:
a spec cites `practice/colab-route.js:221`, someone renames the file, and the
spec keeps reading as authoritative while pointing at nothing. These checks
guard the two properties that make a spec trustworthy — it says whether it is
authorized, and the code it points at still exists.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

VALID_STATUS = ("DRAFT", "AUTHORIZED", "SUPERSEDED", "ABANDONED")

# Only paths rooted at a real top-level directory are checked. Without this the
# scan trips over every `example.js` and `solve(rows)` in a code sample.
ROOTS = (
    "Local_Deployed_Shared/",
    "This-Directory-Only/",
    "arena-book-colab/",
    "concept-graph/",
    "instructions/",
    "scripts/",
)

# `path/to/file.ext`, optionally `:123` — the form specs use to cite code.
CITATION = re.compile(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,6})(?::[~\d]+)?`")


def _specs():
    return sorted(
        p for p in glob.glob(os.path.join(HERE, "*.md"))
        if os.path.basename(p) != "README.md"
    )


def _status(path):
    with open(path, encoding="utf-8") as fh:
        match = re.search(r"(?m)^Status:\s*([A-Z]+)", fh.read(2000))
    return match.group(1) if match else ""


def check_imports():
    """Nothing here is importable. Assert that, so a stray module is noticed."""
    stray = [
        os.path.basename(p)
        for p in glob.glob(os.path.join(HERE, "*.py"))
        if os.path.basename(p) != "watch.py"
    ]
    assert not stray, f"instructions/ holds specs, not code: {stray}"


def check_public_api():
    """Every spec declares a status. A spec without one reads as authorization."""
    missing = []
    bad = []
    for path in _specs():
        with open(path, encoding="utf-8") as fh:
            head = fh.read(2000)
        match = re.search(r"(?m)^Status:\s*([A-Z]+)", head)
        if not match:
            missing.append(os.path.basename(path))
        elif match.group(1) not in VALID_STATUS:
            bad.append(f"{os.path.basename(path)} -> {match.group(1)}")
    assert not missing, f"spec(s) with no `Status:` line near the top: {missing}"
    assert not bad, f"unrecognised status (expected one of {VALID_STATUS}): {bad}"


def check_invariants():
    """Cited files must exist, and the folder doc must not still be a stub."""
    readme = os.path.join(HERE, "README.md")
    if os.path.exists(readme):
        with open(readme, encoding="utf-8") as fh:
            first = fh.readline()
        assert "modulario:template" not in first, \
            "instructions/README.md is still the unfilled template"

    dangling = []
    for path in _specs():
        # A SUPERSEDED or ABANDONED spec is history. Its citations described the
        # tree as it was, and freezing them is the point — forcing them to keep
        # resolving would mean rewriting the record every time the code moves.
        if _status(path) in ("SUPERSEDED", "ABANDONED"):
            continue
        with open(path, encoding="utf-8") as fh:
            body = fh.read()
        for cited in set(CITATION.findall(body)):
            if not cited.startswith(ROOTS):
                continue
            if not os.path.exists(os.path.join(REPO, cited)):
                dangling.append(f"{os.path.basename(path)} -> {cited}")
    assert not dangling, (
        "spec cites path(s) that no longer exist; update the spec or mark it "
        f"SUPERSEDED: {sorted(dangling)}"
    )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
