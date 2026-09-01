"""watch.py — health checks for notes (the KC metadata layer)

Every note must be attachable: front matter `kc:` present, agreeing with the
filename, and naming a concept the registry knows. This duplicates the
compile-time guard ON PURPOSE — the compile guard only runs when someone
compiles, and a note edited between compiles would otherwise sit broken and
silent until the next unrelated compile blamed it.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE_RE = re.compile(r"\A---\s*\nkc:\s*(\S+)\s*\n---\s*\n?(.*)\Z", re.S)


def _notes():
    for path in sorted(glob.glob(os.path.join(HERE, "*.md"))):
        if os.path.basename(path) != "README.md":
            yield path


def check_imports():
    assert os.path.isfile(os.path.join(HERE, "..", "kc_registry.json")), "kc_registry.json missing"


def check_public_api():
    # The folder's "API" is the note contract: front matter parses, filename == kc.
    for path in _notes():
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            m = NOTE_RE.match(fh.read())
        assert m, f"{name}: missing `kc:` front matter"
        kc = m.group(1)
        assert os.path.splitext(name)[0] == kc, f"{name}: filename disagrees with kc: {kc}"
        assert m.group(2).strip(), f"{name}: empty body"


def check_invariants():
    # No orphans: every note names a live registry concept (a rename must move its note).
    with open(os.path.join(HERE, "..", "kc_registry.json"), encoding="utf-8") as fh:
        registry = json.load(fh)
    known = {kc["id"] for kc in registry["kcs"]}
    for path in _notes():
        with open(path, encoding="utf-8") as fh:
            m = NOTE_RE.match(fh.read())
        if m:
            assert m.group(1) in known, f"{os.path.basename(path)}: unknown KC {m.group(1)}"


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print("PASS notes watch")
