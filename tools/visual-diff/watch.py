"""watch.py — health checks for visual-diff

The harness that compares our ARENA notebook design against LessWrong's. These
checks are the ones that can run with no browser and no network: the tools
import, the targets they read are well formed, and the two halves of the
comparison still agree about the role vocabulary.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def check_imports():
    """Every tool has to at least parse — they are run by hand, so a syntax
    error in one of them is otherwise found weeks later at the moment it is
    needed."""
    import py_compile

    for name in ("cdp.py", "capture.py", "diff.py", "pixels.py", "source_diff.py"):
        path = os.path.join(HERE, name)
        assert os.path.exists(path), f"{name} is missing"
        py_compile.compile(path, doraise=True)


def check_public_api():
    import cdp
    import capture
    import diff
    import source_diff

    assert hasattr(cdp, "open_tab") and hasattr(cdp.Tab, "screenshot")
    assert callable(capture.capture)
    assert callable(diff.diff) and callable(diff.report)
    assert callable(source_diff.compare)


def check_invariants():
    targets = json.load(open(os.path.join(HERE, "targets.json")))
    targets = {k: v for k, v in targets.items() if isinstance(v, dict)}

    # The comparison IS the role map. A role present on one target and missing
    # on another silently drops out of every diff, which reads as agreement.
    lw_roles = set(targets["lw"]["roles"])
    for name in ("mockup", "ours"):
        missing = lw_roles - set(targets[name]["roles"]) - set(targets["lw"].get("unmeasurable", []))
        assert not missing, f"{name} has no selector for: {sorted(missing)}"

    for name, spec in targets.items():
        assert spec.get("url"), f"{name} has no url"
        # 🔴 A rail that is built asynchronously must be waited FOR, not waited
        # OUT: the synthetic hover has to land on markup that exists, or the
        # rail is photographed closed and every label reads as invisible.
        if any("toc" in role for role in spec["roles"]) and spec.get("hover"):
            assert "anb-toc" in spec.get("ready", "") or "toc" in spec.get("ready", "") or name == "lw", (
                f"{name} hovers a ToC but does not wait for one in `ready`"
            )

    # probe.js and dom_clone.js are evaluated as EXPRESSIONS, so they must stay
    # one apiece — both are wrapped in `(<file>)(<spec>)` and handed to
    # Runtime.evaluate, where a trailing statement is a SyntaxError at exactly
    # the moment the tool is needed.
    for name in ("probe.js", "dom_clone.js"):
        source = open(os.path.join(HERE, name)).read().strip().rstrip(";")
        body = re.sub(r"^/\*.*?\*/", "", source, flags=re.S).strip()
        assert body.startswith("(function") and body.endswith(")"), (
            f"{name} must stay ONE expression — it is wrapped in a Runtime.evaluate "
            "and a trailing statement makes it a SyntaxError at the moment it is needed"
        )

    # 🔴 EVERY DEVIATION CARRIES A REASON, IN BOTH TOOLS. A difference with no
    # written reason is indistinguishable from drift, and gets re-decided every
    # time someone runs the report.
    clone = open(os.path.join(HERE, "dom_clone.py")).read()
    table = re.search(r"^DEVIATIONS = \{(.*?)^\}", clone, re.S | re.M)
    assert table, "dom_clone.py has lost its DEVIATIONS table"
    for line in table.group(1).splitlines():
        entry = re.match(r'\s*\("([^"]+)", "([^"]+)"\):\s*"(.*)"', line)
        if not entry:
            continue
        assert entry.group(3).strip(), (
            f"deviation {entry.group(1)}.{entry.group(2)} has no reason written for it"
        )

    # Nothing of theirs is vendored. ForumMagnum is GPL-3.0; this folder reads it
    # in place and re-types the values. The lw-toc.* clone is a MEASUREMENT of
    # their rendered page, gitignored beside the screenshots — see dom_clone.py.
    for root, dirs, files in os.walk(os.path.join(HERE, "reference")):
        dirs[:] = [d for d in dirs if d != "ForumMagnum"]
        for f in files:
            assert not f.endswith((".tsx", ".ts")), f"reference/ must not carry their source: {f}"

    # The deliberate-deviation table is the record of what we chose to differ
    # on. An entry with no reason is drift wearing a decision's clothes.
    src = open(os.path.join(HERE, "source_diff.py")).read()
    block = re.search(r"DEVIATIONS = \{(.*?)\n\}", src, re.S)
    assert block, "source_diff.py has lost its DEVIATIONS table"
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert re.search(r':\s*"[^"]{10,}"', line), f"deviation without a reason: {line}"


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"PASS visual-diff ({len(checks)} checks)")
