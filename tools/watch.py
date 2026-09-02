"""watch.py — health checks for tools/

Developer tooling that must stay OUT of the shipped app. These checks hold that
boundary; each tool's own checks live in its subfolder.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def check_imports():
    """Every subfolder is a tool, and a tool has a watcher of its own."""
    for name in sorted(os.listdir(HERE)):
        path = os.path.join(HERE, name)
        if not os.path.isdir(path) or name.startswith((".", "__")):
            continue
        assert os.path.exists(os.path.join(path, "watch.py")), f"{name}/ has no watch.py"
        assert os.path.exists(os.path.join(path, "README.md")), f"{name}/ has no README.md"


def check_public_api():
    """Each tool's watcher passes."""
    for name in sorted(os.listdir(HERE)):
        watcher = os.path.join(HERE, name, "watch.py")
        if not os.path.exists(watcher):
            continue
        done = subprocess.run([sys.executable, watcher], capture_output=True, text=True)
        assert done.returncode == 0, f"{name}/watch.py failed: {done.stderr.strip()[:400]}"


def check_invariants():
    """🔴 NOTHING UNDER tools/ IS SERVED. The app is 103 classic <script> tags in
    one index.html; a tool referenced from there would be shipped to learners and
    would break the moment its developer-only dependencies are missing."""
    index = os.path.join(REPO, "Local_Deployed_Shared", "index.html")
    if os.path.exists(index):
        markup = open(index, encoding="utf-8").read()
        assert "tools/" not in markup, "index.html references tools/ — nothing there is shipped"

    # And the dependency runs one way: a tool may read the app, the app may not
    # import a tool.
    app = os.path.join(REPO, "Local_Deployed_Shared")
    for root, dirs, files in os.walk(app):
        dirs[:] = [d for d in dirs if d not in ("node_modules", "lessons")]
        for name in files:
            if not name.endswith((".js", ".css")):
                continue
            text = open(os.path.join(root, name), encoding="utf-8", errors="ignore").read()
            assert "../tools/" not in text and "/tools/" not in text, (
                f"{os.path.relpath(os.path.join(root, name), REPO)} imports from tools/"
            )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"PASS tools ({len(checks)} checks)")
