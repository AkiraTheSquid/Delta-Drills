"""watch.py — health checks for course-builder

The course builder is five plain scripts that talk through window globals,
so the checks are: every file is present and exports its global, index.html
loads them in dependency order after what they lean on, and the two rules
the README marks 🔴 still hold (no model/key/prompt in the client chat; the
copied MCP instructions never route the editing password through the AI).
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.dirname(HERE)

# Load order = dependency order: data before graph/chat, the page last.
FILES = {
    "mcp-instructions.js": "window.DDCourseMcpInstructions",
    "builder-data.js": "window.DDBuilderData",
    "builder-graph.js": "window.DDBuilderGraph",
    "builder-chat.js": "window.DDBuilderChat",
    "course-builder.js": "window.DDCourseBuilder",
}


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def check_imports():
    for name in FILES:
        path = os.path.join(HERE, name)
        assert os.path.isfile(path) and os.path.getsize(path) > 0, f"missing or empty: {name}"


def check_public_api():
    for name, glob in FILES.items():
        assert f"{glob} =" in _read(os.path.join(HERE, name)), f"{name} no longer defines {glob}"
    # What the builder borrows from other folders.
    chat = _read(os.path.join(SHARED, "conceptual", "conceptual_chat.js"))
    for sym in ("createStream", "resetChat", "loadBundle"):
        assert sym in chat.split("window.DDConceptualChat =")[-1], f"DDConceptualChat no longer exports {sym}"
    assert "window.deltaKcLessonHtml" in _read(os.path.join(SHARED, "concept-graph", "lesson-graph.js")), \
        "lesson-graph.js no longer exports deltaKcLessonHtml (the lesson view needs it)"


def check_invariants():
    html = _read(os.path.join(SHARED, "index.html"))
    assert 'id="page-course-builder"' in html, "index.html lost #page-course-builder"
    pos = []
    for name in ["conceptual/conceptual_chat.js", "courses.js"] + [f"course-builder/{n}" for n in FILES]:
        m = re.search(r'<script src="' + re.escape(name) + r'\?v=\d+"', html)
        assert m, f"index.html does not load {name}"
        pos.append((m.start(), name))
    assert pos[0][0] < pos[2][0] and pos[1][0] < pos[2][0], "course-builder scripts load before their dependencies"
    tail = [p for p, _ in pos[2:]]
    assert tail == sorted(tail), "course-builder scripts out of dependency order"

    chat = _read(os.path.join(HERE, "builder-chat.js"))
    code = re.sub(r"/\*.*?\*/", "", chat, flags=re.DOTALL)
    for bad in ("gpt-", "api_key", "apiKey", "system_prompt", "systemPrompt"):
        assert bad not in code, f"builder-chat.js names `{bad}` — the server owns model, key and prompt"

    text = _read(os.path.join(HERE, "mcp-instructions.js"))
    assert "dd-content login" in text, "MCP instructions must have the learner log in from their own terminal"
    assert "ask me for it, then call content_login" not in text, "MCP instructions route the password through the AI"


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
