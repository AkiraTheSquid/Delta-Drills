"""watch_example_gate.py — the worked-example popup is loaded and asked last.

Split out of watch.py (RED on LOC) the way watch_notebook.py was; watch.py
imports the check and runs it with the rest. Folder-relative, no arguments.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def check_the_example_gate_runs_after_the_lesson_gates():
    """The worked-example popup (example-gate.js) is loaded and asked last.

    events.js asks three gates before rendering a drill: the first-contact
    lesson, the `worked` rung, then the scheduled example. Asking the example
    first would put an example in front of a lesson that is about to show the
    same example; loading the script before lessons.js/ladder.js would leave
    it without the renderer and the KP lookup it borrows from them.
    """
    events = read(os.path.join(HERE, "events.js"))
    lesson = events.index("LessonGate.maybeShow(nextQ")
    worked = events.index("LadderUI.maybeShowWorked(nextQ")
    example = events.index("ExampleGate.maybeShow(nextQ")
    assert lesson < worked < example, (
        "events.js asks the gates out of order — lesson, worked rung, then the "
        "scheduled example")
    html = read(os.path.join(HERE, "..", "index.html"))
    gate = html.index('src="practice/example-gate.js')
    assert html.index('src="practice/lessons.js') < gate and html.index('src="practice/ladder.js') < gate, (
        "example-gate.js is loaded before lessons.js/ladder.js, whose renderer "
        "and KP lookup it uses")
    assert gate < html.index('src="practice/events.js'), (
        "example-gate.js is loaded after events.js, which calls it")
    gate_js = read(os.path.join(HERE, "example-gate.js"))
    assert "ladder_example" in gate_js and "lesson-mode" in gate_js, (
        "example-gate.js no longer reads the server's schedule or takes the lesson screen")
    ladder_js = read(os.path.join(HERE, "ladder.js"))
    assert "SUPPORTED_STAGES = new Set([])" in ladder_js, (
        "ladder.js renders an example BESIDE a drill again — the popup is the only "
        "example on the drill rungs (2026-08-30)")
