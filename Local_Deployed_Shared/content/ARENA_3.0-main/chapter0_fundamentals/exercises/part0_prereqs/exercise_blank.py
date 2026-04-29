"""Thin include-style shim for exercise_template.py.

- Importing from this file re-exports everything from exercise_template.
- Running this file executes exercise_template's __main__ block without duplicating code.
"""

try:
    from .exercise_template import *  # type: ignore[F401,F403]
except Exception:  # pragma: no cover
    from exercise_template import *  # type: ignore[F401,F403]


if __name__ == "__main__":
    import runpy
    from pathlib import Path

    template_path = Path(__file__).with_name("exercise_template.py")
    runpy.run_path(str(template_path), run_name="__main__")
