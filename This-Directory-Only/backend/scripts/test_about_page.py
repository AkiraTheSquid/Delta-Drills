"""Focused regression checks for the About-page editor router."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import about_page_router as about


def main() -> None:
    original_file = about.CONTENT_FILE
    try:
        with tempfile.TemporaryDirectory() as directory:
            about.CONTENT_FILE = Path(directory) / "about-page.json"
            admin = SimpleNamespace(email="sethbgibson@gmail.com")
            result = about.update_about_page(
                about.AboutPageUpdate(html='<h1>Hello</h1><script>alert(1)</script>'),
                user=admin,
            )
            assert result.html == "<h1>Hello</h1>alert(1)"
            assert about.get_about_page().html == result.html
            try:
                about.update_about_page(about.AboutPageUpdate(html="<p>Nope</p>"), user=SimpleNamespace(email="nope@example.com"))
            except HTTPException as exc:
                assert exc.status_code == 403
            else:
                raise AssertionError("non-editor update was accepted")
    finally:
        about.CONTENT_FILE = original_file
    print("about-page router checks passed")


if __name__ == "__main__":
    main()
