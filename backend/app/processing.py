from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from split_LADR4e_chapters import read_exercise_sections, split_pdf_by_exercises
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"Failed to import splitter module: {exc}")


def run_auto_toc(
    pdf_path: Path,
    job_dir: Path,
    timeout: int = 600,
    openai_api_key: str | None = None,
    mathpix_app_id: str | None = None,
    mathpix_app_key: str | None = None,
) -> Path:
    script = PROJECT_ROOT / "glossary_to_csv.py"
    if not script.exists():
        raise FileNotFoundError(f"glossary_to_csv.py not found: {script}")

    chapters_csv = job_dir / "toc_chapters.csv"
    toc_csv = job_dir / "toc.csv"
    toc_md_dir = job_dir / "toc_md"
    toc_md_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        str(pdf_path),
        "--mathpix-out",
        str(toc_md_dir),
        "--csv-path",
        str(toc_csv),
        "--chapters-csv",
        str(chapters_csv),
        "--timeout",
        str(timeout),
    ]
    if settings.openai_model:
        cmd.extend(["--model", settings.openai_model])

    merged_env = os.environ.copy()
    if openai_api_key:
        merged_env["OPENAI_API_KEY"] = openai_api_key
    elif settings.openai_api_key:
        merged_env["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.openai_model:
        merged_env["OPENAI_MODEL"] = settings.openai_model
    if mathpix_app_id:
        merged_env["MATHPIX_APP_ID"] = mathpix_app_id
    elif settings.mathpix_app_id:
        merged_env["MATHPIX_APP_ID"] = settings.mathpix_app_id
    if mathpix_app_key:
        merged_env["MATHPIX_APP_KEY"] = mathpix_app_key
    elif settings.mathpix_app_key:
        merged_env["MATHPIX_APP_KEY"] = settings.mathpix_app_key

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=merged_env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"glossary_to_csv.py failed with exit code {result.returncode}")
    if not chapters_csv.exists():
        raise FileNotFoundError(f"Chapters CSV not found: {chapters_csv}")
    return chapters_csv


def split_chapters(pdf_path: Path, chapters_csv: Path, output_dir: Path, page_offset: int) -> tuple[list[tuple[str, int, int]], list[Path]]:
    sections = read_exercise_sections(chapters_csv)
    output_paths = split_pdf_by_exercises(pdf_path, chapters_csv, output_dir, page_offset)
    return sections, output_paths
