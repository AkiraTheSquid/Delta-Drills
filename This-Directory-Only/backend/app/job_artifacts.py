from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.models import Job, JobArtifact
from app.storage import job_root


def _read_text_if_possible(path: Path, max_chars: int = 500_000) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]..."
    return text


def upsert_job_artifact(
    db: Session,
    job: Job,
    artifact_key: str,
    artifact_kind: str,
    file_path: str | None = None,
    content_text: str | None = None,
) -> JobArtifact:
    artifact = (
        db.query(JobArtifact)
        .filter(JobArtifact.job_id == job.id, JobArtifact.artifact_key == artifact_key)
        .first()
    )
    if artifact is None:
        artifact = JobArtifact(
            job_id=job.id,
            artifact_key=artifact_key,
            artifact_kind=artifact_kind,
        )
        db.add(artifact)

    artifact.file_path = file_path
    artifact.content_text = content_text
    return artifact


def capture_auto_toc_artifacts(db: Session, job: Job) -> None:
    root = job_root(str(job.id))
    chatgpt_dir = root / "runtime" / "chatgpt"
    mathpix_dir = root / "runtime" / "mathpix"
    toc_md_dir = root / "toc_md"

    text_files = {
        "chatgpt.toc_detection.prompt": chatgpt_dir / "toc_detection_prompt.txt",
        "chatgpt.toc_detection.output": chatgpt_dir / "toc_detection_output.txt",
        "chatgpt.toc_detection.meta": chatgpt_dir / "toc_detection_meta.json",
        "chatgpt.toc_csv.prompt": chatgpt_dir / "toc_csv_prompt.txt",
        "chatgpt.toc_csv.output": chatgpt_dir / "toc_csv_output.txt",
        "chatgpt.toc_csv.meta": chatgpt_dir / "toc_csv_meta.json",
        "chatgpt.completion_state": chatgpt_dir / "completion_state.txt",
        "mathpix.completion_state": mathpix_dir / "completion_state.txt",
        "toc.csv": root / "toc.csv",
        "toc.chapters_csv": root / "toc_chapters.csv",
    }

    for artifact_key, path in text_files.items():
        content_text = _read_text_if_possible(path)
        if content_text is None and not path.exists():
            continue
        upsert_job_artifact(
            db,
            job,
            artifact_key=artifact_key,
            artifact_kind="text",
            file_path=str(path) if path.exists() else None,
            content_text=content_text,
        )

    if toc_md_dir.exists():
        for md_path in sorted(toc_md_dir.glob("*.md")):
            upsert_job_artifact(
                db,
                job,
                artifact_key=f"mathpix.markdown.{md_path.stem}",
                artifact_kind="text",
                file_path=str(md_path),
                content_text=_read_text_if_possible(md_path),
            )
        for log_path in sorted(toc_md_dir.glob("*.log")):
            upsert_job_artifact(
                db,
                job,
                artifact_key=f"mathpix.log.{log_path.stem}",
                artifact_kind="text",
                file_path=str(log_path),
                content_text=_read_text_if_possible(log_path),
            )
