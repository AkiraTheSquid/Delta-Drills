from __future__ import annotations

import logging
import shutil
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.db import SessionLocal, get_db
from app.job_artifacts import capture_auto_toc_artifacts
from app.models import Chapter, Job, JobArtifact, User
from app.processing import run_auto_toc, split_chapters
from app.schemas import JobArtifactOut, JobOut
from app.storage import job_chapters_dir, job_input_path, job_root

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


def job_to_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error_message=job.error_message,
    )


@router.post("", response_model=JobOut)
async def create_job(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    chapters_csv: UploadFile | None = File(None),
    auto_toc: bool = Form(False),
    page_offset: int = Form(0),
    openai_api_key: str | None = Form(None),
    mathpix_app_id: str | None = Form(None),
    mathpix_app_key: str | None = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> JobOut:
    if not pdf_file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing PDF filename")

    job = Job(
        user_id=user.id,
        original_filename=pdf_file.filename,
        status="queued",
        pdf_path="",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    input_path = job_input_path(str(job.id), pdf_file.filename)
    with input_path.open("wb") as f:
        shutil.copyfileobj(pdf_file.file, f)
    job.pdf_path = str(input_path)

    if chapters_csv and chapters_csv.filename:
        csv_path = job_root(str(job.id)) / "chapters.csv"
        with csv_path.open("wb") as f:
            shutil.copyfileobj(chapters_csv.file, f)
        job.chapters_csv_path = str(csv_path)

    db.commit()
    background_tasks.add_task(
        process_job,
        str(job.id),
        page_offset,
        auto_toc,
        openai_api_key,
        mathpix_app_id,
        mathpix_app_key,
    )
    return job_to_out(job)


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> JobOut:
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job_to_out(job)


@router.get("/{job_id}/artifacts", response_model=list[JobArtifactOut])
def list_job_artifacts(job_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[JobArtifactOut]:
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    artifacts = (
        db.query(JobArtifact)
        .filter(JobArtifact.job_id == job.id)
        .order_by(JobArtifact.artifact_key.asc())
        .all()
    )
    return [
        JobArtifactOut(
            id=artifact.id,
            artifact_key=artifact.artifact_key,
            artifact_kind=artifact.artifact_kind,
            file_path=artifact.file_path,
            content_text=artifact.content_text,
            created_at=artifact.created_at,
            updated_at=artifact.updated_at,
        )
        for artifact in artifacts
    ]


def process_job(
    job_id: str,
    page_offset: int,
    auto_toc: bool,
    openai_api_key: str | None = None,
    mathpix_app_id: str | None = None,
    mathpix_app_key: str | None = None,
) -> None:
    db = SessionLocal()
    try:
        job_uuid = UUID(job_id)
        job = db.query(Job).filter(Job.id == job_uuid).first()
        if not job:
            return
        job.status = "running"
        job.error_message = None
        db.commit()

        job_dir = job_root(job_id)
        pdf_path = Path(job.pdf_path)
        if auto_toc:
            chapters_csv = run_auto_toc(
                pdf_path,
                job_id,
                job_dir,
                openai_api_key=openai_api_key,
                mathpix_app_id=mathpix_app_id,
                mathpix_app_key=mathpix_app_key,
            )
            capture_auto_toc_artifacts(db, job)
            job.chapters_csv_path = str(chapters_csv)
            toc_csv = job_dir / "toc.csv"
            if toc_csv.exists():
                job.toc_csv_path = str(toc_csv)
        elif job.chapters_csv_path:
            chapters_csv = Path(job.chapters_csv_path)
        else:
            raise RuntimeError("No chapters CSV provided. Upload a CSV or set auto_toc=true.")

        chapters_dir = job_chapters_dir(job_id)
        effective_offset = 0 if auto_toc else page_offset
        sections, output_paths = split_chapters(pdf_path, chapters_csv, chapters_dir, effective_offset)

        db.query(Chapter).filter(Chapter.job_id == job.id).delete()
        for (title, start_page, end_page), output_path in zip(sections, output_paths):
            stat = output_path.stat()
            db.add(
                Chapter(
                    job_id=job.id,
                    title=title,
                    start_page=start_page,
                    end_page=end_page,
                    filename=output_path.name,
                    file_path=str(output_path),
                    file_size=stat.st_size,
                )
            )

        job.status = "completed"
        job.chapters_dir = str(chapters_dir)
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.query(Job).filter(Job.id == job_uuid).first()
        if job:
            capture_auto_toc_artifacts(db, job)
            job.status = "failed"
            job.error_message = str(exc)
            db.commit()
        logger.exception("Job failed")
    finally:
        db.close()
