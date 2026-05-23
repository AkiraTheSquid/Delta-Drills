from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.db import get_db
from app.models import Chapter, Job, User
from app.schemas import ChapterOut

router = APIRouter(prefix="/jobs/{job_id}", tags=["chapters"])


@router.get("/chapters", response_model=list[ChapterOut])
def list_chapters(job_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[ChapterOut]:
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    chapters = db.query(Chapter).filter(Chapter.job_id == job.id).order_by(Chapter.start_page.asc()).all()
    return [
        ChapterOut(
            id=chapter.id,
            title=chapter.title,
            start_page=chapter.start_page,
            end_page=chapter.end_page,
            filename=chapter.filename,
            file_size=chapter.file_size,
            created_at=chapter.created_at,
        )
        for chapter in chapters
    ]


@router.get("/chapters/{chapter_id}/download")
def download_chapter(
    job_id: UUID,
    chapter_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> FileResponse:
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    chapter = db.query(Chapter).filter(Chapter.id == chapter_id, Chapter.job_id == job.id).first()
    if not chapter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chapter not found")
    path = Path(chapter.file_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return FileResponse(path, filename=chapter.filename)
