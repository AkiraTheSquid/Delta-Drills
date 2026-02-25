from __future__ import annotations

import logging
import shutil
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import create_access_token, get_current_user, hash_password, verify_password
from app.db import SessionLocal, get_db
from app.models import Chapter, Job, User
from app.practice_router import router as practice_router
from app.processing import run_auto_toc, split_chapters
from app.questions import load_questions
from app.schemas import ChapterOut, JobOut, Token, UserCreate, UserLogin
from app.storage import job_chapters_dir, job_input_path, job_root


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Split Tool Backend", version="0.1.0")

# ---------------------------------------------------------------------------
# Practice question bank — load CSV into memory on startup
# ---------------------------------------------------------------------------
load_questions()

# ---------------------------------------------------------------------------
# Practice endpoints (adaptive learning)
# ---------------------------------------------------------------------------
app.include_router(practice_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def job_to_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error_message=job.error_message,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/signup", response_model=Token)
def signup(payload: UserCreate, db: Session = Depends(get_db)) -> Token:
    email = payload.email.lower()
    logger.info("signup attempt email=%s password_len=%s", email, len(payload.password or ""))
    user = User(email=email, password_hash=hash_password(payload.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        logger.info("signup failed: email already registered email=%s", email)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    db.refresh(user)
    token = create_access_token(str(user.id))
    logger.info("signup success email=%s user_id=%s", email, user.id)
    return Token(access_token=token)


@app.post("/auth/login", response_model=Token)
def login(payload: UserLogin, db: Session = Depends(get_db)) -> Token:
    email = payload.email.lower()
    logger.info("login attempt email=%s password_len=%s", email, len(payload.password or ""))
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        logger.info("login failed email=%s user_found=%s", email, bool(user))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(str(user.id))
    logger.info("login success email=%s user_id=%s", email, user.id)
    return Token(access_token=token)


@app.post("/jobs", response_model=JobOut)
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


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> JobOut:
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job_to_out(job)


@app.get("/jobs/{job_id}/chapters", response_model=list[ChapterOut])
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


@app.get("/jobs/{job_id}/chapters/{chapter_id}/download")
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
                job_dir,
                openai_api_key=openai_api_key,
                mathpix_app_id=mathpix_app_id,
                mathpix_app_key=mathpix_app_key,
            )
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
            job.status = "failed"
            job.error_message = str(exc)
            db.commit()
        logger.exception("Job failed")
    finally:
        db.close()
