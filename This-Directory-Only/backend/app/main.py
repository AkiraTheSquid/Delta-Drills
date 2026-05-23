from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth_router import router as auth_router
from app.chapters_router import router as chapters_router
from app.jobs_router import router as jobs_router
from app.practice import router as practice_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Split Tool Backend", version="0.1.0")

app.include_router(practice_router)
app.include_router(auth_router)
app.include_router(jobs_router)
app.include_router(chapters_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
