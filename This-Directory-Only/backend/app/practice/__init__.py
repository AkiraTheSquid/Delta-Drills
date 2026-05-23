"""
Practice API package.

Exposes a single aggregated `router` that mounts every sub-router under
`/api/practice`. main.py imports this as the practice router.

Endpoints:
  GET  /api/practice/next-question
  POST /api/practice/submit
  POST /api/practice/submit-local-eval
  POST /api/practice/override
  POST /api/practice/feedback
  POST /api/practice/visual-debug
  GET  /api/practice/visual-debug
  GET  /api/practice/subtopics
  PUT  /api/practice/weights
  POST /api/practice/run-code
  POST /api/practice/ai-explanation
  POST /api/practice/ai-judge
"""

from fastapi import APIRouter

from app.practice.ai_router import router as ai_router
from app.practice.arena_rating_router import router as arena_rating_router
from app.practice.feedback_router import router as feedback_router
from app.practice.questions_router import router as questions_router
from app.practice.subtopic_router import router as subtopic_router

router = APIRouter(prefix="/api/practice", tags=["practice"])
router.include_router(questions_router)
router.include_router(feedback_router)
router.include_router(subtopic_router)
router.include_router(ai_router)
router.include_router(arena_rating_router)

__all__ = ["router"]
