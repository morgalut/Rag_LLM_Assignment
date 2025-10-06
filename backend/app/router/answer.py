from __future__ import annotations
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import AnswerRequest, AnswerResponse, Citation
from app.db.config import settings

router = APIRouter(prefix="", tags=["qa"])

# DI container (created in main.py during startup, then imported here)
qa_service = None  # set in main.py after container build

@router.post("/answer", response_model=AnswerResponse)
def answer(payload: AnswerRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query must not be empty.")
    if qa_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="QA service not ready")

    result = qa_service.answer(query=payload.query, k=payload.k)
    return AnswerResponse(
        answer=result.text,
        citations=[Citation(doc_id=h.doc_id, title=h.title) for h in result.citations],
        retrieved_context=[c for c in result.contexts],
    )
