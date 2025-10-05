from __future__ import annotations
import psycopg
from fastapi import APIRouter, Depends, HTTPException, status
from app.models.schemas import AnswerRequest, AnswerResponse, Citation
from app.db.deps import get_db
from app.db.config import settings

router = APIRouter(prefix="", tags=["qa"])

# Optional DI container (if you have it)
qa_service = None
try:
    from app.container import build_container  # type: ignore
    container = build_container(settings)
    qa_service = container.qa_service
except Exception:
    qa_service = None

@router.post("/answer", response_model=AnswerResponse)
def answer(payload: AnswerRequest, conn: psycopg.Connection = Depends(get_db)):
    if not payload.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query must not be empty.")

    if qa_service is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "Answer service not wired. Provide a DI container exposing `qa_service` "
                "(retrieve→generate→ground) or replace this handler to call your services directly."
            ),
        )

    # Example once DI is available:
    # result = qa_service.answer(query=payload.query, k=payload.k, rerank=payload.rerank)
    # return AnswerResponse(
    #     answer=result.text,
    #     citations=[Citation(doc_id=h.doc_id, title=h.title) for h in result.citations],
    #     retrieved_context=result.contexts,
    # )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="QA service misconfigured.",
    )
