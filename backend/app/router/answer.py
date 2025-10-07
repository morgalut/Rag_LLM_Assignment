from __future__ import annotations
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from app.models.schemas import AnswerRequest, AnswerResponse, Citation

router = APIRouter(prefix="", tags=["qa"])

# Set by main.py at startup
qa_service = None
streaming_generator = None  # optional: reference to Ollama generator if available

ABSTAIN_TEXT = (
    "I don't know based on the provided context. "
    "No retrieved passages met the relevance threshold."
)

@router.post("/answer", response_model=AnswerResponse)
def answer(payload: AnswerRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query must not be empty.")
    if qa_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="QA service not ready")

    result = qa_service.answer(query=payload.query, k=payload.k)
    return AnswerResponse(
        answer=result.text,
        citations=[Citation(doc_id=h.doc_id, title=h.title) for h in result.citations],
        retrieved_context=[h.chunk for h in result.citations],
    )

@router.post("/stream")
def stream(payload: AnswerRequest):
    if qa_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="QA service not ready")
    if streaming_generator is None or not hasattr(streaming_generator, "generate_stream"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Streaming not available")

    # Do retrieval first; stream only if we have strong context
    rr = qa_service.retriever.retrieve(payload.query, k=payload.k)
    strong = qa_service._select_strong(rr.hits)  # reuse service policy

    if len(strong) < 2:
        # Stream an abstention message without invoking the LLM
        def _abstain():
            yield ABSTAIN_TEXT
        return StreamingResponse(_abstain(), media_type="text/plain")

    contexts = [h.chunk for h in strong]

    def _gen():
        yield ""  # warm-up for clients that expect an initial chunk
        for chunk in streaming_generator.generate_stream(payload.query, contexts, "doc_id + title"):
            yield chunk

    return StreamingResponse(_gen(), media_type="text/plain")
