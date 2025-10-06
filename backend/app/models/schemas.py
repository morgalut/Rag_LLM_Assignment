from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question")
    k: int = Field(default=8, ge=1, le=50)
    rerank: bool = Field(default=False)

class Citation(BaseModel):
    doc_id: str
    title: str

class AnswerRequest(BaseModel):
    query: str
    k: int = 5
    
class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_context: List[str]
    

class HealthResponse(BaseModel):
    status: str
    vector_extension: bool
    tables_ok: bool
    count_papers: Optional[int] = None
