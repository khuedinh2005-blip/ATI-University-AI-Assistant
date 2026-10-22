"""
Chat Route for University AI Assistant.
Handles student questions, temporal-aware retrieval, and grounded RAG answer generation.
"""

from datetime import date
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.vector_db import vector_db_service
from backend.services.temporal_ranker import rank_and_filter_chunks, parse_date
from backend.services.rag_engine import generate_answer
from backend.config import settings

router = APIRouter(prefix="", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Student question")
    current_date: Optional[str] = Field(None, description="Reference date (YYYY-MM-DD) for validity checking")
    api_key: Optional[str] = Field(None, description="Optional override for Gemini API Key")


class SourceDocument(BaseModel):
    doc_id: str
    title: str
    category: str
    publication_date: str
    expiration_date: Optional[str] = None
    source_url: str
    similarity_score: float
    combined_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    retrieved_chunks_count: int
    expired_filtered_count: int
    meets_threshold: bool
    is_simulated: bool


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """
    Core RAG endpoint:
    1. Query ChromaDB for Top-20 chunks by cosine similarity.
    2. Filter out expired documents (validity filtering).
    3. Re-rank remaining chunks by recency boost.
    4. Check relevance threshold.
    5. Generate grounded answer via Gemini 2.5 Flash.
    """
    clean_message = payload.message.strip()
    if not clean_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Reference date for temporal filtering
    ref_date = parse_date(payload.current_date) if payload.current_date else date.today()

    # Step 1: Query ChromaDB for top-K chunks
    raw_chunks = vector_db_service.query_chunks(
        query_text=clean_message,
        n_results=settings.TOP_K_CHUNKS
    )

    if not raw_chunks:
        return ChatResponse(
            answer="Cơ sở dữ liệu thông báo hiện đang trống. Vui lòng nhấn nút Đồng bộ (Sync) trong bảng quản trị để cập nhật dữ liệu.",
            sources=[],
            retrieved_chunks_count=0,
            expired_filtered_count=0,
            meets_threshold=False,
            is_simulated=False
        )

    # Step 2 & 3: Temporal filtering & recency re-ranking
    selected_chunks, expired_chunks, meets_threshold = rank_and_filter_chunks(
        chunks=raw_chunks,
        current_date=ref_date,
        top_n=settings.TOP_N_RERANKED,
        threshold=settings.SIMILARITY_THRESHOLD
    )

    # Step 4: Fallback check if no valid chunk meets similarity threshold
    if not meets_threshold or not selected_chunks:
        fallback_msg = (
            "Hiện tại hệ thống không tìm thấy thông tin chính thức phù hợp với câu hỏi của bạn "
            "trong cơ sở dữ liệu các thông báo của nhà trường.\n\n"
            "Vui lòng kiểm tra lại câu hỏi hoặc liên hệ trực tiếp **Văn phòng Khoa CNTT (Phòng 201 Nhà C)** "
            "hoặc cổng Quản lý đào tạo (https://qldt.hanu.edu.vn) để được hỗ trợ cụ thể."
        )
        return ChatResponse(
            answer=fallback_msg,
            sources=[],
            retrieved_chunks_count=len(raw_chunks),
            expired_filtered_count=len(expired_chunks),
            meets_threshold=False,
            is_simulated=False
        )

    # Step 5: Answer generation with Gemini 2.5 Flash
    generation_result = generate_answer(
        question=clean_message,
        chunks=selected_chunks,
        api_key=payload.api_key
    )

    return ChatResponse(
        answer=generation_result["answer"],
        sources=generation_result["sources"],
        retrieved_chunks_count=len(raw_chunks),
        expired_filtered_count=len(expired_chunks),
        meets_threshold=True,
        is_simulated=generation_result.get("is_simulated", False)
    )
