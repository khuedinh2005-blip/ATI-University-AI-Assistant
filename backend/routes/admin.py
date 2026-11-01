"""
Admin & Monitoring Routes for University AI Assistant.
Protected endpoints for triggering ingestion sync, PDF document uploads,
and monitoring knowledge base status.
"""

from datetime import date
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Header, Query, UploadFile, File, Form
from pydantic import BaseModel

from backend.config import settings
from backend.services.vector_db import vector_db_service
from backend.services.pdf_parser import parse_pdf_from_bytes

router = APIRouter(prefix="", tags=["Admin"])


class SyncResponse(BaseModel):
    status: str
    summary: Dict[str, Any]


def verify_admin_key(
    x_admin_key: Optional[str] = Header(None),
    admin_key: Optional[str] = Query(None)
):
    """Verifies that the provided admin key matches configured secret."""
    key = x_admin_key or admin_key
    if not key or key != settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing admin key."
        )


@router.post("/sync", response_model=SyncResponse)
async def trigger_manual_sync(
    x_admin_key: Optional[str] = Header(None),
    admin_key: Optional[str] = Query(None)
):
    """
    Manually triggers incremental ingestion:
    - Fetches latest announcements from University API / Mock Portal / Web Scraper.
    - Compares SHA-256 hash against existing documents.
    - Upserts new/modified announcements into ChromaDB.
    """
    verify_admin_key(x_admin_key, admin_key)
    sync_result = vector_db_service.sync_knowledge_base()
    return SyncResponse(
        status="success",
        summary=sync_result
    )


@router.post("/upload-pdf")
async def upload_pdf_document(
    file: UploadFile = File(..., description="PDF document file to upload"),
    title: Optional[str] = Form(None, description="Optional custom document title"),
    category: Optional[str] = Form(None, description="Optional document category"),
    publication_date: Optional[str] = Form(None, description="Publication date (YYYY-MM-DD)"),
    expiration_date: Optional[str] = Form(None, description="Expiration date (YYYY-MM-DD)"),
    x_admin_key: Optional[str] = Header(None),
    admin_key: Optional[str] = Query(None)
):
    """
    Accepts multipart PDF file upload and optional metadata fields,
    parses the PDF using PyMuPDF (fitz), ingests the document into
    ChromaDB vector database via incremental sync, and returns sync stats.
    Protected by admin key in header or query param.
    """
    verify_admin_key(x_admin_key, admin_key)

    filename = file.filename or "uploaded_document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF files (.pdf) are supported."
        )

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    derived_title = title or filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip().title()
    metadata = {
        "title": derived_title,
        "category": category or "",
        "publication_date": publication_date or date.today().isoformat(),
        "expiration_date": expiration_date or "",
        "source_url": f"upload://{filename}"
    }

    try:
        parsed_docs = parse_pdf_from_bytes(pdf_bytes=pdf_bytes, metadata=metadata)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse PDF document: {str(e)}"
        )

    if not parsed_docs:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any readable text from the uploaded PDF."
        )

    sync_result = vector_db_service.sync_knowledge_base(announcements=parsed_docs)

    return {
        "status": "success",
        "filename": filename,
        "summary": sync_result
    }


@router.get("/status")
async def get_system_status():
    """Returns database size, collection stats, and synchronization logs."""
    stats = vector_db_service.get_stats()
    return {
        "status": "healthy",
        "knowledge_base": stats
    }


@router.get("/health")
async def health_check():
    """Simple ping endpoint for readiness and liveness checks."""
    return {
        "status": "ok",
        "service": "Hanoi University AI Assistant Backend",
        "version": "1.0.0"
    }
