"""
Admin & Monitoring Routes for University AI Assistant.
Protected endpoints for triggering ingestion sync and monitoring system status.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Header, Query
from pydantic import BaseModel

from backend.config import settings
from backend.services.vector_db import vector_db_service

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
    - Fetches latest announcements from University API / Mock Portal.
    - Compares SHA-256 hash against existing documents.
    - Upserts new/modified announcements into ChromaDB.
    """
    verify_admin_key(x_admin_key, admin_key)
    sync_result = vector_db_service.sync_knowledge_base()
    return SyncResponse(
        status="success",
        summary=sync_result
    )


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
