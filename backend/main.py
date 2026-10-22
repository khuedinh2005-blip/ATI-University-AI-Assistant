"""
FastAPI Main Application Entry Point.
Coordinates:
- API routing (/chat, /sync, /status, /health)
- CORS middleware
- Background scheduler lifecycle (APScheduler)
- Initial knowledge base priming
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.routes.chat import router as chat_router
from backend.routes.admin import router as admin_router
from backend.services.scheduler import start_scheduler, stop_scheduler
from backend.services.vector_db import vector_db_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hanu_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup:
    logger.info("Initializing Hanoi University AI Assistant Backend...")
    # Check if vector DB is empty; if so, trigger initial sync
    current_count = vector_db_service.collection.count()
    if current_count == 0:
        logger.info("Knowledge base is empty. Running initial sync...")
        init_result = vector_db_service.sync_knowledge_base()
        logger.info(f"Initial sync complete: {init_result['new']} new docs ingested.")
    else:
        logger.info(f"Knowledge base ready with {current_count} active chunks.")

    # Start periodic background scheduler
    start_scheduler()
    yield

    # Shutdown:
    logger.info("Shutting down backend services...")
    stop_scheduler()


app = FastAPI(
    title="Hanoi University AI Assistant API",
    description="FastAPI Backend for University AI Assistant (Advanced Topics in IT - Fall 2026)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local Streamlit / Web UI clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(chat_router)
app.include_router(admin_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True
    )
