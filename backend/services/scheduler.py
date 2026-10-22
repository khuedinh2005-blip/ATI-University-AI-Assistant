"""
Background Scheduler Service using APScheduler.
Runs periodic synchronization of the University Academic Management announcements.
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from backend.config import settings
from backend.services.vector_db import vector_db_service

logger = logging.getLogger("hanu_scheduler")
scheduler = BackgroundScheduler()

def scheduled_sync_job():
    """Periodic job executed every SYNC_INTERVAL_MINUTES."""
    try:
        logger.info("Executing scheduled knowledge base sync...")
        result = vector_db_service.sync_knowledge_base()
        logger.info(
            f"Scheduled sync complete: new={result['new']}, "
            f"modified={result['modified']}, skipped={result['skipped']}"
        )
    except Exception as e:
        logger.error(f"Error during scheduled sync: {e}")

def start_scheduler():
    """Starts the background scheduler."""
    if not scheduler.running:
        scheduler.add_job(
            scheduled_sync_job,
            "interval",
            minutes=settings.SYNC_INTERVAL_MINUTES,
            id="periodic_hanu_sync",
            replace_existing=True
        )
        scheduler.start()
        logger.info(f"Scheduler started (interval: {settings.SYNC_INTERVAL_MINUTES}m)")

def stop_scheduler():
    """Stops the background scheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler shut down.")
