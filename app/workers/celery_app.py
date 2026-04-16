"""
Celery application configuration.
"""

from celery import Celery
from app.utils.config import get_settings

settings = get_settings()

celery_app = Celery(
    "genesys",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.training_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 4,  # 4 hours max
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)
