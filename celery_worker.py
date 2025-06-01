# celery_worker.py
import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Replace with your Render Redis internal URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6378/0")

celery_app = Celery(
    'leo_trainer_tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='America/Mexico_City', # Adjust to your timezone if needed
    enable_utc=False,
)

# You might need to import functions that your tasks will call
# For example, if evaluate_interaction is needed in a task
# from evaluator import evaluate_interaction