"""
Redis connection and RQ queue setup, shared by the API and the worker.
"""

import os

import redis
from rq import Queue

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
QUEUE_NAME = "klipp"

_redis_conn: redis.Redis | None = None
_raw_redis_conn: redis.Redis | None = None
_queue: Queue | None = None


def get_redis_connection() -> redis.Redis:
    """Connection used for job state (decodes responses to str)."""
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_conn


def get_raw_redis_connection() -> redis.Redis:
    """Connection used by RQ, which needs raw bytes for pickling job data."""
    global _raw_redis_conn
    if _raw_redis_conn is None:
        _raw_redis_conn = redis.from_url(REDIS_URL)
    return _raw_redis_conn


def get_queue() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(QUEUE_NAME, connection=get_raw_redis_connection())
    return _queue
