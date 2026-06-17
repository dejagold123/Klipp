"""
RQ worker entrypoint.

Run with:
    python -m app.worker

This listens on the "klipp" queue and processes jobs with
app.processor.process_job.
"""

from rq import Worker

from app.queue import QUEUE_NAME, get_raw_redis_connection

if __name__ == "__main__":
    conn = get_raw_redis_connection()
    worker = Worker([QUEUE_NAME], connection=conn)
    worker.work()
