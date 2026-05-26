"""
Async job queue for managing processing jobs.
Uses Python's threading + queue module for safe concurrent access
from both the processing pipeline and the Gradio UI.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from backend.data.schemas import ProcessingJob, ProcessingStatus
from backend.utils.config import AppConfig, get_config

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Thread-safe progress counter shared between the processor and the UI.
    """

    def __init__(self, total: int = 0) -> None:
        self._lock = threading.Lock()
        self.total: int = total
        self.processed: int = 0
        self.failed: int = 0
        self.current_file: str = ""
        self.status: str = "idle"
        self.log_messages: List[str] = []
        self._start_time: Optional[float] = None

    def start(self, total: int) -> None:
        with self._lock:
            self.total = total
            self.processed = 0
            self.failed = 0
            self.status = "running"
            self._start_time = time.time()

    def update(self, file: str, success: bool = True) -> None:
        with self._lock:
            self.current_file = file
            if success:
                self.processed += 1
            else:
                self.failed += 1

    def finish(self) -> None:
        with self._lock:
            self.status = "completed"

    def fail(self, error: str = "") -> None:
        with self._lock:
            self.status = "failed"
            if error:
                self.log_messages.append(f"ERROR: {error}")

    def cancel(self) -> None:
        with self._lock:
            self.status = "cancelled"

    def log(self, message: str) -> None:
        with self._lock:
            self.log_messages.append(message)
            # Keep last 500 log messages to avoid unbounded growth
            if len(self.log_messages) > 500:
                self.log_messages = self.log_messages[-500:]

    @property
    def fraction(self) -> float:
        with self._lock:
            if self.total == 0:
                return 0.0
            return min(1.0, (self.processed + self.failed) / self.total)

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def eta_seconds(self) -> Optional[float]:
        done = self.processed + self.failed
        if done == 0 or self._start_time is None:
            return None
        rate = done / self.elapsed_seconds
        if rate < 1e-6:
            return None
        return (self.total - done) / rate

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            # Compute fraction inline to avoid re-acquiring the non-reentrant lock
            fraction = (
                min(1.0, (self.processed + self.failed) / self.total)
                if self.total > 0
                else 0.0
            )
            return {
                "total": self.total,
                "processed": self.processed,
                "failed": self.failed,
                "fraction": fraction,
                "current_file": self.current_file,
                "status": self.status,
                "elapsed": round(self.elapsed_seconds, 1),
                "eta": round(self.eta_seconds or 0, 1),
                "log": list(self.log_messages[-50:]),
            }


class JobQueue:
    """
    Single-threaded job executor with a thread-safe queue.

    Only one processing job runs at a time (GPU resources are exclusive).
    The queue can hold multiple pending jobs.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.cfg = config or get_config()
        self._queue: queue.Queue = queue.Queue()
        self._current_job: Optional[ProcessingJob] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.progress = ProgressTracker()
        self._running = False

    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="job-queue-worker"
        )
        self._worker_thread.start()
        logger.info("Job queue worker started.")

    def stop(self) -> None:
        """Signal the worker to stop after the current job finishes."""
        self._stop_event.set()
        self._running = False
        logger.info("Job queue stop requested.")

    def submit(
        self,
        input_folder: str,
        output_folder: str,
        file_mode: str = "copy",
        processor_fn: Optional[Callable] = None,
    ) -> str:
        """
        Enqueue a new processing job.
        Returns the job_id.
        """
        job_id = str(uuid.uuid4())[:8]
        job = ProcessingJob(
            job_id=job_id,
            input_folder=input_folder,
            output_folder=output_folder,
            file_mode=file_mode,  # type: ignore[arg-type]
            status=ProcessingStatus.PENDING,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._queue.put((job, processor_fn))
        logger.info("Job %s queued: %s → %s", job_id, input_folder, output_folder)
        return job_id

    def cancel_current(self) -> None:
        """Cancel the currently running job (best-effort)."""
        self.progress.cancel()

    def _worker_loop(self) -> None:
        """Main loop: dequeue and execute jobs sequentially."""
        while not self._stop_event.is_set():
            try:
                job, fn = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._current_job = job
            logger.info("Starting job %s", job.job_id)

            try:
                if fn is not None:
                    fn(job, self.progress)
                else:
                    logger.warning(
                        "Job %s has no processor function; skipping.", job.job_id
                    )
            except Exception as exc:
                logger.exception("Job %s failed: %s", job.job_id, exc)
                self.progress.fail(str(exc))
            finally:
                self._queue.task_done()
                self._current_job = None

        logger.info("Job queue worker stopped.")

    @property
    def is_busy(self) -> bool:
        return self._current_job is not None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()
