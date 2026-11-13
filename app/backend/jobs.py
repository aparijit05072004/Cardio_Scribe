"""Job queue with a single-worker GPU lock.

Why a queue at all: one ECG costs roughly 4 s of LLM decoding plus a *backward*
pass for per-lead saliency, on top of a 4-bit model already holding ~6 GB. Two
requests arriving together on one 24 GB card (or a 16 GB T4) is how you get a
CUDA OOM, so exactly one job touches the device at a time and everyone else
waits in line with a visible position.
"""
from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from . import config
from .schemas import AnalysisResult, JobStatus


@dataclass
class Job:
    job_id: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "Waiting for the GPU worker..."
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_mono: float = field(default_factory=time.monotonic)
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    payload: Optional[dict[str, Any]] = None

    @property
    def stage_index(self) -> int:
        try:
            return config.STAGE_IDS.index(self.stage)
        except ValueError:
            return 0

    def to_status(self, queue_position: Optional[int] = None,
                  include_result: bool = False) -> JobStatus:
        return JobStatus(
            job_id=self.job_id,
            status=self.status,              # type: ignore[arg-type]
            stage=self.stage,
            stage_index=self.stage_index,
            n_stages=len(config.STAGE_IDS),
            message=self.message,
            queue_position=queue_position,
            error=self.error,
            created_at=self.created_at,
            result=self.result if include_result else None,
        )


class JobManager:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._order: list[str] = []          # queued ids, for position reporting
        self._worker: Optional[asyncio.Task] = None
        self._runner = None                  # set by app startup

    # --- lifecycle ----------------------------------------------------------
    def start(self, runner) -> None:
        """`runner(job, progress_cb) -> AnalysisResult`, executed in a thread."""
        self._runner = runner
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._work(), name="cardioscribe-gpu-worker")

    async def stop(self) -> None:
        if self._worker and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass

    # --- submission ---------------------------------------------------------
    def submit(self, payload: dict[str, Any]) -> tuple[Job, int]:
        self._evict()
        job = Job(job_id=str(uuid.uuid4()), payload=payload)
        self._jobs[job.job_id] = job
        self._order.append(job.job_id)
        position = len(self._order)
        job.message = (
            "Next in line for the GPU worker."
            if position == 1 else
            f"Queued at position {position}; one ECG is processed at a time."
        )
        self._queue.put_nowait(job.job_id)
        return job, position

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def queue_position(self, job_id: str) -> Optional[int]:
        try:
            return self._order.index(job_id) + 1
        except ValueError:
            return None

    @property
    def depth(self) -> int:
        return len(self._order)

    # --- worker -------------------------------------------------------------
    async def _work(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                self._queue.task_done()
                continue
            if job_id in self._order:
                self._order.remove(job_id)

            job.status = "processing"
            job.stage = "qc"
            job.message = "Acquired the GPU worker; validating the upload..."

            def progress(stage: str, message: str, _job: Job = job) -> None:
                _job.stage = stage
                _job.message = message

            try:
                if self._runner is None:
                    raise RuntimeError("No runner configured for the job manager.")
                result = await loop.run_in_executor(
                    None, self._runner, job, progress
                )
                job.result = result
                job.status = "completed"
                job.stage = "done"
                job.message = (
                    f"Audited report ready in {result.elapsed_s:.1f} s."
                    if result.elapsed_s else "Audited report ready."
                )
            except asyncio.CancelledError:
                job.status = "failed"
                job.stage = "done"
                job.error = "Server shutting down."
                job.message = job.error
                self._queue.task_done()
                raise
            except Exception as exc:
                job.status = "failed"
                job.stage = "done"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = job.error
                # Full trace to the server log only: it can name local paths.
                traceback.print_exc()
            finally:
                if job.payload is not None:
                    job.payload = None       # drop uploaded bytes promptly
                self._queue.task_done()

    # --- retention ----------------------------------------------------------
    def _evict(self) -> None:
        now = time.monotonic()
        stale = [
            jid for jid, j in self._jobs.items()
            if j.status in ("completed", "failed")
            and now - j.created_mono > config.JOB_TTL_SECONDS
        ]
        for jid in stale:
            self._jobs.pop(jid, None)
        while len(self._jobs) > config.MAX_JOBS_RETAINED:
            oldest, job = next(iter(self._jobs.items()))
            if job.status in ("queued", "processing"):
                break
            self._jobs.pop(oldest, None)


manager = JobManager()
