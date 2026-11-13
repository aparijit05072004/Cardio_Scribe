"""FastAPI service: one process serving both the JSON API and the UI.

Serving the SPA from the same origin as the API means no CORS dance, no second
process, and nothing fetched from a CDN at runtime — which is what "local
inference only, no patient data leaves the machine" actually requires.
"""
from __future__ import annotations

import asyncio
import csv
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, runtime_demo, runtime_live, wfdb_io
from .jobs import Job, manager
from .schemas import (
    HealthResponse, JobStatus, ReviewRequest, ReviewResponse,
    SubmitResponse, TemplateInfo,
)

REVIEW_FIELDS = [
    "timestamp_utc", "record_name", "mode", "generated_report", "edited_report",
    "measured_hr", "flagged_for_review", "likert_accuracy",
    "likert_completeness", "likert_safety", "reviewer_notes",
]


def active_mode() -> str:
    return "live" if runtime_live.is_configured() else "demo"


def _run_job(job: Job, progress) -> object:
    """Executed in a worker thread — exactly one at a time (see jobs.py)."""
    payload = job.payload or {}
    record = wfdb_io.prepare_record(
        payload["hea_bytes"], payload["dat_bytes"], payload["stem"]
    )
    try:
        progress("qc", "Signal QC — 500 Hz, 12 leads, NaN / flat / clipping...")
        raw, qc = wfdb_io.read_and_qc(record)
        engine = runtime_live if active_mode() == "live" else runtime_demo
        return engine.run(
            record, raw, qc,
            instruction=payload.get("instruction"),
            use_calibration=payload.get("use_calibration", True),
            sex=payload.get("sex"),
            k_retrieve=payload.get("k_retrieve", 3),
            progress_cb=progress,
        )
    finally:
        record.cleanup()          # the waveform never outlives the request


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start(_run_job)
    mode = active_mode()
    print(f"[cardioscribe] mode={mode} | http://{config.HOST}:{config.PORT}")
    if mode == "demo":
        print("[cardioscribe] DEMO: no weights loaded; no report will be generated.")
    yield
    await manager.stop()


app = FastAPI(
    title="CardioScribe v2",
    description="Local ECG-to-report service. Research prototype — not a medical device.",
    version="2.3.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- API --------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    mode = active_mode()
    ready, missing = False, []
    if mode == "live":
        try:
            mod = runtime_live.load()
            missing = runtime_live.missing_symbols(mod)
            ready = not missing
        except RuntimeError as exc:
            missing = [str(exc)]
    return HealthResponse(
        status="ok",
        mode=mode,                       # type: ignore[arg-type]
        runtime_dir=config.RUNTIME_DIR or None,
        runtime_ready=ready,
        missing=missing,
        threshold=config.DEFAULT_UNCERTAIN_THRESHOLD,
        sampling_rate=config.SAMPLING_RATE,
        queue_depth=manager.depth,
        review_csv=str(config.REVIEW_CSV_PATH),
    )


@app.get("/api/templates", response_model=list[TemplateInfo])
def templates() -> list[TemplateInfo]:
    return [TemplateInfo(**t) for t in runtime_live.instruction_templates()]


@app.post("/api/analyze", response_model=SubmitResponse)
async def analyze(
    hea_file: UploadFile = File(..., description="WFDB .hea header"),
    dat_file: UploadFile = File(..., description="WFDB .dat signal"),
    template_name: str = Form("full_report"),
    use_calibration: bool = Form(True),
    sex: Optional[str] = Form(None),
) -> SubmitResponse:
    try:
        stem = wfdb_io.validate_filenames(hea_file.filename or "", dat_file.filename or "")
    except wfdb_io.UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    hea_bytes = await hea_file.read()
    dat_bytes = await dat_file.read()
    if len(hea_bytes) + len(dat_bytes) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {config.MAX_UPLOAD_BYTES / 1e6:.0f} MB.",
        )

    instruction = next(
        (t["instruction"] for t in runtime_live.instruction_templates()
         if t["name"] == template_name),
        None,
    )
    sex_norm = None
    if sex and sex.strip().lower() not in ("", "unspecified", "none"):
        sex_norm = "female" if sex.strip().lower().startswith("f") else "male"

    job, position = manager.submit({
        "hea_bytes": hea_bytes,
        "dat_bytes": dat_bytes,
        "stem": stem,
        "instruction": instruction,
        "use_calibration": use_calibration,
        "sex": sex_norm,
    })
    return SubmitResponse(job_id=job.job_id, status="queued", queue_position=position)


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return job.to_status(manager.queue_position(job_id), include_result=True)


@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str) -> StreamingResponse:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")

    async def events():
        last = None
        # Hard ceiling so a wedged job cannot hold a connection open forever.
        deadline = asyncio.get_running_loop().time() + 900
        while True:
            job = manager.get(job_id)
            if job is None:
                yield 'data: {"status":"failed","error":"Job expired."}\n\n'
                return
            snapshot = job.to_status(manager.queue_position(job_id)).model_dump_json()
            if snapshot != last:
                yield f"data: {snapshot}\n\n"
                last = snapshot
            if job.status in ("completed", "failed"):
                return
            if asyncio.get_running_loop().time() > deadline:
                yield 'data: {"status":"failed","error":"Stream timed out."}\n\n'
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/review", response_model=ReviewResponse)
def save_review(req: ReviewRequest) -> ReviewResponse:
    """Append a clinician reading to `human_review_sheet.csv`.

    This is the intake format notebook section 50 scores the gold set from, so
    every review captured here feeds that evaluation directly.
    """
    path = config.REVIEW_CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "record_name": req.record_name,
        "mode": req.mode,
        "generated_report": req.generated_report,
        "edited_report": req.edited_report,
        "measured_hr": "" if req.measured_hr is None else req.measured_hr,
        "flagged_for_review": req.flagged_for_review,
        "likert_accuracy": req.likert_accuracy,
        "likert_completeness": req.likert_completeness,
        "likert_safety": req.likert_safety,
        "reviewer_notes": req.reviewer_notes,
    }
    try:
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        with path.open("r", encoding="utf-8") as fh:
            rows = max(0, sum(1 for _ in fh) - 1)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write review: {exc}") from exc
    return ReviewResponse(saved=True, path=str(path.resolve()), rows=rows)


# --- UI ---------------------------------------------------------------------

if config.WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(config.WEB_DIR / "index.html"))
