"""Runtime configuration for the CardioScribe v2 web app.

Everything is environment-driven so the same code runs as a zero-dependency
demo or against the real GPU pipeline, with no edits.
"""
from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent
WEB_DIR = APP_DIR / "web"


def _flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


# --- mode -------------------------------------------------------------------
# LIVE requires a directory holding `cardioscribe_runtime.py` (see
# app/tools/export_runtime.py). Absent, the app runs in DEMO mode, which loads
# no weights and never presents generated text as model output.
RUNTIME_DIR = os.environ.get("CARDIOSCRIBE_RUNTIME_DIR", "").strip()
FORCE_DEMO = _flag("CARDIOSCRIBE_FORCE_DEMO", False)

# --- signal expectations (mirror the notebook's constants) ------------------
SAMPLING_RATE = int(os.environ.get("CARDIOSCRIBE_FS", 500))
NUM_LEADS = 12
SEQ_LEN = 5000                       # 10 s at 500 Hz
CLIP_THRESHOLD_MV = 4.0              # notebook section 1 QC rail
MIN_SECONDS = 5.0                    # shorter strips are rejected outright

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Fallback only. LIVE mode overrides this with the runtime's own calibrated
# value; see notebook section 32 (risk-coverage curve).
DEFAULT_UNCERTAIN_THRESHOLD = 0.852

# --- limits -----------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.environ.get("CARDIOSCRIBE_MAX_UPLOAD_MB", 32)) * 1024 * 1024
MAX_JOBS_RETAINED = int(os.environ.get("CARDIOSCRIBE_MAX_JOBS", 50))
JOB_TTL_SECONDS = int(os.environ.get("CARDIOSCRIBE_JOB_TTL", 3600))

# Waveform JSON payload: 5000 x 12 float is ~1.4 MB. Sending every sample keeps
# the client honest about what the model saw, which matters more than bytes on
# a localhost deployment, so the default is no decimation.
SIGNAL_DECIMATE = max(1, int(os.environ.get("CARDIOSCRIBE_SIGNAL_DECIMATE", 1)))

# --- clinician review -------------------------------------------------------
REVIEW_CSV_PATH = Path(
    os.environ.get("CARDIOSCRIBE_REVIEW_CSV", str(REPO_ROOT / "human_review_sheet.csv"))
).expanduser()

# --- server -----------------------------------------------------------------
# Bound to loopback by default: inference is local-only and no ECG should leave
# the machine. Override deliberately, and only inside a trusted network.
HOST = os.environ.get("CARDIOSCRIBE_HOST", "127.0.0.1")
PORT = int(os.environ.get("CARDIOSCRIBE_PORT", 8000))
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "CARDIOSCRIBE_ALLOWED_ORIGINS",
        f"http://127.0.0.1:{PORT},http://localhost:{PORT}",
    ).split(",") if o.strip()
]

# --- pipeline stages shown in the UI ---------------------------------------
STAGES = [
    ("queued",     "Queued for the single-GPU worker"),
    ("qc",         "Signal QC — 500 Hz, 12 leads, NaN / flat / clipping"),
    ("measure",    "Deterministic measurement — R-peaks, HR, RR, intervals"),
    ("encode",     "Frozen encoder + Q-Former, calibrated probabilities"),
    ("generate",   "4-bit LLM report decoding"),
    ("audit",      "Evidence gate, validator, saliency and precedent lookup"),
    ("done",       "Audited report ready"),
]
STAGE_IDS = [s[0] for s in STAGES]
