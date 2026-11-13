"""LIVE runtime — adapter onto the notebook's `cardioscribe()` pipeline.

This module imports `cardioscribe_runtime`, a plain Python module exported from
the notebook (see app/tools/export_runtime.py). The import is lazy and happens
once, in the single GPU worker thread, so model weights land in VRAM exactly
once for the life of the process.

NOTE ON TESTING: this adapter is written against contracts read directly out of
the notebook (return keys, array shapes and dtypes are all verified), but it has
never been executed against real weights — that needs a GPU, PTB-XL and the
trained checkpoints, none of which exist in this repository. Run

    python -m backend.runtime_live --self-check

on the machine that has them before trusting it.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from . import config, wfdb_io
from .schemas import AnalysisResult, Diagnosis, Provenance, QCInfo, Retrieved

ProgressCb = Callable[[str, str], None]

# Everything the adapter touches on the runtime module. Checked up front so a
# half-exported runtime fails with a precise list instead of an AttributeError
# halfway through a GPU job.
REQUIRED_SYMBOLS = [
    "cardioscribe",            # section 39, the one-call pipeline
    "SAMPLING_RATE", "NUM_LEADS", "SEQ_LEN", "LEAD_NAMES", "SUPERCLASSES",
]
OPTIONAL_SYMBOLS = [
    "UNCERTAIN_THRESHOLD_CALIBRATED",  # section 32; falls back to 0.852
    "INSTRUCTION_TEMPLATES",           # section 9c; falls back to a default list
    "measure_intervals", "interval_flags",   # section 46/47 (v2.2)
    "apply_bandpass",
    "repair_report",                   # section 42b (v2.1)
    "plot_report_card",                # section 40
]

_module: Any = None
_lock = threading.Lock()

SCHEMA_FIELDS = ["rhythm", "rate", "axis", "intervals",
                 "conduction", "st_t", "hypertrophy", "impression"]
_FIELD_RE = re.compile(r"^\s*([a-z_]+)\s*:\s*(.*)$", re.IGNORECASE)


def parse_report_fields(report: str) -> dict[str, str]:
    """Pull the 8-field schema out of generated prose, where it is present.

    The model is trained to emit `field: value` lines, but is not guaranteed to;
    anything unparsed is simply absent rather than guessed at.
    """
    fields: dict[str, str] = {}
    current: Optional[str] = None
    for line in (report or "").splitlines():
        m = _FIELD_RE.match(line)
        if m and m.group(1).lower() in SCHEMA_FIELDS:
            current = m.group(1).lower()
            fields[current] = m.group(2).strip()
        elif current and line.strip():
            fields[current] = (fields[current] + " " + line.strip()).strip()
    return fields


def runtime_dir() -> str:
    return config.RUNTIME_DIR


def is_configured() -> bool:
    return bool(config.RUNTIME_DIR) and not config.FORCE_DEMO


def load(force: bool = False) -> Any:
    """Import `cardioscribe_runtime` once. Raises RuntimeError with a usable message."""
    global _module
    with _lock:
        if _module is not None and not force:
            return _module
        if not config.RUNTIME_DIR:
            raise RuntimeError(
                "CARDIOSCRIBE_RUNTIME_DIR is not set, so there is no live runtime "
                "to import. The app is running in demo mode."
            )
        path = os.path.abspath(os.path.expanduser(config.RUNTIME_DIR))
        if not os.path.isdir(path):
            raise RuntimeError(f"CARDIOSCRIBE_RUNTIME_DIR does not exist: {path}")
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            import cardioscribe_runtime as mod  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"Could not import cardioscribe_runtime from {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        missing = missing_symbols(mod)
        if missing:
            raise RuntimeError(
                "cardioscribe_runtime is missing required symbols: "
                + ", ".join(missing)
                + ". Export them from the notebook; see app/tools/export_runtime.py."
            )
        _module = mod
        return _module


def missing_symbols(mod: Any) -> list[str]:
    return [s for s in REQUIRED_SYMBOLS if not hasattr(mod, s)]


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, 4)


def _threshold(mod: Any) -> float:
    return float(getattr(mod, "UNCERTAIN_THRESHOLD_CALIBRATED",
                         config.DEFAULT_UNCERTAIN_THRESHOLD))


def instruction_templates() -> list[dict[str, str]]:
    """The notebook's 5 prompt templates, or a sane default when unavailable."""
    default = [
        {"name": "full_report", "label": "Full 8-field structured report",
         "instruction": "Given the following 12-lead ECG recording, generate a "
                        "complete structured clinical report."},
        {"name": "impression_only", "label": "Brief clinical impression",
         "instruction": "Given the following 12-lead ECG recording, provide a "
                        "brief clinical impression."},
        {"name": "rhythm_analysis", "label": "Rhythm and rate",
         "instruction": "Analyze the rhythm of this 12-lead ECG recording."},
        {"name": "st_t_analysis", "label": "ST-segment and T-wave",
         "instruction": "Describe any ST-segment or T-wave abnormalities in this "
                        "12-lead ECG."},
        {"name": "finding_query", "label": "Significant findings list",
         "instruction": "List all significant findings from this 12-lead ECG recording."},
    ]
    if not is_configured():
        return default
    try:
        mod = load()
    except RuntimeError:
        return default
    templates = getattr(mod, "INSTRUCTION_TEMPLATES", None)
    if not isinstance(templates, (list, tuple)) or not templates:
        return default
    out = []
    for t in templates:
        if not isinstance(t, dict) or "name" not in t:
            continue
        name = str(t["name"])
        out.append({
            "name": name,
            "label": name.replace("_", " ").title(),
            "instruction": str(t.get("instruction", "")),
        })
    return out or default


def run(
    record: wfdb_io.PreparedRecord,
    raw: np.ndarray,
    qc: QCInfo,
    instruction: Optional[str] = None,
    use_calibration: bool = True,
    sex: Optional[str] = None,
    k_retrieve: int = 3,
    progress_cb: Optional[ProgressCb] = None,
) -> AnalysisResult:
    started = time.time()
    mod = load()

    def step(stage: str, msg: str) -> None:
        if progress_cb:
            progress_cb(stage, msg)

    step("encode", "Frozen encoder + Q-Former, calibrated probabilities...")
    step("generate", "Decoding the report with the 4-bit LLM (~4 s)...")

    res = mod.cardioscribe(
        record.record_path,
        instruction=instruction,
        k_retrieve=k_retrieve,
        use_calibration=use_calibration,
    )

    step("audit", "Validator, saliency and precedent retrieval...")

    lead_names = list(getattr(mod, "LEAD_NAMES", config.LEAD_NAMES))
    report = str(res.get("report", "") or "")
    notes: list[str] = []

    # --- waveform -----------------------------------------------------------
    signal = np.asarray(res.get("signal"), dtype=float)
    if signal.ndim != 2:
        raise RuntimeError(f"runtime returned signal with shape {signal.shape}; expected 2D")
    if signal.shape[0] == len(lead_names) and signal.shape[1] != len(lead_names):
        signal = signal.T                      # tolerate [12, N] orientation
    dec = config.SIGNAL_DECIMATE
    signal_out = np.round(signal[::dec], 4).tolist()

    # --- attention: may legitimately be None (recorder found no weights) ----
    attention_raw = res.get("attention")
    attention_out: Optional[list[float]] = None
    if attention_raw is not None:
        att = np.asarray(attention_raw, dtype=float).ravel()
        if att.size:
            if att.size != signal.shape[0]:
                idx = np.linspace(0, att.size - 1, signal.shape[0])
                att = np.interp(idx, np.arange(att.size), att)
            attention_out = np.round(att[::dec], 8).tolist()
    else:
        notes.append("Cross-attention was unavailable for this record.")

    # --- per-lead saliency --------------------------------------------------
    lead_importance: dict[str, float] = {}
    li = res.get("lead_importance")
    if li is not None:
        arr = np.asarray(li, dtype=float).ravel()
        for i, name in enumerate(lead_names[:arr.size]):
            val = _as_float(arr[i])
            if val is not None:
                lead_importance[name] = val

    # --- diagnoses and probabilities ---------------------------------------
    diagnoses = []
    for item in res.get("diagnoses") or []:
        try:
            label, prob = item
        except (TypeError, ValueError):
            continue
        p = _as_float(prob)
        if p is not None:
            diagnoses.append(Diagnosis(label=str(label), probability=p))

    probabilities = {
        str(k): v for k, v in (
            (k, _as_float(v)) for k, v in (res.get("probabilities") or {}).items()
        ) if v is not None
    }

    # --- retrieved precedents ----------------------------------------------
    retrieved = []
    for item in res.get("retrieved") or []:
        try:
            sim, text, ecg_id = item
        except (TypeError, ValueError):
            continue
        s = _as_float(sim)
        if s is not None:
            retrieved.append(Retrieved(similarity=s, report_text=str(text), ecg_id=int(ecg_id)))

    measurements = {
        str(k): _as_float(v) for k, v in (res.get("measurements") or {}).items()
    }

    # --- optional v2.2 deterministic intervals ------------------------------
    intervals: Optional[dict[str, Any]] = None
    if hasattr(mod, "measure_intervals"):
        step("measure", "Deterministic PR / QRS / QT / QTc / axis (section 46)...")
        try:
            prepped = (mod.apply_bandpass(raw) if hasattr(mod, "apply_bandpass")
                       else wfdb_io.bandpass(raw, fs=qc.sampling_rate))
            iv = mod.measure_intervals(prepped, fs=qc.sampling_rate)
            intervals = {k: _as_float(v) for k, v in iv.items()
                         if isinstance(v, (int, float, np.floating))}
            for k, v in iv.items():
                if isinstance(v, str):
                    intervals[k] = v
            if hasattr(mod, "interval_flags"):
                # A per-record sex from the UI overrides whatever the runtime
                # derived dataset-wide (notebook v2.3 resolve_sex_encoding).
                intervals["flags"] = list(mod.interval_flags(iv, sex=sex))
        except Exception as exc:
            notes.append(f"Interval measurement failed: {type(exc).__name__}: {exc}")
            intervals = None

    issues = [str(i) for i in (res.get("issues") or [])]
    repair_actions = [str(a) for a in (res.get("repair_actions") or [])]
    if res.get("repaired"):
        report = str(res["repaired"])
        notes.append("Report shown is the validator-repaired text; see the audit log.")

    return AnalysisResult(
        record_name=record.stem,
        provenance=Provenance(
            mode="live",
            report_is_model_output=True,
            probabilities_are_model_output=True,
            measurement_source="runtime (neurokit2)",
            notes=notes,
        ),
        qc=qc,
        report=report,
        report_fields=parse_report_fields(report),
        diagnoses=diagnoses,
        probabilities=probabilities,
        threshold=_threshold(mod),
        measurements=measurements,
        intervals=intervals,
        signal=signal_out,
        signal_decimation=dec,
        lead_names=lead_names,
        sampling_rate=int(getattr(mod, "SAMPLING_RATE", qc.sampling_rate)),
        attention=attention_out,
        lead_importance=lead_importance,
        retrieved=retrieved,
        confident=bool(res.get("confident", False)),
        issues=issues,
        flagged_for_review=bool(res.get("flagged_for_review", True)),
        repair_actions=repair_actions,
        elapsed_s=round(time.time() - started, 3),
    )


def self_check() -> int:
    """Validate an exported runtime without running inference. Returns an exit code."""
    print("CardioScribe runtime self-check")
    print(f"  CARDIOSCRIBE_RUNTIME_DIR = {config.RUNTIME_DIR or '(unset)'}")
    if not config.RUNTIME_DIR:
        print("  FAIL: not set. The app would run in demo mode.")
        return 1
    try:
        mod = load(force=True)
    except RuntimeError as exc:
        print(f"  FAIL: {exc}")
        return 1
    print(f"  imported {getattr(mod, '__file__', '?')}")
    for sym in REQUIRED_SYMBOLS:
        print(f"    required  {sym:<34} {'ok' if hasattr(mod, sym) else 'MISSING'}")
    for sym in OPTIONAL_SYMBOLS:
        print(f"    optional  {sym:<34} {'ok' if hasattr(mod, sym) else 'absent'}")

    fs = getattr(mod, "SAMPLING_RATE", None)
    leads = getattr(mod, "LEAD_NAMES", [])
    print(f"  sampling rate {fs} | {len(leads)} leads | threshold {_threshold(mod):.3f}")
    if fs != config.SAMPLING_RATE:
        print(f"  WARNING: runtime is {fs} Hz but the app expects {config.SAMPLING_RATE} Hz.")
    if not callable(getattr(mod, "cardioscribe", None)):
        print("  FAIL: cardioscribe is not callable.")
        return 1
    print("  PASS: the runtime exports everything the adapter needs.")
    print("  (This checks symbols only — it runs no inference.)")
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        raise SystemExit(self_check())
    print(__doc__)
    raise SystemExit(0)
