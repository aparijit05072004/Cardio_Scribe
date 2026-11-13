"""DEMO runtime — runs the real deterministic signal path, loads no weights.

Why this exists: the trained checkpoints are not in the repository and cannot
be (`.gitignore` excludes `*.pt`, `*.npz`, `cardioscribe_lora/`), and the
retrieval head additionally needs the PTB-XL training metadata and its cached
text embeddings. So a clone of this repo cannot run `cardioscribe()`. Without a
demo path the UI would be unreviewable by anyone but its author.

The honesty rule this module follows: everything derived from the *signal* is
really computed — band-pass, R-peaks, heart rate, RR variability. Everything
that would otherwise come from a *trained model* is withheld, not invented. The
report is an explicit placeholder, never clinical prose, and `Provenance` says
so in the payload so the UI cannot present it as a draft report.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from . import config, wfdb_io
from .schemas import AnalysisResult, Diagnosis, Provenance, QCInfo

ProgressCb = Callable[[str, str], None]

PLACEHOLDER_REPORT = """DEMO MODE — THIS IS NOT MODEL OUTPUT.

No trained CardioScribe weights are loaded, so no report has been generated.
The measurements below ARE real: they come from the same 0.5-40 Hz zero-phase
band-pass and R-peak detection the pipeline uses.

To produce an actual report, export the notebook runtime and start the server
with CARDIOSCRIBE_RUNTIME_DIR set. See app/README.md."""

PLACEHOLDER_FIELDS = {
    "rhythm": "(no model loaded)",
    "rate": "(see measured heart rate)",
    "axis": "(no model loaded)",
    "intervals": "(see measurements)",
    "conduction": "(no model loaded)",
    "st_t": "(no model loaded)",
    "hypertrophy": "(no model loaded)",
    "impression": "(no model loaded — demo mode)",
}


def _detect_rpeaks(lead_ii: np.ndarray, fs: int) -> np.ndarray:
    """A small, honest R-peak detector for demo mode.

    `neurokit2` is used instead when installed, because that is what the
    notebook uses; this fallback keeps the demo dependency-light. It is a
    Pan-Tompkins-shaped pipeline: band-pass is already applied upstream, then
    differentiate, square, integrate over ~120 ms, and take peaks above a
    fraction of the window's robust maximum, with a physiological refractory
    period.
    """
    x = np.asarray(lead_ii, dtype=float)
    if x.size < fs:
        return np.array([], dtype=int)

    diff = np.ediff1d(x, to_begin=0.0)
    squared = diff ** 2
    win = max(1, int(0.12 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    if not np.any(integrated > 0):
        return np.array([], dtype=int)

    from scipy.signal import find_peaks
    threshold = 0.35 * np.percentile(integrated, 99.0)
    peaks, _ = find_peaks(
        integrated,
        height=max(threshold, 1e-12),
        distance=int(0.25 * fs),        # 250 ms refractory -> max 240 bpm
    )
    # Snap each detection onto the true local extremum of the filtered signal.
    snapped = []
    half = int(0.05 * fs)
    for p in peaks:
        lo, hi = max(0, p - half), min(len(x), p + half + 1)
        seg = x[lo:hi]
        snapped.append(lo + int(np.argmax(np.abs(seg))))
    return np.unique(np.asarray(snapped, dtype=int))


def measure_from_signal(filtered: np.ndarray, fs: int) -> tuple[dict, str]:
    """Heart rate and RR variability from lead II. Returns (measurements, source)."""
    lead_ii = filtered[:, 1]
    source = "demo-detector"
    rpeaks: np.ndarray

    try:
        import neurokit2 as nk  # noqa: F401  (optional)
        _, info = nk.ecg_peaks(lead_ii, sampling_rate=fs)
        rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
        source = "neurokit2"
    except Exception:
        rpeaks = _detect_rpeaks(lead_ii, fs)

    measurements: dict[str, Optional[float]] = {
        "heart_rate": None, "rr_mean_ms": None, "rr_std_ms": None,
    }
    if rpeaks.size >= 3:
        rr_ms = np.diff(rpeaks) / fs * 1000.0
        rr_ms = rr_ms[(rr_ms > 200) & (rr_ms < 3000)]      # drop impossible intervals
        if rr_ms.size:
            measurements["heart_rate"] = round(float(60000.0 / np.median(rr_ms)), 1)
            measurements["rr_mean_ms"] = round(float(np.mean(rr_ms)), 1)
            measurements["rr_std_ms"] = round(float(np.std(rr_ms)), 1)
    return measurements, source


def _standardize(filtered: np.ndarray) -> np.ndarray:
    """Stand-in for the dataset-level per-lead statistics.

    The real pipeline divides by statistics fitted on PTB-XL folds 1-8, which
    are not available without the dataset. Per-lead centring here is for
    DISPLAY ONLY — it is never fed to a model in demo mode, and the UI labels
    the y-axis accordingly.
    """
    centred = filtered - filtered.mean(axis=0, keepdims=True)
    scale = np.maximum(centred.std(axis=0, keepdims=True), 1e-6)
    return centred / scale


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

    def step(stage: str, msg: str) -> None:
        if progress_cb:
            progress_cb(stage, msg)

    step("measure", "Band-pass 0.5-40 Hz (zero-phase SOS) and R-peak detection...")
    filtered = wfdb_io.bandpass(raw, fs=qc.sampling_rate)
    filtered = wfdb_io.pad_or_crop(filtered)
    measurements, source = measure_from_signal(filtered, qc.sampling_rate)

    step("encode", "Demo mode: no encoder loaded, so no probabilities are produced.")
    step("generate", "Demo mode: no language model loaded, so no report is generated.")
    step("audit", "Packaging the signal and the measurements that were computed...")

    display = _standardize(filtered)
    dec = config.SIGNAL_DECIMATE
    signal_out = np.round(display[::dec], 4).tolist()

    notes = [
        "No trained weights are loaded: this is the demo path.",
        "Band-pass, R-peak detection, heart rate and RR variability are really computed.",
        "No report, no diagnosis probabilities, no attention, no saliency and no "
        "precedent retrieval — all four need the trained model and PTB-XL.",
    ]
    if source == "demo-detector":
        notes.append(
            "R-peaks came from the app's own detector; install neurokit2 to use "
            "the same one as the notebook."
        )
    if qc.n_samples != config.SEQ_LEN:
        notes.append(f"Signal was cropped/padded from {qc.n_samples} to {config.SEQ_LEN} samples.")

    return AnalysisResult(
        record_name=record.stem,
        provenance=Provenance(
            mode="demo",
            report_is_model_output=False,
            probabilities_are_model_output=False,
            measurement_source=source,
            notes=notes,
        ),
        qc=qc,
        report=PLACEHOLDER_REPORT,
        report_fields=dict(PLACEHOLDER_FIELDS),
        diagnoses=[],                 # withheld rather than invented
        probabilities={},
        threshold=config.DEFAULT_UNCERTAIN_THRESHOLD,
        measurements=measurements,
        intervals=None,
        signal=signal_out,
        signal_decimation=dec,
        lead_names=list(config.LEAD_NAMES),
        sampling_rate=qc.sampling_rate,
        attention=None,
        lead_importance={},
        retrieved=[],
        confident=False,
        issues=["Demo mode: no model output to validate."],
        flagged_for_review=True,
        repair_actions=[],
        elapsed_s=round(time.time() - started, 3),
    )
