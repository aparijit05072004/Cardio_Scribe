"""Upload handling, WFDB pairing and signal QC.

`cardioscribe()` calls `wfdb.rdsamp(record_path)` on a record stem with no
extension, which needs BOTH files: the `.hea` header (sampling rate, units,
lead order) and the `.dat` binary. A `.dat` alone cannot be read at all, so the
pairing is enforced before anything touches the GPU.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

from . import config
from .schemas import QCInfo

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}
SAFE_STEM = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class UploadError(ValueError):
    """Rejected before inference. The message is shown to the user verbatim."""


@dataclass
class PreparedRecord:
    directory: Path
    stem: str

    @property
    def record_path(self) -> str:
        """The extension-less stem path `wfdb.rdsamp` expects."""
        return str(self.directory / self.stem)

    def cleanup(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


def validate_filenames(hea_name: str, dat_name: str) -> str:
    """Check the pair before reading any bytes. Returns the shared stem."""
    for name in (hea_name, dat_name):
        if not name:
            raise UploadError("A file was uploaded without a filename.")
        if Path(name).suffix.lower() in IMAGE_SUFFIXES:
            raise UploadError(
                f"'{name}' looks like an image or PDF. CardioScribe reads the raw 1D "
                "waveform, not a picture of one — 2D ECG image input is an explicit "
                "non-goal of the project. Upload the WFDB .hea and .dat files."
            )

    hea_stem, hea_ext = os.path.splitext(os.path.basename(hea_name))
    dat_stem, dat_ext = os.path.splitext(os.path.basename(dat_name))

    if hea_ext.lower() != ".hea":
        raise UploadError(f"Expected a .hea header file, got '{hea_name}'.")
    if dat_ext.lower() != ".dat":
        raise UploadError(f"Expected a .dat signal file, got '{dat_name}'.")
    if hea_stem != dat_stem:
        raise UploadError(
            f"The two files must share a name: got '{hea_stem}.hea' and "
            f"'{dat_stem}.dat'. WFDB finds the signal through the header, so the "
            "stems have to match exactly."
        )
    if not SAFE_STEM.match(hea_stem):
        raise UploadError(
            f"'{hea_stem}' is not a usable record name. Use letters, digits, "
            "dot, dash or underscore only."
        )
    return hea_stem


def prepare_record(hea_bytes: bytes, dat_bytes: bytes, stem: str) -> PreparedRecord:
    """Write the pair into an isolated temp directory.

    Nothing is written inside the repository, and the caller deletes the whole
    directory as soon as inference finishes: no patient waveform is retained.
    """
    if not hea_bytes:
        raise UploadError("The .hea header file is empty.")
    if not dat_bytes:
        raise UploadError("The .dat signal file is empty.")
    total = len(hea_bytes) + len(dat_bytes)
    if total > config.MAX_UPLOAD_BYTES:
        raise UploadError(
            f"Upload is {total / 1e6:.1f} MB, over the "
            f"{config.MAX_UPLOAD_BYTES / 1e6:.0f} MB limit."
        )

    directory = Path(tempfile.mkdtemp(prefix="cardioscribe_upload_"))
    try:
        (directory / f"{stem}.hea").write_bytes(hea_bytes)
        (directory / f"{stem}.dat").write_bytes(dat_bytes)
    except OSError as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise UploadError(f"Could not stage the upload: {exc}") from exc
    return PreparedRecord(directory=directory, stem=stem)


def read_and_qc(record: PreparedRecord) -> tuple[np.ndarray, QCInfo]:
    """Read the record and run the notebook's Phase-1 quality control.

    Returns the raw signal as [n_samples, n_leads] in mV.
    """
    try:
        raw, fields = wfdb.rdsamp(record.record_path)
    except Exception as exc:  # wfdb raises a wide variety of errors
        raise UploadError(
            f"WFDB could not read this record: {exc}. The .hea header must "
            "describe the .dat file that came with it."
        ) from exc

    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2:
        raise UploadError(f"Expected a 2D signal, got shape {raw.shape}.")

    n_samples, n_leads = raw.shape
    fs = int(fields.get("fs") or 0)
    units = (fields.get("units") or ["mV"])[0]

    if n_leads != config.NUM_LEADS:
        raise UploadError(
            f"Expected {config.NUM_LEADS} leads, got {n_leads}. CardioScribe is "
            "trained on standard 12-lead records."
        )
    if fs != config.SAMPLING_RATE:
        raise UploadError(
            f"Expected {config.SAMPLING_RATE} Hz, got {fs} Hz. The pipeline reads "
            "PTB-XL at its native 500 Hz and does not resample on upload; "
            "resample with an anti-aliased filter first if you must."
        )
    if n_samples < config.MIN_SECONDS * fs:
        raise UploadError(
            f"Signal is {n_samples / fs:.1f} s. At least {config.MIN_SECONDS:.0f} s "
            "is required; the model expects a 10 s strip."
        )
    if not np.all(np.isfinite(raw)):
        raise UploadError(
            "Signal QC failed: the waveform contains NaN or infinite samples."
        )

    warnings: list[str] = []
    std = raw.std(axis=0)
    flat_leads = [config.LEAD_NAMES[i] for i in np.where(std < 1e-6)[0]]
    if len(flat_leads) == n_leads:
        raise UploadError("Signal QC failed: every lead is flat (zero variance).")
    if flat_leads:
        warnings.append(
            f"Flat (zero-variance) lead(s): {', '.join(flat_leads)}. "
            "Robustness testing showed macro-AUROC falls ~0.036 with 3 leads off."
        )

    clipped_mask = np.abs(raw).max(axis=0) >= config.CLIP_THRESHOLD_MV
    clipped_leads = [config.LEAD_NAMES[i] for i in np.where(clipped_mask)[0]]
    if clipped_leads:
        warnings.append(
            f"Possible ADC clipping (|x| >= {config.CLIP_THRESHOLD_MV} mV) in: "
            f"{', '.join(clipped_leads)}."
        )

    if n_samples != config.SEQ_LEN:
        warnings.append(
            f"Signal is {n_samples} samples; it will be cropped or padded to "
            f"{config.SEQ_LEN} ({config.SEQ_LEN / fs:.0f} s)."
        )

    qc = QCInfo(
        sampling_rate=fs,
        n_leads=n_leads,
        n_samples=n_samples,
        duration_s=round(n_samples / fs, 3),
        units=units,
        clipped=bool(clipped_leads),
        clipped_leads=clipped_leads,
        flat_leads=flat_leads,
        warnings=warnings,
    )
    return raw, qc


# --- deterministic signal path (mirrors notebook sections 4a-4c) ------------

def bandpass(raw: np.ndarray, fs: int = config.SAMPLING_RATE) -> np.ndarray:
    """4th-order Butterworth 0.5-40 Hz in SOS form, zero-phase.

    Identical in form to the notebook: SOS for numerical stability at 4th
    order, `filtfilt` so QRS timing is not shifted.
    """
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, raw, axis=0)


def pad_or_crop(sig: np.ndarray, length: int = config.SEQ_LEN) -> np.ndarray:
    if sig.shape[0] == length:
        return sig
    if sig.shape[0] > length:
        return sig[:length]
    pad = np.zeros((length - sig.shape[0], sig.shape[1]), dtype=sig.dtype)
    return np.vstack([sig, pad])
