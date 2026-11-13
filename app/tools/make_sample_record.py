#!/usr/bin/env python3
"""Write a synthetic 10 s, 500 Hz, 12-lead WFDB record for testing the app.

This is a SIGNAL GENERATOR for exercising the upload path — not an ECG of any
person, and not clinically meaningful. It builds compact-support P, QRS and T
lobes at a chosen heart rate and projects them onto the six limb leads by
frontal axis, which is enough for R-peak detection and the UI to have something
honest to draw.

    python app/tools/make_sample_record.py --out /tmp/demo_ecg --hr 72
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import wfdb

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LIMB_ANGLES = {"I": 0.0, "II": 60.0, "III": 120.0,
               "aVR": -150.0, "aVL": -30.0, "aVF": 90.0}


def _lobe(out: np.ndarray, start: float, width: float, amp: float) -> None:
    a, b = max(int(round(start)), 0), min(int(round(start + width)), len(out))
    if b > a:
        out[a:b] += amp * np.sin(np.linspace(0, np.pi, b - a))


def synth_12lead(hr=72.0, pr_ms=160.0, qrs_ms=100.0, qt_ms=380.0,
                 axis_deg=60.0, dur_s=10.0, fs=500, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur_s * fs)
    ms = fs / 1000.0
    rr_n = (60.0 / hr) * fs

    base = np.zeros(n)
    beat = 0
    while True:
        r = beat * rr_n + 0.5 * fs
        if r >= n:
            break
        _lobe(base, r - pr_ms * ms, 80 * ms, 0.12)                  # P
        _lobe(base, r - qrs_ms * ms / 2, qrs_ms * ms, 1.0)          # QRS
        _lobe(base, r + qt_ms * ms * 0.45, qt_ms * ms * 0.45, 0.22)  # T
        beat += 1

    sig = np.zeros((n, 12))
    a = np.radians(axis_deg)
    for lead, ang in LIMB_ANGLES.items():
        sig[:, LEAD_NAMES.index(lead)] = base * np.cos(a - np.radians(ang))
    for i, lead in enumerate(["V1", "V2", "V3", "V4", "V5", "V6"]):
        sig[:, LEAD_NAMES.index(lead)] = base * (0.4 + 0.25 * i)

    sig += rng.normal(0, noise, sig.shape)
    # gentle baseline wander, so the band-pass has something to remove
    t = np.arange(n) / fs
    sig += 0.05 * np.sin(2 * np.pi * 0.25 * t)[:, None]
    return sig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output stem, e.g. /tmp/demo_ecg")
    p.add_argument("--hr", type=float, default=72.0)
    p.add_argument("--axis", type=float, default=60.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--fs", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    sig = synth_12lead(hr=args.hr, axis_deg=args.axis,
                       dur_s=args.seconds, fs=args.fs, seed=args.seed)
    wfdb.wrsamp(
        record_name=out.name,
        fs=args.fs,
        units=["mV"] * 12,
        sig_name=LEAD_NAMES,
        p_signal=sig.astype(np.float64),
        fmt=["16"] * 12,
        adc_gain=[1000.0] * 12,
        baseline=[0] * 12,
        write_dir=str(out.parent),
    )
    print(f"Wrote {out}.hea and {out}.dat "
          f"({sig.shape[0]} samples x {sig.shape[1]} leads @ {args.fs} Hz, ~{args.hr:.0f} bpm)")
    print("SYNTHETIC TEST SIGNAL — not a real ECG, not clinically meaningful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
