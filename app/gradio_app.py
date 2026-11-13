#!/usr/bin/env python3
"""CardioScribe v2 — single-file Gradio UI.

The Python-native alternative to the FastAPI + SPA app. It calls the *same*
backend service layer, so both front ends agree on validation, QC, the single-
GPU lock and what demo mode is allowed to claim.

Gradio over Streamlit for this workload: Streamlit re-runs the whole script on
every widget interaction, which fights a 4-bit model held in VRAM. Gradio loads
the runtime once at startup and `.queue(default_concurrency_limit=1)` gives the
single-GPU serialization the pipeline needs — a second concurrent request while
`lead_saliency()` runs its backward pass is how you OOM a 16 GB card.

    pip install gradio
    python app/gradio_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import gradio as gr  # noqa: E402

from backend import config, runtime_demo, runtime_live, wfdb_io  # noqa: E402
from backend.app import REVIEW_FIELDS, active_mode, save_review as api_save_review  # noqa: E402
from backend.schemas import ReviewRequest  # noqa: E402

TEMPLATES = runtime_live.instruction_templates()
TEMPLATE_LABELS = {t["label"]: t for t in TEMPLATES}


def analyze(files, template_label, use_calibration, sex_label, progress=gr.Progress()):
    """Validate the upload, run the pipeline, return every panel's content."""
    if not files:
        raise gr.Error("Upload both the .hea header and the matching .dat signal file.")

    hea_path = dat_path = None
    for item in files:
        path = Path(getattr(item, "name", str(item)))
        suffix = path.suffix.lower()
        if suffix in wfdb_io.IMAGE_SUFFIXES:
            raise gr.Error(
                f"'{path.name}' is an image or PDF. CardioScribe reads the raw 1D "
                "waveform, not a picture of one — 2D ECG image input is an explicit "
                "non-goal. Upload the WFDB .hea and .dat files."
            )
        if suffix == ".hea":
            hea_path = path
        elif suffix == ".dat":
            dat_path = path

    if hea_path is None or dat_path is None:
        raise gr.Error("Both a .hea file and a .dat file are required.")

    try:
        stem = wfdb_io.validate_filenames(hea_path.name, dat_path.name)
    except wfdb_io.UploadError as exc:
        raise gr.Error(str(exc)) from exc

    progress(0.05, desc="Staging the upload...")
    record = wfdb_io.prepare_record(hea_path.read_bytes(), dat_path.read_bytes(), stem)
    try:
        progress(0.15, desc="Signal QC — 500 Hz, 12 leads, NaN / flat / clipping...")
        try:
            raw, qc = wfdb_io.read_and_qc(record)
        except wfdb_io.UploadError as exc:
            raise gr.Error(str(exc)) from exc

        stages = {"qc": 0.25, "measure": 0.4, "encode": 0.55, "generate": 0.75, "audit": 0.9}

        def on_progress(stage: str, message: str) -> None:
            progress(stages.get(stage, 0.5), desc=message)

        sex = None
        if sex_label and sex_label != "Unspecified":
            sex = sex_label.lower()

        engine = runtime_live if active_mode() == "live" else runtime_demo
        result = engine.run(
            record, raw, qc,
            instruction=TEMPLATE_LABELS.get(template_label, {}).get("instruction"),
            use_calibration=bool(use_calibration),
            sex=sex,
            progress_cb=on_progress,
        )
    finally:
        record.cleanup()

    progress(1.0, desc="Done.")
    return (
        _verdict_md(result),
        _evidence_md(result),
        result.report,
        result.report if result.provenance.report_is_model_output else "",
        _precedent_md(result),
        _provenance_md(result),
        result.record_name,
        result.measurements.get("heart_rate"),
        result.flagged_for_review,
        result.provenance.mode,
    )


def _verdict_md(r) -> str:
    head = ("### ⚠️ FLAGGED FOR CLINICIAN REVIEW" if r.flagged_for_review
            else "### ✅ CONFIDENT DRAFT — READY FOR SIGN-OFF")
    lines = [head, f"`record {r.record_name}` · `abstention ≥ {r.threshold:.3f}` "
                   f"· `{r.elapsed_s:.1f} s` · `mode {r.provenance.mode}`"]
    if r.issues:
        lines.append("")
        lines += [f"- {i}" for i in r.issues]
    if not r.provenance.report_is_model_output:
        lines.append("\n> **The report text is a placeholder, not model output.** "
                     "No trained weights are loaded.")
    return "\n".join(lines)


def _evidence_md(r) -> str:
    out = ["#### Calibrated probabilities"]
    if r.probabilities:
        for name, p in r.probabilities.items():
            mark = " ✅" if p >= r.threshold else ""
            out.append(f"- **{name}** {p * 100:.1f}%{mark}")
    else:
        out.append("_Withheld — no classifier is loaded._")

    out.append("\n#### Deterministic measurements")
    m = r.measurements or {}
    out.append(
        f"- Heart rate: **{m.get('heart_rate') or 'n/a'}** bpm · "
        f"RR mean: **{m.get('rr_mean_ms') or 'n/a'}** ms · "
        f"RR sd: **{m.get('rr_std_ms') or 'n/a'}** ms"
    )
    if r.intervals:
        iv = r.intervals
        out.append(
            f"- PR **{iv.get('pr_ms') or 'n/a'}** · QRS **{iv.get('qrs_ms') or 'n/a'}** · "
            f"QT **{iv.get('qt_ms') or 'n/a'}** · QTc **{iv.get('qtc_bazett_ms') or 'n/a'}** ms · "
            f"axis **{iv.get('axis_deg') or 'n/a'}°**"
        )
        if iv.get("flags"):
            out.append(f"- Interval flags: **{', '.join(iv['flags'])}**")

    out.append("\n#### Lead importance")
    if r.lead_importance:
        top = sorted(r.lead_importance.items(), key=lambda kv: -kv[1])[:6]
        out.append(" · ".join(f"**{k}** {v * 100:.1f}%" for k, v in top))
    else:
        out.append("_Withheld — saliency needs the trained encoder._")

    out.append("\n#### Signal quality")
    out.append(f"{r.qc.sampling_rate} Hz · {r.qc.n_leads} leads · {r.qc.duration_s:.1f} s")
    for w in r.qc.warnings:
        out.append(f"- ⚠️ {w}")
    return "\n".join(out)


def _precedent_md(r) -> str:
    if not r.retrieved:
        return "_Withheld — retrieval needs the PTB-XL training embeddings._"
    return "\n".join(
        f"{i + 1}. `sim {p.similarity:.3f}` · `ecg_id {p.ecg_id}` — {p.report_text}"
        for i, p in enumerate(r.retrieved)
    )


def _provenance_md(r) -> str:
    lines = [
        f"- Mode: **{r.provenance.mode}**",
        f"- Measurements from: **{r.provenance.measurement_source}**",
        f"- Report is model output: **{'yes' if r.provenance.report_is_model_output else 'no'}**",
    ]
    lines += [f"- {n}" for n in r.provenance.notes]
    return "\n".join(lines)


def save_review(record_name, generated, edited, measured_hr, flagged, mode, acc, comp, safe, notes):
    if not record_name:
        return "Run an analysis before saving a review."
    try:
        res = api_save_review(ReviewRequest(
            record_name=record_name,
            generated_report=generated or "",
            edited_report=edited or "",
            measured_hr=measured_hr,
            flagged_for_review=bool(flagged),
            mode=mode or "demo",
            likert_accuracy=int(acc),
            likert_completeness=int(comp),
            likert_safety=int(safe),
            reviewer_notes=notes or "",
        ))
    except Exception as exc:
        return f"Could not save: {exc}"
    return f"Saved — {res.rows} row(s) in `{res.path}`"


def build() -> gr.Blocks:
    mode = active_mode()
    with gr.Blocks(title="CardioScribe v2", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## 🫀 CardioScribe v2 — ECG → audited clinical report\n"
            "> **Research prototype — not a medical device.** Clinician-in-the-loop "
            "decision support only, running entirely on this machine. "
            "*It drafts; a cardiologist decides.*"
        )
        if mode == "demo":
            gr.Markdown(
                "### ⚠️ Demo mode — no trained weights loaded\n"
                "Band-pass, R-peak detection, heart rate and RR variability are really "
                "computed. The report, probabilities, attention, saliency and precedent "
                "retrieval are **withheld, not invented**. Set `CARDIOSCRIBE_RUNTIME_DIR` "
                "and restart for the real pipeline."
            )

        st_record, st_hr = gr.State(""), gr.State(None)
        st_flagged, st_mode = gr.State(False), gr.State(mode)

        with gr.Row():
            with gr.Column(scale=2):
                files = gr.File(label="WFDB pair (.hea + .dat)", file_count="multiple",
                                file_types=[".hea", ".dat"])
            with gr.Column(scale=1):
                template = gr.Dropdown([t["label"] for t in TEMPLATES],
                                       value=TEMPLATES[0]["label"], label="Report template")
                sex = gr.Radio(["Unspecified", "Male", "Female"], value="Unspecified",
                               label="Sex (sets the QTc upper bound: 450 / 460 ms)")
                calib = gr.Checkbox(value=True, label="Calibrated probabilities (ECE 0.082 → 0.022)")
                run = gr.Button("Analyze ECG", variant="primary")

        verdict = gr.Markdown()
        with gr.Row():
            with gr.Column():
                raw_report = gr.Textbox(label="Report (model output)", lines=10, interactive=False)
                evidence = gr.Markdown()
            with gr.Column():
                edited = gr.Textbox(label="Clinician sign-off", lines=10)
                with gr.Row():
                    acc = gr.Slider(1, 5, value=5, step=1, label="Accuracy")
                    comp = gr.Slider(1, 5, value=5, step=1, label="Completeness")
                    safe = gr.Slider(1, 5, value=5, step=1, label="Safety")
                notes = gr.Textbox(label="Reviewer notes", placeholder="optional")
                save = gr.Button("Save to human_review_sheet.csv")
                save_status = gr.Markdown()

        with gr.Row():
            precedent = gr.Markdown(label="Nearest precedent")
            provenance = gr.Markdown(label="Provenance")

        run.click(
            analyze,
            inputs=[files, template, calib, sex],
            outputs=[verdict, evidence, raw_report, edited, precedent, provenance,
                     st_record, st_hr, st_flagged, st_mode],
        )
        save.click(
            save_review,
            inputs=[st_record, raw_report, edited, st_hr, st_flagged, st_mode,
                    acc, comp, safe, notes],
            outputs=[save_status],
        )
    return demo


if __name__ == "__main__":
    app = build()
    # concurrency_limit=1 is the GPU lock: one ECG on the device at a time.
    app.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("CARDIOSCRIBE_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("CARDIOSCRIBE_GRADIO_PORT", 7860)),
        show_api=False,
    )
