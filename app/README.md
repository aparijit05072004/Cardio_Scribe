<div align="center">

# 🖥️ CardioScribe v2 — Web Application

**Upload a 12-lead WFDB record → watch it process → get an audited report**

`FastAPI` · `no-build SPA` · `single-GPU queue` · `local inference only`

</div>

> ⚠️ **Research prototype — not a medical device.** Clinician-in-the-loop decision support
> only. Every output is drafted for a qualified human to accept, edit or reject.

---

## Run it in 30 seconds

```bash
pip install -r app/requirements-app.txt
python -m uvicorn backend.app:app --app-dir app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. No Node, no build step, no CDN.

Need a record to try it with? There is no ECG in this repository, so generate a synthetic one:

```bash
python app/tools/make_sample_record.py --out /tmp/demo_ecg --hr 72
```

Then drop `/tmp/demo_ecg.hea` **and** `/tmp/demo_ecg.dat` into the page together.

---

## The two modes

The trained checkpoints are not in this repository and cannot be — `.gitignore` excludes
`*.pt`, `*.npz` and `cardioscribe_lora/`, and precedent retrieval additionally needs the
PTB-XL training metadata with its cached text embeddings. So a fresh clone **cannot** run
`cardioscribe()`. The app handles that honestly rather than pretending otherwise.

| | **DEMO** (default) | **LIVE** |
|:--|:--|:--|
| Needs | nothing but the pip install | GPU + checkpoints + PTB-XL |
| Band-pass, R-peaks, HR, RR | ✅ really computed | ✅ |
| Report, probabilities, attention, saliency, precedent | **withheld** | ✅ |
| Report panel says | "placeholder, not model output" | the generated draft |

**Demo mode never fabricates a clinical report.** Everything derived from the signal is
genuinely computed; everything that would come from a trained model is withheld, and the
`provenance` block in every response states which is which. A demo that emitted
plausible-looking clinical prose unlabelled is precisely the failure the model card forbids.

### Switching to LIVE

```bash
python app/tools/export_runtime.py --out ./runtime     # notebook → importable module

# validate the export without running inference
CARDIOSCRIBE_RUNTIME_DIR=./runtime PYTHONPATH=app \
  python -m backend.runtime_live --self-check

# then serve it
CARDIOSCRIBE_RUNTIME_DIR=./runtime \
  python -m uvicorn backend.app:app --app-dir app --host 127.0.0.1 --port 8000
```

`--self-check` validates that the exported module exports every symbol the adapter needs,
with the right shapes, **without running inference**. Run it before trusting a live
deployment.

> The live adapter is written against contracts read directly out of the notebook — return
> keys, array shapes and dtypes all verified against the source — but it has **never been
> executed against real weights**, because that needs a GPU this repository's CI does not
> have. The self-check is how you close that gap on your own machine.

---

## Why it is shaped this way

Four facts about the pipeline dictate the architecture:

**1 · WFDB needs both files.** `wfdb.rdsamp()` takes a record stem with no extension and
reads the `.dat` *through* the `.hea`. A `.dat` alone cannot be read at all, so the pair is
matched by filename stem in the browser, again in the API, and only then written to disk.
2D ECG images are rejected outright — an explicit non-goal of the project.

**2 · One GPU, one job.** A single ECG costs ~3.8 s of LLM decoding (measured: 377 s for
100 reports) plus a *backward* pass for per-lead saliency, on top of a 4-bit model already
holding ~6 GB. Two concurrent requests on a 24 GB card — or a 16 GB T4 — is how you OOM. So
uploads return a `job_id` immediately and exactly one job touches the device at a time,
with a visible queue position. Never a blocking POST.

**3 · The result is structured, not a blob.** `cardioscribe()` returns twelve fields. The
UI renders them natively — interactive waveform, calibrated probabilities against the
abstention threshold, deterministic measurements, saliency, precedent reports, guardrail
verdict — instead of burying them in a PNG.

**4 · Nothing leaves the machine.** Bound to `127.0.0.1`. No CDN, no fonts, no analytics,
no third-party JavaScript at all: the ECG strip is drawn on a raw 2D canvas. Uploads are
written to a temp directory that is deleted the moment inference finishes.

---

## API

| Endpoint | Purpose |
|:--|:--|
| `GET /api/health` | mode, runtime readiness, missing symbols, queue depth |
| `GET /api/templates` | the 5 instruction templates from notebook §9c |
| `POST /api/analyze` | multipart `hea_file` + `dat_file` → `{job_id, queue_position}` |
| `GET /api/jobs/{id}/stream` | Server-Sent Events: 7 pipeline stages, live |
| `GET /api/jobs/{id}` | terminal status plus the full result |
| `POST /api/review` | appends a clinician reading to `human_review_sheet.csv` |

Interactive docs at `/docs`. The response shape is defined once in
[`backend/schemas.py`](backend/schemas.py) and shared by the SPA and the Gradio app.

### Clinician review feeds the gold set

`POST /api/review` appends to the same `human_review_sheet.csv` that notebook §50 scores the
clinician gold set from. Every reading captured through the UI feeds that evaluation
directly — which is the one blueprint step no amount of code can finish.

---

## Configuration

| Variable | Default | Effect |
|:--|:--|:--|
| `CARDIOSCRIBE_RUNTIME_DIR` | *(unset)* | Directory holding `cardioscribe_runtime.py`. Set it to enable LIVE mode |
| `CARDIOSCRIBE_FORCE_DEMO` | `0` | `1` pins demo mode even with a runtime configured |
| `CARDIOSCRIBE_HOST` / `_PORT` | `127.0.0.1` / `8000` | Bind address. Change only inside a trusted network |
| `CARDIOSCRIBE_REVIEW_CSV` | `<repo>/human_review_sheet.csv` | Where reviews are appended |
| `CARDIOSCRIBE_MAX_UPLOAD_MB` | `32` | Upload ceiling |
| `CARDIOSCRIBE_SIGNAL_DECIMATE` | `1` | Send every Nth sample to the browser. `1` keeps the waveform exact (~500 KB) |
| `CARDIOSCRIBE_JOB_TTL` | `3600` | Seconds a finished job is retained |

---

## The Gradio alternative

If you would rather not run a second surface at all:

```bash
pip install gradio
python app/gradio_app.py        # http://127.0.0.1:7860
```

It calls the **same** `backend/` service layer, so validation, QC, the GPU lock and
demo-mode honesty behave identically. Gradio rather than Streamlit because Streamlit re-runs
its script on every widget interaction, which fights a 4-bit model pinned in VRAM;
`queue(default_concurrency_limit=1)` is the same single-GPU serialization the FastAPI worker
implements.

---

## Layout

```
app/
├── backend/
│   ├── config.py        env-driven settings, pipeline stages
│   ├── schemas.py       the API contract — one source of truth
│   ├── wfdb_io.py       pairing, QC, band-pass, temp-dir lifecycle
│   ├── runtime_demo.py  real signal path, withholds model output
│   ├── runtime_live.py  adapter onto cardioscribe(), + --self-check
│   ├── jobs.py          asyncio queue, single-GPU lock, SSE
│   └── app.py           FastAPI: API + SPA from one process
├── web/                 index.html · app.js · styles.css  (no dependencies)
├── gradio_app.py        the Python-native alternative
└── tools/
    ├── export_runtime.py     notebook → cardioscribe_runtime.py
    └── make_sample_record.py synthetic WFDB pair for testing
```

---

## What is verified, and what is not

**Verified end to end** on a machine with no GPU and no weights: upload validation
(image rejection, stem mismatch, QC rails), the job queue under 4 concurrent submits,
SSE streaming, review CSV round-trip, static serving, and the full UI rendered in a real
browser in both modes — including the live adapter driven by a stub runtime that mimics
the notebook's exact contracts, which confirmed the sex-specific QTc bound (450 / 460 ms)
flows correctly from the UI through to `interval_flags`.

**Not verified:** LIVE mode against real weights. See the note above.

---

<div align="center">

[⬅️ Project README](../README.md) · [📘 Documentation](../docs/PROJECT_DOCUMENTATION.md) · [🏷️ Model card](../docs/MODEL_CARD.md)

**It drafts. A cardiologist decides.**

</div>
