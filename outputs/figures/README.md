# 🖼️ Figure index

Every figure here was extracted directly from the executed cells of
[`../../CardioScribe_v2.ipynb`](../../CardioScribe_v2.ipynb). Nothing was redrawn, retouched or
regenerated outside the notebook run, so each image is exactly what that run produced.

| # | Figure | Phase / section | What it shows |
|:--|:--|:--|:--|
| 01 | [`01_superclass_distribution.png`](01_superclass_distribution.png) | Phase 1 · §2 | SCP **superclass counts** (NORM / MI / STTC / CD / HYP) across 21,837 records, next to the multi-label **co-occurrence heatmap**. Confirms the task is multi-label, not single-label, and shows the NORM-heavy class imbalance the BCE loss has to live with. |
| 02 | [`02_preprocessing_raw_vs_filtered.png`](02_preprocessing_raw_vs_filtered.png) | Phase 2 · §4c | **Raw vs preprocessed** ECG for one record: 0.5–40 Hz 4th-order zero-phase `sosfiltfilt` band-pass plus dataset-level per-lead standardization. Baseline wander is gone and the QRS morphology survives — the point of keeping the mV scale. |
| 03 | [`03_rpeak_detection_lead_ii.png`](03_rpeak_detection_lead_ii.png) | Phase 2 · §5 | **R-peak detection on lead II** via `neurokit2`. This is the *only* job beat detection has in CardioScribe: heart rate and RR variability. No beat chopping — the whole 10 s strip reaches the encoder, because AF, flutter, AV block and ectopy need it. |
| 04 | [`04_resnet1d_training_curves.png`](04_resnet1d_training_curves.png) | Phase 4 · §10d | **ResNet1D baseline** (Doc 1's arm): train/val loss, validation macro-AUROC against the 0.90 gate line, and per-class confusion counts on fold 10. |
| 05 | [`05_encoder_comparison_resnet_vs_transformer.png`](05_encoder_comparison_resnet_vs_transformer.png) | Phase 4 · §10g | **ResNet1D vs ECGTransformer** head-to-head: loss curves, validation AUROC with the gate, and per-class AUROC. ResNet reaches 0.9307, the wav2vec2-style transformer 0.9169 — the gate passes either way. |
| 06 | [`06_calibration_temperature_scaling.png`](06_calibration_temperature_scaling.png) | Phase 9 · §31 | **Reliability before and after per-class temperature scaling**, fitted on the validation fold only. Expected Calibration Error drops 0.0818 → 0.0221 with macro-AUROC unchanged by construction. This is what makes the `p=` values in the Diagnosis-Driven Prompt mean something. |
| 07 | [`07_risk_coverage_curve.png`](07_risk_coverage_curve.png) | Phase 9 · §32 | **Risk–coverage curve.** Sweeping the confidence threshold trades coverage against primary-diagnosis error. Holding risk at 10 % means answering 48.4 % of ECGs and routing the rest to a human — and it sets the abstention threshold to **0.852**, replacing the guessed 0.35. |
| 08 | [`08_clinical_report_card.png`](08_clinical_report_card.png) | Phase 10 · §40 | **The report card.** One page: lead II with Q-Former cross-attention shaded over the timeline, lead V2 for context, calibrated class probabilities against the threshold, per-lead gradient saliency, measured numbers, the generated narrative, retrieved precedent reports, and the guardrail verdict (*flagged for review*). This is the artifact a cardiologist audits. |

## Regenerating these

They are produced by running the notebook end to end. To re-extract them from a fresh run:

```python
import json, base64, pathlib
nb = json.load(open("CardioScribe_v2.ipynb"))
cells = {11: "01_superclass_distribution", 17: "02_preprocessing_raw_vs_filtered",
         20: "03_rpeak_detection_lead_ii", 46: "04_resnet1d_training_curves",
         49: "05_encoder_comparison_resnet_vs_transformer",
         118: "06_calibration_temperature_scaling", 120: "07_risk_coverage_curve",
         137: "08_clinical_report_card"}
for idx, name in cells.items():
    for out in nb["cells"][idx].get("outputs", []):
        png = out.get("data", {}).get("image/png")
        if png:
            raw = "".join(png) if isinstance(png, list) else png
            pathlib.Path(f"outputs/figures/{name}.png").write_bytes(base64.b64decode(raw))
```
