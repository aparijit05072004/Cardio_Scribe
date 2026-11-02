<div align="center">

# 🏷️ Model Card — CardioScribe v2

**ECG-to-report research prototype**

`version 2.0` · `PTB-XL 1.0.3` · `research use only`

</div>

> ## ⚠️ Not a medical device
> No regulatory clearance has been sought or obtained. This system must not be used
> for clinical decision-making. It drafts; a cardiologist decides.

---

## 1. Model details

| | |
|:--|:--|
| **Name** | CardioScribe v2 |
| **Type** | Multimodal signal-language model — 1D ECG signal in, structured English clinical report out |
| **Developed by** | Aparijit Chakraborty |
| **Version / date** | 2.0 · October 2026 |
| **License** | MIT (code). Data and pretrained models carry their own licences — see [LICENSE](../LICENSE) |
| **Repository** | [`aparijit05072004/CardioScribe-v2`](https://github.com/aparijit05072004/CardioScribe-v2) |

### Architecture

```
12-lead ECG (10 s @ 500 Hz, 12 × 5000, mV)
  → 0.5–40 Hz 4th-order zero-phase SOS band-pass
  → dataset-level per-lead standardization (train-fitted, keeps mV scale)
  → frozen wav2vec2-style transformer encoder (D=768)
  → Q-Former bridge (32 learned queries, 4 layers, 8 heads)
  → { multi-label classifier · HR/RR regressor · 32 soft tokens }
  → 4-bit NF4 LLM + LoRA (r=16, α=32, dropout 0.05)
  → output validator (numbers, negations, calibrated confidence)
  → structured report, or flagged for human review
```

A side branch runs R-peak detection (`neurokit2`, lead II) for heart rate and RR variability. These are the **measured numbers** the generated text is checked against.

**Decoder used in the committed run:** `Qwen/Qwen2.5-1.5B-Instruct` (ungated, fits a smaller GPU). Any 7–8B instruct model — Llama-3/3.1-8B-Instruct or Qwen-class — drops in via the `LLM_NAME` constant; the soft-token head re-fits itself to the loaded model's hidden size.

---

## 2. Intended use

### ✅ In scope

* **Research and education** on ECG-to-text generation, multimodal bridging, and clinical hallucination control.
* **Clinician-in-the-loop decision support** in a research setting, with **mandatory human review of every output**.
* A **reproducible baseline** for the ablations it ships: encoder arm, bridge type, prompt grounding, preprocessing.

### ❌ Explicitly out of scope

* Autonomous diagnosis, triage, or **any** clinical decision without a qualified human in the loop.
* Deployment in a care setting, or any use implying regulatory clearance.
* Patient-facing output of any kind.
* Pediatric ECGs, non-standard lead placements, or paced rhythms — none were specifically validated.
* Any input other than a 10-second 12-lead ECG. **2D ECG images are explicitly a non-goal.**

### Intended users

ML researchers, clinical informatics researchers, and cardiologists evaluating assistive drafting — all with the understanding that this is a prototype whose generation stage is **under-trained by design in the committed run**.

---

## 3. Training data

| | |
|:--|:--|
| **Dataset** | [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) — 21,837 clinical 12-lead ECGs, 10 s, 500 Hz, 18,885 patients |
| **Splits** | PTB-XL's own patient-level `strat_fold`: 1–8 train, 9 validation, 10 test. Folds 9 and 10 are human-validated. No patient crosses a split. |
| **Labels** | 5 SCP diagnostic superclasses — NORM, MI, STTC, CD, HYP — multi-label |
| **Text** | PTB-XL's own cardiologist/machine reports: ~63 % German, ~30 % English, ~7 % Swedish or mixed |
| **Text processing** | Boilerplate stripped; German/Swedish translated to English by a ~120-term ECG domain dictionary plus MarianMT (`Helsinki-NLP/opus-mt-de-en`), **locally**; mapped to an 8-field schema |
| **Not used** | MIMIC-IV-ECG (~800k) — requires PhysioNet credentialing, CITI training and a signed DUA. PTB-XL+ (12SL intervals) — not in this snapshot. |

> ⚠️ **The text targets contain machine-translation noise.** Reports are short, mostly German, and translated automatically. Every finding-level metric therefore measures **agreement with a noisy reference**, not agreement with ground truth.

---

## 4. Evaluation

All numbers are **PTB-XL fold 10**, patient-level held out, human-validated. Full detail in [`results/results_summary.json`](../results/results_summary.json) and the [project documentation](PROJECT_DOCUMENTATION.md).

### Encoder and bridge

| Metric | Value |
|:--|--:|
| ResNet1D macro-AUROC (gate arm) | **0.9307** ✅ gate ≥ 0.90 |
| ECGTransformer macro-AUROC (deployed) | 0.9169 |
| Q-Former classifier macro-AUROC | 0.8920 |
| Per class (Stage 1) | NORM 0.946 · MI 0.923 · STTC 0.929 · CD 0.904 · HYP 0.888 |
| Measured-number MAE | HR 9.12 bpm · RR mean 93.1 ms · RR std 48.6 ms |
| ECE after per-class temperature scaling | 0.0818 → **0.0221** |

### Report generation (n = 100)

| Metric | Value | Blueprint target |
|:--|--:|:--|
| Finding micro F1 | 0.643 | beat ResNet+MLP baseline ⚠️ |
| Finding macro F1 | 0.210 | — |
| Hallucination rate | 0.360 | < 0.05 ❌ |
| HR agreement | n/a (stated in 0/100) | ≥ 0.95 ❌ |
| Wrong-label corroboration | 0.800 | low is better ❌ |
| Flagged for human review | 10/100 | — |
| BLEU-4 / ROUGE-L | 0.070 / 0.298 | secondary |

> 🔴 **Critical caveat.** Stage-2 instruction tuning ran **10 optimizer steps** of a planned ~7,200 (`SMOKE_TEST=True`). The decoder is essentially untrained, and every number in this section is a floor rather than a result. Set `CARDIOSCRIBE_SMOKE=0` to run the blueprint schedule.
>
> 🔧 **But not every failure is downstream of that.** Two have causes the training budget would not have touched: the gold `rate` field contained no numbers, so the model could not learn to state a heart rate at any budget; and the hallucination metric scored against the reference only, while the blueprint defines it as unsupported by "the reference **or signal**". Both are addressed in notebook §9a and §42–45 — **written, not yet run**. See [RUN_PLAN.md](RUN_PLAN.md), which is explicit that strict hallucination < 0.05 is likely unreachable against PTB-XL's terse references without the clinician gold set.

### Robustness

Macro-AUROC loss from a clean baseline of 0.8898, perturbing the **raw** signal and pushing it through the real preprocessing chain:

| Perturbation | Worst-case Δ |
|:--|--:|
| Gaussian noise (5–20 dB SNR) | −0.004 |
| 50 Hz mains (≤ 0.2 mV) | −0.002 |
| Baseline wander | −0.002 |
| **Leads off (3)** | **−0.036** |
| **Gain mismatch (×2.0)** | **−0.036** |

The band-pass absorbs mains hum and wander as designed. The real exposures are **missing leads** and **device gain mismatch** — the honest preview of cross-site deployment cost.

### Not evaluated

* **External validation** on a never-seen dataset. The harness exists (§41) and is **deliberately unrun** — a fabricated number would be worse than none.
* **Clinician gold set.** `human_review_sheet.csv` is the intake format for two cardiologists plus an adjudicator; the review has not been performed.
* **Subgroup / fairness analysis** by age, sex, site or device.
* **Preprocessing ablation** (filter band, Symlet-8 wavelet) — implemented, flag off.

---

## 5. Limitations

| Limitation | Detail |
|:--|:--|
| **Under-trained decoder** | Stage 2 is a 10-step smoke test in the committed run. The single most important caveat on this card. |
| **Five superclasses only** | NORM / MI / STTC / CD / HYP. The full 71 SCP statements and most specific rhythms are out of scope. |
| ~~**Numbers are HR and RR only** — PR, QRS, QT/QTc and axis need PTB-XL+~~ | **This was wrong and is corrected in v2.2.** PTB-XL+ is a convenient *source* of vendor 12SL/Glasgow measurements, not a prerequisite: PR, QRS, QT, QTc and the frontal axis are all derivable from the waveform. §46 measures them, verified against synthetic beats whose intervals are exact by construction (residual bias PR −1.4 ± 4.5 ms, QRS −6.8 ± 1.6, QT −10.4 ± 0.8, axis −0.1 ± 0.6°). §47 extends the validator to check all of them; §48 cross-checks against PTB-XL+ when present. *Not yet run on real records.* |
| **Single-lead R-peak detection is fragile** | §5 and §12 detect R-peaks on lead II alone. Under axis deviation lead II's QRS is small and inverted, the detector locks onto the wrong deflection, and every downstream measurement is corrupted — while each lead measured alone is correct. Found while verifying §46, which detects on the 12-lead vector magnitude instead. **The committed HR values in `measured_numbers.npz` were computed the old way**, so some are suspect; §46's detector is the fix and `rpeak_source` records which path was used. |
| **No external validation** | Every number is PTB-XL fold 10. Performance on other devices, populations and sites is unmeasured. |
| **No clinician gold set** | Reference text is PTB-XL's own translated report, not an independent reading. |
| **Supervised encoder pretraining** | Not self-supervised on MIMIC-IV-ECG as the blueprint prefers — MIMIC needs PhysioNet credentialing. |
| **Rule-based finding extraction** | Negation-aware, but misses phrasings outside `FINDING_VOCAB`, which biases both F1 and the hallucination rate. It is also substring-based: "left ventricular hypertrophy" registered as both `lvh` and the generic `hypertrophy`, double-counting one finding on both sides of the comparison. Collapsed symmetrically in §42b. |
| **Translation noise** | Machine-translated German reports as targets *and* as the reference for scoring. |
| **Demographics unaudited** | No subgroup analysis; fairness is unmeasured. |
| **Bridge choice unsettled** | At equal budget the simpler MLP bridge outperforms the Q-Former. |

### Foreseeable failure modes

1. **Corroborating a wrong diagnosis.** At p=0.95 a false injected label is repeated ~80 % of the time. Mitigated — not eliminated — by noisy-label training, the validator and abstention.
2. **Silence on rare findings.** Left-axis deviation, extrasystole and ST-elevation all score F1 0.00. The model does not report what it has not learned, which is safer than inventing it but is still a miss.
3. **Omitted numbers.** Heart rate appears in 0/100 reports, so the strongest numeric guardrail never fires. The cause was upstream of the decoder — no gold target contained a number — and is fixed in §9a; once re-run, §42b also writes the measured value in and reports the model's unaided rate separately, so the repair cannot be mistaken for the model having learned it.
4. **Degradation on lead-off and gain mismatch** — the two conditions most likely to differ across sites and devices.

---

## 6. Safety mechanisms

| Mechanism | Section | What it does |
|:--|:--|:--|
| **Calibrated probabilities** | §31 | Per-class temperature scaling fitted on the validation fold. ECE 0.082 → 0.022, so `p=` values in the prompt mean something. |
| **Data-driven abstention** | §32 | Risk–coverage curve sets the threshold at 0.852 for ≤ 10 % risk — 48.4 % coverage, the rest routed to a human. |
| **Noisy-label DDP training** | §19d | Candidate diagnoses in the training prompt come from the classifier's *own* predictions, so the model learns classifier errors exist. |
| **Output validator** | §22 | Checks stated numbers against measurements (±10 bpm) and detects negation/assertion contradictions. |
| **Evidence gate and repair** *(v2.1, not yet run)* | §42, §42b | Strikes clauses asserting findings the calibrated classifier and the measurements will not support, writes in the measured heart rate, and hedges below the calibrated threshold. It reads the signal only — never the reference — and records every edit, so the difference between the model's text and the shipped text is auditable rather than silent. Note the trade: it refuses to assert what this pipeline cannot check, which costs recall on axis deviation, low voltage, pacing, ectopy and QT prolongation, none of which have an evidence channel here. |
| **Wrong-label stress test** | §27, §34 | Quantifies corroboration of false evidence, rather than assuming grounding helps. |
| **Mandatory human review** | throughout | Failing or low-confidence outputs are flagged, never silently returned. |
| **Grounding and explainability** | §36–38 | Cross-attention over the timeline, per-lead saliency, and retrieved precedent reports make each claim auditable. |

---

## 7. Privacy, ethics and data use

* **PTB-XL is open-access and de-identified**, distributed by PhysioNet under CC-BY 4.0. It is **not redistributed** in this repository.
* **All inference is local.** No signal or report text is sent to an external API — including during translation, which runs MarianMT locally. This is required practice if MIMIC-IV-ECG is ever added, as its DUA forbids sending data to external LLM APIs.
* **No patient data is committed.** `.gitignore` excludes the dataset, all checkpoints and all caches.
* **Clinical overreach is the main ethical risk**, and is addressed by labelling this a research tool with mandatory human review, by abstention rather than forced answers, and by stating every unmeasured dimension on this card.
* **Fairness is unmeasured.** No subgroup analysis has been performed; absence of evidence of bias is not evidence of absence.
* **If used with hospital data**, ethics/IRB approval, consent status, and de-identification of headers and report text (names, IDs, dates) are prerequisites, not options.

---

## 8. Citation

```bibtex
@software{chakraborty2026cardioscribe,
  title  = {CardioScribe v2: An auditable ECG-to-report signal-language model},
  author = {Chakraborty, Aparijit},
  year   = {2026},
  url    = {https://github.com/aparijit05072004/CardioScribe-v2},
  note   = {Research prototype. Not a medical device.}
}

@article{wagner2020ptbxl,
  title   = {PTB-XL, a large publicly available electrocardiography dataset},
  author  = {Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and
             Kreiseler, Dieter and Lunze, Fatima I. and Samek, Wojciech and Schaeffter, Tobias},
  journal = {Scientific Data}, volume = {7}, number = {1}, pages = {154}, year = {2020},
  doi     = {10.1038/s41597-020-0495-6}
}
```

---

<div align="center">

[⬅️ README](../README.md) · [📘 Project documentation](PROJECT_DOCUMENTATION.md) · [✅ Blueprint compliance](BLUEPRINT_COMPLIANCE.md)

**It drafts. A cardiologist decides.**

</div>
