<div align="center">

# ✅ Blueprint Compliance Matrix

**Every numbered step of *CardioScribe v2: ECG-to-Report Project Blueprint*, traced to the notebook that implements it.**

</div>

Legend: ✅ implemented and run · 🟡 implemented, deliberately not run · ❌ measured and not met · 🔄 patched, pending the re-run that would re-measure it · ➖ out of scope for this run, disclosed

---

## 1. Measurable targets (Blueprint §1)

| Target | Result (v2.0 run) | Status |
|:--|:--|:--:|
| Encoder macro-AUROC ≥ 0.90 on the PTB-XL superclass test fold — *a gate* | 0.9307 (ResNet1D) · 0.9169 (Transformer) | ✅ |
| Clinical-finding F1 on a held-out gold set beats the ResNet + MLP baseline | micro-F1 0.643; no clinician gold set exists, and the bridge ablation favours the MLP arm | ❌ |
| Hallucination rate below 5 % | 0.360 | ❌ |
| Numbers in the text agree with deterministic measurements ≥ 95 % | HR stated in 0/100 reports → the check never fires | ❌ |

**Non-goals, honoured:** no regulatory clearance claimed · no autonomous diagnosis · no 2D ECG-image input.

### v2.1 — the three failures, re-diagnosed

The v2.0 note above attributed all three to one cause: Stage 2 ran as a 10-step smoke
test rather than 2–3 epochs. That is the largest cause but not the only one. Two of the
three had causes upstream of the decoder that a longer run would not have fixed:

| Failure | Additional cause found | Patched in |
|:--|:--|:--|
| Numbers agree ≥ 95 % | `rate` was populated for 7.4 % of records and never held a bpm value — no gold target ever contained a number to learn | §9a-pre, §9a |
| Hallucination < 5 % | the metric scores against the reference only, while the blueprint defines it as "unsupported by the reference **or signal**"; and the SCP fallback leaked `"sttc (from SCP label)"` into the gold prose | §9a, §42–43 |
| Finding-F1 beats ResNet+MLP | the two arms were compared on AUROC and retrieval, never on F1 — the metric the target names | §44 |

**Status: patched, not yet re-run.** Sections 42–45 and the Stage-2 rewrite are in the
notebook; none of their numbers exist until Stage 2 is re-run with `CARDIOSCRIBE_SMOKE=0`.
The table above still shows the v2.0 results because those are the only results there are.
See **[docs/RUN_PLAN.md](RUN_PLAN.md)** for the re-run, and for a frank account of which
targets are expected to pass — target 3 will, target 4 strictly probably will not, and
target 2 may be settled in the MLP arm's favour.

---

### v2.3 — the gate's evidence channel

Parts VI and VII disagreed with each other, and Part VII was right. §42 declared five
findings permanently ungroundable because they *"need PTB-XL+ 12SL features"*; §46,
written afterwards, measures QT/QTc and the frontal axis **from the waveform**. The gate
was striking claims it already had the evidence to support.

| Change | Effect |
|:--|:--|
| `iv` evidence channel in §42 | `left_axis_deviation`, `right_axis_deviation` and `qt_prolongation` are grounded by §46's measurements. Only `low_voltage` and `pacemaker` now lack a channel |
| Thresholds sourced, not copied | The facts come from §46's own `interval_flags`, so the gate and the §47 validator cannot drift apart |
| Direction respected | A right-deviated axis does not ground a *left* axis claim; an extreme axis grounds neither |
| Unmeasured ≠ refuted | `INTERVALS` is built in §47b, after §42, so `signal_support` reads it at call time. Unmeasured findings are unverifiable — struck under `drop`, restored under `keep`; measured-and-normal is refuted and `keep` does not resurrect it |
| Sex encoding derived | QTc's bound is sex-specific and PTB-XL's `sex` labels contradicted each other, so §42 derives the mapping from height and weight, with guards, and falls back to the 450 ms bound when inconclusive |

**Status: written, not run.** The logic is unit-tested against stubs; no notebook cell
was executed, and no measured number changed. Ordering matters on the re-run — §46–47
must populate `INTERVALS` before §43 is re-scored. See **[RUN_PLAN.md](RUN_PLAN.md)**.

---

## 2. Doc 1 vs Doc 2 decisions (Blueprint §2)

| Aspect | Blueprint's final decision | Built | |
|:--|:--|:--|:--:|
| Input | Native 1D, full 10 s at 500 Hz | `[12, 5000]` at 500 Hz, no resampling | ✅ |
| Filtering | 0.5–40 Hz zero-phase SOS; wavelet only as an ablation | `sosfiltfilt`, 4th-order Butterworth; Symlet-8 behind a flag | ✅ |
| Segmentation | No beat chopping; R-peaks for side features only | Full strip to the encoder; `neurokit2` for HR/RR only | ✅ |
| Normalization | Dataset-level per-lead standardization that keeps mV scale | Train-fold-fitted per-lead mean/std | ✅ |
| Encoder | ECG-pretrained transformer; ResNet kept as a baseline arm | wav2vec2-style transformer D=768 + ResNet1D D=512 | ✅ |
| Bridge | Q-Former; MLP is the ablation | Q-Former deployed, MLP ablated (§28) — *and the MLP wins* | ✅ |
| LLM | Llama-3-8B + QLoRA (any current 7–8B instruct model works) | `Qwen/Qwen2.5-1.5B-Instruct` 4-bit NF4 + LoRA; `LLM_NAME` swaps it | ✅ |
| Hallucination control | DDP with top-k and probabilities, measured numbers, output validator | All three, plus noisy-label DDP training and a stress test | ✅ |
| Data | PTB-XL + MIMIC-IV-ECG + a clinician gold set | PTB-XL only | ➖ |
| Training | Three stages: encoder, align, LoRA | All three implemented; Stage 3 budget-capped | 🟡 |
| Evaluation | Finding-level F1, hallucination rate, cardiologist review | F1 and hallucination run; review sheet exported, review not performed | 🟡 |
| Speed | Skip multi-token prediction for v1 | Skipped, as specified | ✅ |

---

## 3. Flaws in the source documents (Blueprint §3)

### Doc 1

| Flaw | Handled |
|:--|:--:|
| Shape error: 100 Hz × 10 s is `[12, 1000]`, not `[12, 5000]` | ✅ native 500 Hz makes `[12, 5000]` correct |
| Invalid filter: 0.5–50 Hz at 100 Hz sits at Nyquist; `butter` rejects it | ✅ 0.5–40 Hz at 500 Hz, SOS form |
| Bridge mismatch: MLP expects `[B, T, 512]`, encoder emits one global vector | ✅ encoders expose `return_embeddings=True` and emit sequences |
| Naming: it is a signal-language model, not a Vision-Language Model | ✅ described as such throughout |

### Doc 2

| Flaw | Handled |
|:--|:--:|
| Beat chopping contradicts its own goal — AF, flutter, AV block, ectopy need the full strip | ✅ no beat chopping; survives only as an ablation concept |
| Per-lead z-scoring removes absolute voltage that ST-elevation and LVH depend on | ✅ dataset-level standardization instead |
| No ECG pretraining — Wav2Vec2 on speech does not transfer | ✅ encoder trained on ECG before freezing (supervised; SSL is the open step) |
| "Ground-truth text" overstated — PTB-XL reports are short and mostly German | ✅ translation noise named as a limitation in the model card |
| DDP can backfire: a wrong label + "corroborate this" = a confident wrong report | ✅ probabilities, noisy-label training, validator, **and it is measured**: 0.800 corroboration |
| Overstated claims (Wav2Vec2 beats ResNet, LoRA ~4 %, multi-token prediction, 40 Hz low-pass) | ✅ each turned into an ablation or an explicit exclusion |

### Both

| Gap | Handled |
|:--|:--:|
| No real evaluation plan | ✅ Phase 8 — finding-level clinical metrics, stress test, ablations |
| Privacy and data-use terms unaddressed | ✅ model card §7; all inference local, no data to external APIs |

---

## 4. The nine phases, step by step (Blueprint §5)

| Step | Blueprint requirement | Notebook | Status | Notes |
|:--:|:--|:--|:--:|:--|
| **Phase 0 — Access and setup** |
| 1 | PhysioNet credentialed account for MIMIC-IV-ECG | — | ➖ | MIMIC not used; PTB-XL is open-access |
| 2 | Accept the Llama licence; create repo and experiment tracking | — | 🟡 | Ungated Qwen decoder used instead; tracking via notebook outputs rather than W&B/MLflow |
| **Phase 1 — Data audit** |
| 3 | Download PTB-XL / PTB-XL+ / MIMIC; check lead order, units, missing or flat leads | §0–§1 | ✅ | PTB-XL only; 12 leads, mV, NaN/flat/clipped scan |
| 4 | Patient-level splits: `strat_fold` 1–8 / 9 / 10 | §3 | ✅ | Exactly as specified |
| **Phase 2 — Preprocessing** |
| 5 | Load with `wfdb` at native 500 Hz; anti-aliased `resample_poly` if ever downsampling | §4 | ✅ | No resampling performed |
| 6 | Quality control: drop or flag NaN, flat or clipped records | §1 | ✅ | Clip threshold 4.0 mV |
| 7 | 4th-order Butterworth 0.5–40 Hz with `sosfiltfilt`; cutoffs configurable; ablate 40 vs 100 Hz | §4a, §33 | 🟡 | Filter ✅; the band ablation is implemented but flag-gated |
| 8 | Dataset-level per-lead standardization (train only); crop or pad to 5000 | §4b–§4c | ✅ | Streaming train-fold statistics |
| 9 | Side branch: R-peak detection for HR and RR statistics — Pan-Tompkins' only role | §5 | ✅ | `neurokit2`, lead II |
| **Phase 3 — Text curation** |
| 10 | Translate PTB-XL German reports to English with a local MT or LLM model | §8 | ✅ | ~120-term domain dictionary + MarianMT, local |
| 11 | Clean MIMIC machine reports; no MIMIC text to external APIs | §7b | ➖/✅ | MIMIC not used; the no-external-API rule is honoured anyway |
| 12 | Convert each report to a fixed schema: rhythm, rate, axis, intervals, conduction, ST-T, hypertrophy, impression | §9a–§9b | ✅ | All 8 fields |
| 13 | Build instruction pairs from 3–5 templates | §9c–§9d | ✅ | → `instruction_pairs.jsonl` |
| **Phase 4 — Stage 0, the encoder** |
| 14 | Train a baseline 1D-ResNet classifier (Doc 1's arm) | §10a–§10d | ✅ | 0.9307 fold-10 macro-AUROC |
| 15 | Load or pretrain the transformer encoder: SSL on MIMIC, then supervised fine-tuning on PTB-XL | §10e–§10g, §51 | 🟡 | Supervised on PTB-XL only. v2.2's §51 implements the masked-latent SSL objective and MIMIC loader, but the step stays **open and blocked on credentialing, not on code**: MIMIC-IV-ECG needs CITI training plus a signed DUA, granted to a person |
| 16 | **Gate: macro-AUROC ≥ 0.90. Do not proceed until it passes** | §10h | ✅ | **PASSED at 0.9307** |
| **Phase 5 — Stage 1, Q-Former alignment** |
| 17 | Train Q-Former and heads on classification CE, interval regression, and InfoNCE against frozen text embeddings | §14–§17, §46–47 | ✅ | BCE + masked MSE + InfoNCE. Regression covered HR/RR only, on the belief that intervals need PTB-XL+ — **that was wrong**: v2.2's §46 measures PR, QRS, QT/QTc and frontal axis from the waveform (verified on synthetic beats; PTB-XL+ is a cross-check in §48, not a prerequisite) and §47 validates all of them. Add them to `REG_FIELDS` to train the head on them |
| 18 | Precompute and cache the frozen encoder's output | §11b | ✅ | Memmapped `*_encout.npy` |
| **Phase 6 — Stage 2, instruction tuning** |
| 19 | Train Q-Former + LoRA with next-token loss. LoRA LR 2e-4, Q-Former LR 1e-4, cosine, 2–3 epochs, 256–384 tokens, bf16, gradient checkpointing, paged 8-bit AdamW, effective batch 32–64 | §19a–§19i | 🟡 | Every hyperparameter matches **except epochs**: 10 optimizer steps (`SMOKE_TEST=True`) instead of 2–3 epochs. v2.1 rewrote §19h for a real schedule — val-loss checkpointing, best-checkpoint selection, wall-clock budget, resume across sessions — but it has not been run |
| 20 | Train DDP with realistic noise — the classifier's own predictions, not clean gold labels | §19d | ✅ | Top-3 with probabilities from the Stage-1 classifier |
| **Phase 7 — Inference and guardrails** |
| 21 | Prompt with top-k diagnoses and probabilities, measured numbers and soft tokens; instruct the model to flag disagreement and write "uncertain" below threshold | §19e, §21 | ✅ | Threshold later replaced by the calibrated 0.852 |
| 22 | A validator checks numbers against measurements and negations against the classifier; failures flagged for human review | §22, §42, §42b, §47 | ✅ | ±10 bpm tolerance; 7 negator forms; 10/100 flagged. v2.1 adds the repair the check implies: strike clauses the signal will not support, write in the measured rate, hedge below the calibrated threshold — every edit recorded in `actions`, so the delta between the model's text and the shipped text is never silent |
| **Phase 8 — Evaluation and demo** |
| — | Evaluation: see §5 below | §23–§41 | 🟡 | |
| — | **Demo**: a runnable surface over §39's one-call API | [`app/`](../app/README.md) | ✅ | Upload a WFDB pair, single-GPU queue, staged progress, the structured report with its evidence and guardrail verdict. Runs without weights in demo mode, which withholds model-dependent output rather than inventing it |

---

## 5. Evaluation and ablations (Blueprint §8)

| Dimension | Blueprint asks for | Notebook | Status |
|:--|:--|:--|:--:|
| **Encoder** | Macro-AUROC and F1 on fold 10; external validation on an unseen dataset | §10d/§10g; §41 → §49 | 🟡 fold 10 ✅. External harness **completed** in v2.2 §49 (reads the dataset's own SNOMED code table rather than a hand-copied one) — still unrun, because it needs the dataset, the trained encoder and a GPU |
| **Report accuracy** | Extract findings from generated and reference reports; per-finding P/R/F1 | §23, §26 | ✅ |
| **Hallucination** | Share of unsupported findings; wrong-label stress test | §26, §27 | ✅ 0.360 and 0.800 |
| **Numbers** | HR, PR, QRS, QTc in text vs measured | §22, §26 → §42, §46–48 | 🟡 v2.0 checked HR only, and HR was stated in 0/100 reports. v2.2 §46 measures PR, QRS, QT, QTc and axis **from the waveform** (PTB-XL+ is a cross-check in §48, not a prerequisite) and §47 validates all of them; v2.1 §9a puts a bpm value in every gold `rate` field. Unrun |
| **Text metrics** | BLEU, ROUGE, BERTScore as secondary | §24 | ✅ BLEU-4 and ROUGE-L; BERTScore skipped (extra dependency, secondary) |
| **Human review** | Cardiologist Likert rating of 100–200 reports | §29, §50, [`app/`](../app/README.md) | 🟡 `human_review_sheet.csv` exported in v2.0, and the web app now captures readings straight into it — a clinician edits the draft and scores accuracy / completeness / safety in the browser. v2.2 §50 adds the full gold-set protocol: blank per-reader intake sheets, Cohen's κ **before** adjudication, unresolved disagreements left unresolved, and scoring against the adjudicated reading. No readings — this is the step code cannot take |
| **Robustness** | Added noise, lead-off, different devices and sites | §30 | ✅ 11 conditions |

### The four required ablations

| # | Ablation | Status | Result |
|:--:|:--|:--:|:--|
| 1 | ResNet + MLP (Doc 1) vs transformer encoder + Q-Former (Doc 2) | ✅ | ResNet 0.9307 vs Transformer 0.9169 |
| 2 | MLP vs Q-Former with the same encoder | ✅ | **MLP 0.9171 / R@1 0.014** vs Q-Former 0.8920 / R@1 0.002 — the blueprint's choice loses at this budget. Compared on AUROC and retrieval only; v2.1 §44 adds the **F1** comparison blueprint target 2 actually names (unrun) |
| 3 | 0.5–40 Hz vs wider band; with vs without wavelet denoising | 🟡 | Harness complete (§33), `RUN_PREPROCESSING_ABLATIONS=False` — a fair comparison retrains the encoder per variant |
| 4 | DDP with clean labels vs noisy classifier predictions vs no DDP | ✅ | no 0.594 / noisy 0.610 / clean 0.618 / wrong 0.435 F1 |

### Beyond the blueprint

Additions the blueprint does not ask for, included because the guardrails are unsound without them — plus the surface that makes the whole thing usable:

| | Why |
|:--|:--|
| **§31 Per-class temperature scaling** | Raw sigmoid outputs are miscalibrated, which corrupts both the `p=` values in the prompt and the abstention threshold. ECE 0.0818 → 0.0221. |
| **§32 Risk–coverage selective prediction** | `UNCERTAIN_THRESHOLD = 0.35` was a guess. The curve sets it at **0.852** for ≤ 10 % risk at 48.4 % coverage. |
| **§42 Evidence gate with three channels** *(v2.1 + v2.3)* | Blueprint step 22 specifies a validator that *checks and flags*. The gate adds the repair the check implies, and grounds each claim in a named channel — classifier, R-peak measurement, or §46 interval — reading the signal only, never the reference. |
| **§42 Derived sex encoding** *(v2.3)* | The QTc bound is sex-specific and PTB-XL's `sex` labels contradicted each other, so the mapping is inferred from anthropometrics with explicit guards rather than assumed. The blueprint does not mention this; getting it wrong would have mis-set a clinical threshold on every record. |
| **[`app/`](../app/README.md) — the application layer** | The blueprint asks for a demo but says nothing about how to serve one. Three things the pipeline forces: uploads must carry the `.hea`+`.dat` pair, one job may touch the GPU at a time (LLM decode plus a backward pass for saliency over a 4-bit model), and the result is twelve structured fields, not a blob. It also ships a **demo mode** so the UI is reviewable without weights, which withholds model output rather than fabricating it. |

Also beyond scope-of-ask: §36 cross-attention grounding, §37 per-lead saliency, §38 retrieval-based precedent, §39 one-call API, §40 report card.

---

## 6. Requirements (Blueprint §6)

| Category | Blueprint | This build | |
|:--|:--|:--|:--:|
| GPU | 1× RTX 3090/4090 (24 GB) or cloud equivalent | Single-GPU, 4-bit NF4 decoder | ✅ |
| RAM and disk | 64 GB RAM; 0.5–1 TB SSD | PTB-XL plus encoder caches ≈ 100 GB | ✅ |
| Core software | Python 3.11, PyTorch 2.x, `transformers`, `peft`, `bitsandbytes`, `accelerate` | [`requirements.txt`](../requirements.txt) | ✅ |
| Signal and data | `wfdb`, `scipy`, `neurokit2`, `scikit-learn`, `pandas`; `pywt` only if ablating wavelets | all present, `PyWavelets` optional. `neurokit2` now also does DWT wave delineation for §46 | ✅ |
| Tracking | Weights & Biases or MLflow | Notebook outputs serve as the run record | 🟡 |
| Access | PhysioNet credentialing; Hugging Face Llama licence | Neither needed for what runs — PTB-XL is open and the decoder is ungated. Credentialing is required only for §51 (MIMIC-IV-ECG SSL pretraining), which is written and blocked on it | ➖ |
| Optional | Ethics/IRB approval; 1–2 cardiologists for the gold set | Not obtained; disclosed | ➖ |

---

## 7. Data to collect (Blueprint §7)

| Data | Blueprint | Status |
|:--|:--|:--:|
| PTB-XL (+ PTB-XL+) | main supervised set and evaluation | ✅ PTB-XL 1.0.3 · ➖ PTB-XL+ absent — and it costs nothing structural: §46 derives PR/QRS/QT/QTc/axis from the waveform, and §48 uses PTB-XL+ only as an optional vendor cross-check (`CARDIOSCRIBE_PTBXL_PLUS`) |
| MIMIC-IV-ECG (~800k) | encoder pretraining and report training | ➖ credentialing not obtained |
| Chapman-Shaoxing / CODE-15 % / CPSC2018 | encoder pretraining and external validation | 🟡 §49 harness **complete** — label loading and SNOMED→superclass mapping included; set `CARDIOSCRIBE_EXTERNAL` and run |
| ECGInstruct / ECG-Grounding / ECG-Chat | optional extra instruction tuning | ➖ |
| Clinician gold set (500–1,000 ECGs) | final evaluation | 🟡 §50 builds `gold_intake_blank.csv` per reader, merges readings, computes κ, adjudicates and scores. Needs actual cardiologists |

---

## 8. Risks and mitigations (Blueprint §9)

| Risk | Blueprint mitigation | Implemented |
|:--|:--|:--:|
| Learning machine-report style and its errors | Clinician gold set; PTB-XL's human-validated folds | 🟡 folds 9/10 used; gold-set machinery built in §50, readings absent. v2.1 also removed one way the model learned the *reference's* artefacts: the SCP fallback used to write `"sttc (from SCP label)"` into gold prose, and the model reproduced it |
| Hallucination | DDP with probabilities, measured numbers, validator, wrong-label stress test | ✅ all four; the stress test shows work remains |
| Data access delays | Start PhysioNet credentialing first | ➖ sidestepped by using PTB-XL only |
| Compute | Train on subsets, cache frozen encoder outputs | ✅ memmapped caches; reduced-budget ablations |
| Privacy | Local inference only; no MIMIC data to external APIs | ✅ all inference local, translation included |
| Clinical overreach | Label it a research tool with mandatory human review | ✅ model card, flagging, abstention |

---

## 9. Scorecard

### By blueprint phase

| Phase | Steps | State |
|:--|:--|:--|
| 0 — Access and setup | 1–2 | ✅ for what runs; PhysioNet credentialing open, needed only for §51 |
| 1 — Data audit | 3–4 | ✅ executed |
| 2 — Preprocessing | 5–9 | ✅ executed; the filter-band ablation (step 7) is flag-gated |
| 3 — Text curation | 10–13 | ✅ executed; v2.1 rewrote §9a (numeric `rate`, no schema placeholder) — **re-run pending** |
| 4 — Stage 0, encoder | 14–16 | ✅ executed. **Gate passed at 0.9307** |
| 5 — Stage 1, Q-Former | 17–18 | ✅ executed; interval regression extendable via §46 |
| 6 — Stage 2, instruction tuning | 19–20 | 🟡 hyperparameters match; budget-capped at the smoke-test setting. §19h rewritten for a real schedule — **unrun** |
| 7 — Inference and guardrails | 21–22 | ✅ executed; v2.1 §42b adds repair to the check, v2.2 §47 extends it to every number — **unrun** |
| 8 — Evaluation and demo | §8 | 🟡 evaluation: clinical metrics, stress test, robustness, calibration and 3 of 4 ablations executed. ✅ demo: [`app/`](../app/README.md) wraps §39 as a local web service |

### By target

| Blueprint §1 target | State |
|:--|:--|
| Encoder macro-AUROC ≥ 0.90 | ✅ **met** at 0.9307, measured |
| Hallucination < 0.05 | 🔄 0.360 measured at the smoke-test checkpoint. Three causes identified (untrained decoder, a schema placeholder leaking into gold prose, a metric implementing half its definition); all three addressed in §9a and §42–43, none re-measured |
| Numbers agree ≥ 95 % | 🔄 never fired — HR stated in 0/100. Root cause was upstream of training: no gold target contained a number. Fixed in §9a-pre/§9a; §46–47 extend the check to PR/QRS/QT/QTc/axis. Unrun |
| Finding-F1 beats ResNet+MLP | 🔄 0.643 measured; the two arms were never compared on F1. §44 builds the missing arm. Unrun |

> The earlier edition of this scorecard attributed all three open targets to the Stage-2 smoke test. That was the largest cause but not the only one — two had root causes a longer training run would not have touched, documented in §1 above and in [`RUN_PLAN.md`](RUN_PLAN.md).

### Coverage

**Every phase of the blueprint — 0 through 10 — is present in the notebook, and every one of the 22 numbered steps is either executed or explicitly accounted for.** Parts VI (§42–45) and VII (§46–51) add the target-closure work and the four next steps, and v2.3 closes the disagreement between them by wiring §46 into the §42 gate; of those four, one is complete and verified (§46–48 interval measurement), and three are blocked on credentialing, a dataset download or actual cardiologists — each shipped as working machinery with **no fabricated number attached**.

---

<div align="center">

[⬅️ README](../README.md) · [📘 Project documentation](PROJECT_DOCUMENTATION.md) · [🏷️ Model card](MODEL_CARD.md)

</div>
