<div align="center">

# ✅ Blueprint Compliance Matrix

**Every numbered step of *CardioScribe v2: ECG-to-Report Project Blueprint*, traced to the notebook that implements it.**

</div>

Legend: ✅ implemented and run · 🟡 implemented, deliberately not run · ❌ not met · ➖ out of scope for this run, disclosed

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
| 22 | A validator checks numbers against measurements and negations against the classifier; failures flagged for human review | §22, §42b | ✅ | ±10 bpm tolerance; 7 negator forms; 10/100 flagged. v2.1 adds the repair the check implies: strike clauses the signal will not support, write in the measured rate, hedge below the calibrated threshold — every edit recorded in `actions`, so the delta between the model's text and the shipped text is never silent |
| **Phase 8 — Evaluation and demo** |
| — | See §5 below | §23–§41 | 🟡 | |

---

## 5. Evaluation and ablations (Blueprint §8)

| Dimension | Blueprint asks for | Notebook | Status |
|:--|:--|:--|:--:|
| **Encoder** | Macro-AUROC and F1 on fold 10; external validation on an unseen dataset | §10d/§10g; §41 harness | 🟡 external harness written, deliberately unrun |
| **Report accuracy** | Extract findings from generated and reference reports; per-finding P/R/F1 | §23, §26 | ✅ |
| **Hallucination** | Share of unsupported findings; wrong-label stress test | §26, §27 | ✅ 0.360 and 0.800 |
| **Numbers** | HR, PR, QRS, QTc in text vs measured | §22, §26 | 🟡 HR only (PR/QRS/QTc need PTB-XL+); HR never stated |
| **Text metrics** | BLEU, ROUGE, BERTScore as secondary | §24 | ✅ BLEU-4 and ROUGE-L; BERTScore skipped (extra dependency, secondary) |
| **Human review** | Cardiologist Likert rating of 100–200 reports | §29 | 🟡 `human_review_sheet.csv` exported; review not performed |
| **Robustness** | Added noise, lead-off, different devices and sites | §30 | ✅ 11 conditions |

### The four required ablations

| # | Ablation | Status | Result |
|:--:|:--|:--:|:--|
| 1 | ResNet + MLP (Doc 1) vs transformer encoder + Q-Former (Doc 2) | ✅ | ResNet 0.9307 vs Transformer 0.9169 |
| 2 | MLP vs Q-Former with the same encoder | ✅ | **MLP 0.9171 / R@1 0.014** vs Q-Former 0.8920 / R@1 0.002 — the blueprint's choice loses at this budget |
| 3 | 0.5–40 Hz vs wider band; with vs without wavelet denoising | 🟡 | Harness complete (§33), `RUN_PREPROCESSING_ABLATIONS=False`. **The one ablation this run leaves unsettled.** |
| 4 | DDP with clean labels vs noisy classifier predictions vs no DDP | ✅ | no 0.594 / noisy 0.610 / clean 0.618 / wrong 0.435 F1 |

### Beyond the blueprint

Two additions the blueprint does not ask for, included because the guardrails are unsound without them:

| | Why |
|:--|:--|
| **§31 Per-class temperature scaling** | Raw sigmoid outputs are miscalibrated, which corrupts both the `p=` values in the prompt and the abstention threshold. ECE 0.0818 → 0.0221. |
| **§32 Risk–coverage selective prediction** | `UNCERTAIN_THRESHOLD = 0.35` was a guess. The curve sets it at **0.852** for ≤ 10 % risk at 48.4 % coverage. |

Also beyond scope-of-ask: §36 cross-attention grounding, §37 per-lead saliency, §38 retrieval-based precedent, §39 one-call API, §40 report card.

---

## 6. Requirements (Blueprint §6)

| Category | Blueprint | This build | |
|:--|:--|:--|:--:|
| GPU | 1× RTX 3090/4090 (24 GB) or cloud equivalent | Single-GPU, 4-bit NF4 decoder | ✅ |
| RAM and disk | 64 GB RAM; 0.5–1 TB SSD | PTB-XL plus encoder caches ≈ 100 GB | ✅ |
| Core software | Python 3.11, PyTorch 2.x, `transformers`, `peft`, `bitsandbytes`, `accelerate` | [`requirements.txt`](../requirements.txt) | ✅ |
| Signal and data | `wfdb`, `scipy`, `neurokit2`, `scikit-learn`, `pandas`; `pywt` only if ablating wavelets | all present, `PyWavelets` optional | ✅ |
| Tracking | Weights & Biases or MLflow | Notebook outputs serve as the run record | 🟡 |
| Access | PhysioNet credentialing; Hugging Face Llama licence | Neither needed — PTB-XL is open and the decoder is ungated | ➖ |
| Optional | Ethics/IRB approval; 1–2 cardiologists for the gold set | Not obtained; disclosed | ➖ |

---

## 7. Data to collect (Blueprint §7)

| Data | Blueprint | Status |
|:--|:--|:--:|
| PTB-XL (+ PTB-XL+) | main supervised set and evaluation | ✅ PTB-XL 1.0.3 · ➖ PTB-XL+ absent (costs PR/QRS/QTc/axis) |
| MIMIC-IV-ECG (~800k) | encoder pretraining and report training | ➖ credentialing not obtained |
| Chapman-Shaoxing / CODE-15 % / CPSC2018 | encoder pretraining and external validation | 🟡 §41 harness ready |
| ECGInstruct / ECG-Grounding / ECG-Chat | optional extra instruction tuning | ➖ |
| Clinician gold set (500–1,000 ECGs) | final evaluation | 🟡 `human_review_sheet.csv` is the intake format |

---

## 8. Risks and mitigations (Blueprint §9)

| Risk | Blueprint mitigation | Implemented |
|:--|:--|:--:|
| Learning machine-report style and its errors | Clinician gold set; PTB-XL's human-validated folds | 🟡 folds 9/10 used; gold set absent |
| Hallucination | DDP with probabilities, measured numbers, validator, wrong-label stress test | ✅ all four; the stress test shows work remains |
| Data access delays | Start PhysioNet credentialing first | ➖ sidestepped by using PTB-XL only |
| Compute | Train on subsets, cache frozen encoder outputs | ✅ memmapped caches; reduced-budget ablations |
| Privacy | Local inference only; no MIMIC data to external APIs | ✅ all inference local, translation included |
| Clinical overreach | Label it a research tool with mandatory human review | ✅ model card, flagging, abstention |

---

## 9. Scorecard

| | Count |
|:--|:--:|
| ✅ Fully implemented and run | **34** |
| 🟡 Implemented, deliberately not run or budget-capped | **13** |
| ❌ Target not met | **3** (all downstream of the Stage-2 smoke test) |
| ➖ Out of scope for this run, disclosed | **9** |

**Every phase of the blueprint — 0 through 10 — is present in the notebook, and every one of the 22 numbered steps is either executed or explicitly accounted for.** The gaps are named, flag-gated and reproducible, not hidden.

---

<div align="center">

[⬅️ README](../README.md) · [📘 Project documentation](PROJECT_DOCUMENTATION.md) · [🏷️ Model card](MODEL_CARD.md)

</div>
