<div align="center">

# Run plan — closing the blueprint §1 targets

**What the v2.1 patch changed, what has to be re-run, and what will and will not pass.**

</div>

---

## 1. Why a re-run is needed at all

Three of the four blueprint §1 targets failed in the v2.0 run:

| Target | v2.0 result | Status |
|:--|:--|:--:|
| Encoder macro-AUROC ≥ 0.90 | 0.9307 (ResNet1D) · 0.9169 (Transformer) | ✅ |
| Hallucination rate < 0.05 | 0.360 | ❌ |
| Stated numbers agree ≥ 95 % | HR stated in 0/100 reports — the check never fired | ❌ |
| Finding-F1 beats the ResNet+MLP baseline | micro-F1 0.643 | ❌ |

`results_summary.json` attributes all three to one cause: Stage 2 ran as a 10-step
smoke test (`SMOKE_TEST=True`) instead of the blueprint's 2–3 epochs. That is the
largest cause, but it is not the only one. Two of the three failures have causes
upstream of the decoder that a longer run would not have fixed:

**The gold targets contained no numbers.** `extract_schema` populated the `rate`
field for 7.4 % of records and never with a bpm value — it scraped words like
"bradycardia" out of the report text. No training example ever showed the model a
heart rate, so the model never stated one, so the agreement check never fired.
Target 3 was unreachable regardless of how long Stage 2 ran.

**A schema placeholder leaked into the gold prose.** When a superclass label had no
matching text, the schema fallback wrote the literal string `"sttc (from SCP label)"`
into the gold report. The v2.0 model learned to emit exactly that, which is visible
in the §25 sample output.

**The hallucination metric implements half its own definition.** The blueprint defines
hallucination as findings "unsupported by the reference **or signal**". §26 scores
only against the reference, so a finding the classifier and the measurements both
back still counts as a hallucination if PTB-XL's terse German-translated report
happens not to mention it.

**The two bridge arms were never compared on F1.** §28 compares them on AUROC and
retrieval only — and the MLP arm wins there (0.9171 vs 0.8920, R@1 0.014 vs 0.002).
Target 2 names a metric neither arm was ever scored on.

---

## 2. What the patch changed

| # | Cell | Change |
|:--:|:--|:--|
| 1 | **36** (new) | Measures heart rate once per record from lead-II R-peaks, caches to `hr_by_ecg_id.npz` |
| 2 | **37** (§9a) | `rate` now carries the measured bpm for every record; the SCP fallback writes clinical phrasing instead of `"<class> (from SCP label)"` |
| 3 | **38** (§9b) | Passes the measurement through |
| 4 | **39** (§9c) | `CARDIOSCRIBE_TEMPLATES` knob to trim the pair count |
| 5 | **77** (§19b) | `MICRO_BATCH` 1 → 4 (env-overridable); `GRAD_ACCUM` derived so the effective batch stays 32 |
| 6 | **91** (§19h) | Stage-2 loop rewritten: val-loss checkpointing, best-checkpoint selection, wall-clock budget, resume across sessions |
| 7 | **145** (§42) | Signal-evidence model: per-class temperature and thresholds fitted **on fold 9 only** |
| 8 | **147** (§42b) | Evidence gate and report repair — the validator blueprint step 22 implies |
| 9 | **149** (§43) | Re-scoring with both hallucination definitions, reported side by side |
| 10 | **151** (§44) | The missing MLP arm with a soft-token head, for the head-to-head |
| 11 | **153** (§45) | Re-scored target table; writes `results/results_summary.json` |

Every v2.0 output is still in the notebook. Cells whose source changed carry a banner
saying their output predates the patch.

### What the gate may and may not look at

The evidence gate reads the calibrated classifier and the deterministic measurements.
**It never reads the reference report.** If it did, the strict hallucination metric
would be measuring the gate against itself. Thresholds are fitted on fold 9; fold 10
is touched only at scoring time.

---

## 3. The re-run

### Prerequisites

- PTB-XL 1.0.3 at `CARDIOSCRIBE_DATA`, a writable `CARDIOSCRIBE_WORK`
- One 16–24 GB GPU (the v2.0 run used 2× T4 on Kaggle)
- The Stage-0 encoder and Stage-1 Q-Former checkpoints from the v2.0 run. **Stage 1
  does not need re-running**: §13 embeds `report_en`, not the schema, so the schema
  change does not touch it.

### How long Stage 2 actually takes

The one measured number is **~1.05 s per sample** at micro-batch 1 (the committed
smoke run, one T4). Everything else follows from it:

| Templates | Train pairs | Opt-steps/epoch | micro-batch 1 | micro-batch 4 |
|:--:|--:|--:|--:|--:|
| 2 | ~34,800 | 1,088 | 10.2 h | **3.4–4.4 h** |
| 3 | ~52,300 | 1,632 | 15.2 h | **5.1–6.5 h** |
| 5 (all) | ~77,100 | 2,409 | 22.5 h | **7.5–9.6 h** |

The micro-batch-4 column is a **2.3–3× range, not a measurement.** Batching mainly
amortises bitsandbytes dequantisation and kernel-launch overhead, and how much it
buys on a given card is not something to assert without running it. §19h prints
seconds-per-step every 25 steps, so you will know your real rate within ~5 minutes
of starting — and the cell prints its own budget estimate before it begins.

Everything outside Stage 2 is minor: Phase 3's one-time HR measurement is ~20–30 min
over 21,799 records (cached afterwards), reloading checkpoints ~10–15 min, generating
100 reports ~6 min (measured: 377 s), and §42–45 ~5 min.

### Path A — one session, ~3–4 h total

A full epoch does not fit. Cap the steps instead — and cap them *properly*, with
`CARDIOSCRIBE_S2_MAX_STEPS` rather than wall-clock alone, so the cosine schedule's
`T_max` matches the budget and the LR anneals to `eta_min` exactly when the run ends
instead of stopping mid-curve at a high value.

```bash
export CARDIOSCRIBE_SMOKE=0            # the one flag that mattered
export CARDIOSCRIBE_TEMPLATES=3
export CARDIOSCRIBE_MICRO_BATCH=4
export CARDIOSCRIBE_S2_MAX_STEPS=700   # 22,400 examples seen; ~0.43 of an epoch
export CARDIOSCRIBE_TIME_BUDGET_H=3.5  # safety net, not the primary control
```

700 optimizer steps at effective batch 32 is **70× the committed run** and a normal
amount of training for a LoRA + bridge adapter — most of the adaptation happens
early. It is a *partial epoch*, and §45 records `steps_run`, so the artifact never
claims more training than it got.

| Step | Cells | Time |
|:--|:--|:--|
| Config | 0–4 | < 1 min |
| Measure HR, rebuild schema and pairs (Phase 3) | 36–41 | ~20–30 min, cached after |
| Reload encoder + Stage-1 checkpoints | 54–70 | ~10–15 min |
| Load LLM, rebuild prompts and loaders | 75–89 | ~5 min |
| **Stage 2 (700 steps)** | **91–92** | **~2.2–2.8 h** |
| Generation + v2.0 metrics | 95–113 | ~10 min |
| Calibration and selective prediction | 117–120 | ~5 min |
| **Target closure** | **145–153** | **~5 min** |

### Path B — one full epoch, ~6–8 h total

Drop `CARDIOSCRIBE_S2_MAX_STEPS`, keep `CARDIOSCRIBE_TEMPLATES=3`. Fits a 9 h Kaggle
session with room to spare.

### Path C — the blueprint's 2–3 epochs, 3–4 sessions

All 5 templates, `CARDIOSCRIBE_S2_EPOCHS=3` — 22–29 h of Stage 2. Run across sessions:

```bash
export CARDIOSCRIBE_TIME_BUDGET_H=8
export CARDIOSCRIBE_RESUME=1         # from session 2 onward
```

§19h checkpoints to `stage2_state.pt` on the budget and reloads optimizer and
scheduler state on resume. Note Kaggle's ~30 GPU-hours/week quota: Path C plus the
head-to-head pass is close to the whole week.

### Budget summary

| | Stage 2 | Everything else | **Total** | Head-to-head adds |
|:--|--:|--:|--:|--:|
| **A** — 700 steps | 2.2–2.8 h | ~1 h | **~3.5 h** | +2.5–3 h |
| **B** — 1 epoch | 5.1–6.5 h | ~1 h | **~6–7.5 h** | +5–6.5 h |
| **C** — 3 epochs | 22–29 h | ~1 h | **~23–30 h** | +22–29 h |

The head-to-head (§44) needs a second Stage-2 pass at the **same** budget, since
scoring an untrained MLP arm against a trained Q-Former is not a comparison. If you
can only afford one pass, run Path A twice rather than Path B once — target 2 needs
both arms to mean anything, and §45 reports `NOT RUN` rather than guessing.

### The head-to-head (target 2)

Target 2 needs a second Stage-2 pass, because scoring an untrained MLP arm against a
trained Q-Former arm would not be a comparison. After Path A or B:

1. `mlp_arm = build_mlp_arm()` (§44) — warm-starts from §28's Stage-1 MLP weights
2. `qformer_model = mlp_arm`, then re-run cells 91–92 with the **same** flags
3. Restore the Q-Former checkpoint and run:

```python
ARMS = {'Q-Former (Doc 2)': evaluate_arm(qformer_model, 'qformer'),
        'MLP (Doc 1)'     : evaluate_arm(build_mlp_arm(), 'mlp')}
MLP_ARM_F1 = ARMS['MLP (Doc 1)']['repaired']['micro_f1']   # §45 reads this
```

Budget a second ~3 h pass. Until it runs, §45 reports target 2 as `NOT RUN` rather
than guessing.

---

## 4. What to expect — honestly

**Target 3 (numbers) will pass, and the table says why.** §45 reports agreement
twice. *After validator* will be ~1.00 with HR stated in 100 % of reports, because
the validator writes the measured value in — that is near-tautological and is
labelled as such. *Model unaided* is the number that says whether the model learned
to state a rate at all; it was 0/100 in v2.0 and should be substantially non-zero
once the gold targets contain numbers. Read the unaided column.

**Target 4 (hallucination < 0.05 strict) probably will not pass, and that is a real
finding about the data, not a bug.** Strict hallucination is 1 − precision against
PTB-XL report text that is short, often a single clause, and machine-translated from
German. A generated finding that the signal fully supports still counts against us
whenever the reference simply did not mention it. The gate cuts the false-positive
count — that is what it guarantees — but the ceiling on strict precision is set by
how much the references omit.

The blueprint's own wording ("reference **or** signal") anticipates this, and §43
reports that number too. **It is not evidence the target passed**: the gate is driven
by the same signal the metric credits, so the two are not independent. Treat it as a
description of the gate, not a score. The only clean way to pass target 4 as written
is the gold set the blueprint asks for — 500–1,000 cardiologist-written reports — and
that is listed as not run for good reason.

**Watch the counts, not only the ratio.** Striking an ungrounded claim that happened
to be correct removes a true positive as well as a false one, shrinking the TP+FP
denominator. The strict *ratio* can rise while the false-positive *count* falls. §43
prints tp / fp / fn so the trade is visible. `CARDIOSCRIBE_GATE_PRECISION` (default
0.90) is the dial: raise it for fewer, safer assertions and lower recall.

**Target 2 may be settled against the Q-Former.** If the MLP arm wins the head-to-head
as it won on AUROC, that is the blueprint's ablation doing its job — it exists to
settle Doc 1 vs Doc 2 — and the answer would be to ship Doc 1's simpler bridge. §45
reports whichever arm wins; it does not relabel a loss as a pass.

**Five findings can never be signal-supported.** Axis deviation, low voltage, pacing,
ectopy and QT prolongation are neither superclasses nor measured here; they need
PTB-XL+ 12SL features. With `CARDIOSCRIBE_GATE_UNGROUNDABLE=drop` (default) the gate
strikes them, which costs recall on them — visible in §43's per-finding table. That
is the cost of refusing to assert what nothing in this pipeline can check. Set it to
`keep` to see the other side of the trade.

---

## 4b. Part VII (§46–51) — what runs and what cannot

Added in v2.2. Only §46–48 produce numbers without new access.

| Section | Needs | Cost | Status |
|:--|:--|:--|:--|
| §46 measurement | nothing — runs on CPU | self-test is instant | **ready; verified on synthetic beats** |
| §47 validator + interval cache | the signal caches | ~0.4 s/record, so ~1 min for the 100 eval records | **ready** |
| §48 PTB-XL+ cross-check | PTB-XL+ download (open access) | minutes | gated on `CARDIOSCRIBE_PTBXL_PLUS` |
| §49 external validation | Chapman/CPSC download **and** the trained encoder **and** a GPU | ~1 h | gated on `CARDIOSCRIBE_EXTERNAL` |
| §50 gold set | **two cardiologists and an adjudicator** | weeks of clinician time | tooling ready; no readings exist |
| §51 SSL pretraining | PhysioNet credentialing (days–weeks) + GPU-days | GPU-days | gated on `CARDIOSCRIBE_MIMIC` |

Extra flags: `CARDIOSCRIBE_PTBXL_PLUS`, `CARDIOSCRIBE_EXTERNAL`, `CARDIOSCRIBE_MIMIC`.

> If you want PR/QRS/QTc/axis in the *generated* reports rather than only validated,
> add them to `REG_FIELDS` (§12) and to the prompt via `format_measurements_full`
> (§47), then re-run Stage 1 and Stage 2. That is a larger change than §47 alone and
> is not done here.

---

## 5. Flags

| Variable | Default | Effect |
|:--|:--:|:--|
| `CARDIOSCRIBE_SMOKE` | `1` | **Set to `0`.** The single flag behind the v2.0 failures |
| `CARDIOSCRIBE_TEMPLATES` | all 5 | Instruction templates kept, hence pairs per epoch |
| `CARDIOSCRIBE_MICRO_BATCH` | `4` | Drop to 1 on OOM; `GRAD_ACCUM` compensates |
| `CARDIOSCRIBE_S2_EPOCHS` | `2` | Blueprint asks 2–3 |
| `CARDIOSCRIBE_S2_MAX_STEPS` | none | Hard step cap. Sets the cosine `T_max` too, so the LR anneals over the budget you actually intend to spend |
| `CARDIOSCRIBE_EVAL_EVERY` | `150` | Optimizer steps between val-loss evaluations |
| `CARDIOSCRIBE_TIME_BUDGET_H` | none | Checkpoint and stop before a session timeout |
| `CARDIOSCRIBE_RESUME` | `0` | Resume Stage 2 from `stage2_state.pt` |
| `CARDIOSCRIBE_GATE_PRECISION` | `0.90` | Validation precision the gate thresholds target |
| `CARDIOSCRIBE_GATE_UNGROUNDABLE` | `drop` | `drop` or `keep` findings with no evidence channel |
| `CARDIOSCRIBE_RATE_QUALIFIER` | `0` | Add "(bradycardia)" to the repaired rate line — see §42b |
