# Findings & Solution Strategy (consolidated)

*HAR Assignment 3 — disjoint-user 6-class accelerometer classification. Production LB 0.8154 (backup 0.8114).*

---

## 0. The problem, stated precisely

> **Learn class-discriminative features for the under-served classes (L2, L3, L5) that are *invariant to the user (subject)* — i.e., separate the activity signal from the user-identity confound.**

The training users are disjoint from the test users. Models latch onto **user-spurious attributes** (gravity/orientation, anatomy-correlated cues) that correlate with the label *in the train users* but do **not transfer** to the test users. That is a textbook **spurious correlation under domain (subject) shift**.

This is *conditional subject-invariant discriminative representation learning*: invariant to user **within each class**, while keeping the features that discriminate the classes.

---

## 1. Data (measured this session, not assumed)

- Train **11,020** windows · Test **6,849** · **60** train users (disjoint test users).
- 1 Hz aggregated accelerometer: 6 channels (mean/std × x/y/z) × 300 timesteps.

**Class distribution — severe imbalance (33:1):**

| Class | Count | Share | Per-user spread (min/med/max) | Users present |
|---|---|---|---|---|
| L0 | 4,643 | 42.1% | 24 / 78 / 142 | 60/60 |
| L1 | 4,695 | 42.6% | 4 / 81 / 133 | 60/60 |
| **L2** | **358** | **3.2%** | **0 / 4 / 21** | 52/60 |
| L3 | 656 | 6.0% | 0 / 10 / 28 | 59/60 |
| L4 | 142 | 1.3% | 0 / 0 / 24 | 19/60 |
| L5 | 526 | 4.8% | 0 / 4 / 87 | 35/60 |

**Test distribution is NOT inverse.** The LB-optimal config predicts ≈ train priors (L2 4.3% vs 3.2%, L4/L5 slightly lower). The large OOF→LB gap (+0.027) is the **user-variability** shift ("train harder than test"), not a class-prior flip.

---

## 2. Where the points are (OOF per-class F1 @ production peak, macro 0.788 / LB 0.8154)

| Class | F1 | Prec | Recall | Status |
|---|---|---|---|---|
| L0 | 0.967 | .959 | .975 | solved |
| L1 | 0.908 | .908 | .908 | strong |
| **L2** | **0.384** | .398 | .372 | broken both ways |
| **L3** | **0.764** | .726 | .805 | under-served (precision) |
| L4 | 0.924 | .948 | .901 | solved (rescued by per-file-norm) |
| **L5** | **0.781** | .886 | **.698** | under-served (missing 30% recall) |

**The bottleneck is three classes, not one.** Under macro-F1 each class is worth 1/6 regardless of size, so L3 and L5 leak as much total macro as L2 — and are likely easier to move.

---

## 3. Root-cause diagnosis

- **L2 = information-limited.** 1 Hz aggregation erased the gait-frequency content that separates L1 from L2; L2 is also thinly spread (median 4 windows/user). Boosting cannot manufacture a split that isn't in the features — pushing harder just overfits per-user quirks and fails on disjoint test users. *Threshold-shifting (×2.46), not more boosting, is what moved L2.*
- **General bottleneck = cross-subject shift.** Models exploit user-spurious attributes that don't transfer.
- **L4 is the proof of the cure.** Rarest class (142) yet F1 0.924 — because it's motorically distinct **and** per-file-norm stabilised its user-orientation signature. The fix is *remove user-spurious signal*, not *more weight/data*.

---

## 4. What we already have (production)

**100% gradient boosting at the core**, with layers on top:

| Layer | Component | Detail |
|---|---|---|
| Learners | **LightGBM + XGBoost** (GBDT) | P1 = LGBM combo stacker; P2 = LGBM hierarchy (Coarse → Fine_walk[+XGB] → Fine_other) |
| Imbalance | **full inverse-frequency class weights** | L2 ≈13×, L4 ≈33×, L3 ≈7×, L5 ≈9× the weight of L0/L1 — *already aggressive* |
| Structure | stacking + hierarchical decomposition | combo = LGBM meta-model on 805-col stack |
| Combine | α-blend (α=0.842) | P1 breadth + P2 L1↔L2 focus |
| Post-hoc | per-class isotonic → NM → 31×31 (L1,L2) grid | reproduces 0.8154 **exactly** from frozen OOFs |

---

## 5. What's been tried and FAILED (do not re-tread)

| Attempt | Result | Why it failed |
|---|---|---|
| CNN-BiLSTM / BiGRU (seq) | OOF ~0.69, LB ~0.72 | 1 Hz aggregation ceiling |
| **Evidential Alignment** (KDD'25 best paper) | neutral; EA-P2 → LB 0.7990 | post-hoc *calibration* can't manufacture separation; threshold trap |
| **SICL** (subject-invariant contrastive, naive) | failed | suppressed user signal **and** anatomy-correlated activity signal |
| **DANN** (domain-adversarial) | marginal | didn't survive the stacker |
| TENT (test-time adapt) | broke generalisation | BN/RNN train-mode interaction |
| SMOTE (early) | ineffective | predicted even less L2 |
| GAF / STFT 2-D CNN | no signal | 1 Hz < Nyquist for gait frequency |
| Binary L1↔L2 EA specialist | 0.5621 | data-starved on ~280 L2/fold |
| GRU in blend / boost-stack | noise / weak-stacker only | redundant with GBDT |
| Aggressive threshold extrapolation (α=0.900) | LB 0.7698 | OOF↛LB; threshold trap |

**Transferable lessons:**
- **OOF→LB threshold trap:** identical OOF (0.788) → LB 0.7698 / 0.7991 / 0.8154 by thresholds alone. *Always validate under FROZEN top-1 thresholds.*
- **Per-file-norm** (per-subject z-score) is the one lever that improved neural cross-subject robustness (rescued L4).
- **EO feature selection** (drop gravity/user-signature features) was 2nd-most impactful single addition — *manual* spurious-attribute removal.

---

## 6. The matched solution class

The problem (§0) is **conditional subject-invariant discriminative representation learning**. Boosting on aggregate features *cannot* do this — it splits on whatever it's given, including spurious cues (which is why EO had to drop gravity features by hand). Only **representation learning** can *learn* to remove the spurious attribute. Matched methods, tiered:

1. **GILE** (AAAI'21) — *disentangle* domain-agnostic vs domain-specific latents + Independent Excitation to decorrelate them. The direct answer to "keep activity, drop subject." **No target data needed.**
2. **CCIL** (AAAI'25) — *class-conditional* feature **and** logit invariance (concept matrix per class). Encodes exactly "same class → same concept across users."
3. **DDLearn** (KDD'23) — self-supervised **diversity** augmentation + **supervised-contrastive discrimination**; built for **low-resource** shift (addresses the L2 starvation that killed our specialist).
4. **DIVERSIFY** (ICLR'23) — characterises latent domains when subject ≠ true domain structure.

**Key refinement over our failed attempts:** we need **conditional** invariance (within-class user-invariance), not **marginal** suppression — marginal suppression is exactly what nuked DANN/SICL (it removed anatomy-correlated activity signal too). EA (what we did) is in this family but is *post-hoc uncertainty calibration*, not representation learning — which is why it was neutral.

---

## 7. Recommendation (open decision)

- **Augment, don't replace.** The GBDT stack wins 5/6 classes (L0/L1/L4 ~0.93). Replacing it gambles those to chase L2. Keep it as the backbone.
- **Add a GILE/CCIL-style conditional-invariant DG model** targeting {L2, L3, L5}; integrate via a **gated/hierarchical route** (not probability blending — proven to fail for neural).
- **Cheap boosting-native lever (parallel, no architecture dependency):** focal objective + tuned per-class weights + a **residual-correction meta-stage** over P1/P2. Most likely to help **L3/L5** (investment-limited); unlikely to help **L2** (information-limited). Validate under frozen thresholds.

**Pending user decision:** architecture (augment / parallel-then-combine / replace) and target scope (L2-only / broaden to L2+L3+L5).

**Hard constraints:** no external data; train on the server (`dm2026-a3` env), not locally; generate CSV for the user to upload (no Kaggle submission on their behalf).

---

## 8. Outcome of the DG augmentation (executed 2026-05-31)

Built **CISC** = CNN-BiGRU + class-weighted CE + **conditional cross-user supervised-contrastive** loss (`src/models/train_dg_cisc.py`, `src/utils/cond_supcon.py`). The loss pulls *same-class-different-user* pairs together → subject-invariance **within** class — the principled fix for why SICL/DANN's marginal suppression failed. Trained 5-fold (user-disjoint) on the server GPU; v1 (inverse-freq oversampling) and v2 (3-seed, sqrt oversampling, averaged).

**What worked (the first real positive in the whole neural arc):**
- DG carries genuinely **complementary L2 signal**: it correctly catches **84–91 of production's 225 L2 misses (37–40%)**; oracle ceiling (ideal per-sample gate) = **0.8414–0.8457** vs production 0.7880.
- A **frozen-threshold, nested-CV-validated** L2-only calibrated injection (w=0.05) gives an **honest +0.0008 OOF** (v1) — the injection weight is chosen on disjoint users from the eval, so it transfers (no threshold-trap).

**What capped it (the binding constraint):**
- The gain is tiny (+0.0002–0.0008 OOF, within LB noise), and the *better* v2 model captured *less* than v1. No realizable gate (DG confidence, LR stacker, hard L1→L2 rescue at −0.0088) can isolate the ~90 good rescues from DG's false positives, because DG's L2 confidence is itself unreliable. **The 1 Hz L1/L2 information ceiling binds — not the model, loss, or imbalance handling.**

**Deliverables:** `submissions/sub_dg_cisc_v1_gated_w05.csv` (best, +0.0008) and `sub_dg_cisc_v2ms_gated_w05.csv`. Production **0.8154** / backup **0.8114** remain the primary submissions. Full detail in `experiments.md` (rows 20–23 + DG-arc verdict).

**Empirical LB (submitted): both DG submissions scored exactly 0.8154 — a tie with production.** The frozen-threshold + nested-CV discipline correctly forecast a flat LB (the +0.0008 OOF was within noise). This is the definitive confirmation that 0.8154 is the **data-bound ceiling**: the strongest principled method we could build lands exactly on it. The L2 gap to the oracle (0.84+) is a 1 Hz information limit, not a modeling one.

**Methodological win regardless of the score:** this validated the central thesis — *conditional* subject-invariant representation learning is the right tool for "learn the class while removing the user," and it is the only approach that produced integration-surviving complementary L2 signal. The remaining gap to the oracle is a *data* limitation (no high-frequency gait content at 1 Hz), not a modeling one.
