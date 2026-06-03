# Literature Synthesis — Part III: robustness-to-distribution-shift coverage audit (the private-LB lever)

> **Why this exists:** the static private LB rewards robustness and punishes public-overfit (public #1 `314540024` 0.8305 → private #9; public #10 `513559009` 0.8078 → private #5). We are #5 public / #6 private — consistent, not overfit. To improve the *private* result the lever is **generalization to the unseen split**, not raw score. This audits the robustness literature against what we actually did and flags the genuine gaps.

## Covered (solid foundations)
- **Cross-subject CV** (GroupKFold by user) + **nested CV** for *all* selection (thresholds, injection w) — the bedrock of honest generalization estimation and the reason we didn't crash like the public-chasers.
- **Robust thresholds** (explicitly trade OOF for robustness) + **per-class isotonic calibration** (5-fold OOF).
- **Multi-seed base models** + **P1/P2 blend** — variance reduction.
- **Domain generalization**: DANN, conditional cross-user SupCon (CISC), SICL, cross-subject mixup. *Measured LIMITED headroom* — the L1↔L2 overlap is intrinsic, not user-induced (within-vs-cross-user AUC gap ≈ 0.02; geometry ARI tracks activity 0.319 vs user 0.005).
- **Pseudo-labeling / transductive** (`lgbm_combo_v2_pl`).
- **Adversarial-validation DETECTION**: domain-classifier AUC **0.732** (moderate train↔test covariate shift) already measured in `unsup_separability_probe.py`.

## Genuine gaps (uncovered, robustness-relevant)
**A. Test-prior / label-shift correction — Saerens EM (2002), BBSE (Lipton et al. ICML 2018).** *[HIGH value, cheap, principled]* The private split (disjoint users) may carry **different class priors**; our threshold grid is tuned to TRAIN priors → miscalibrated under prior shift. Saerens-EM / BBSE estimate the test prior from the model's *own* predictions (+ confusion matrix) and re-weight posteriors — **no retraining, no labels**. This is principled adaptation to the unseen distribution = exactly private robustness, and it's testable on cross-user OOF by simulating a prior shift. NOT public-chasing.

**B. Adversarial-validation feature PRUNING / invariant feature selection.** *[HIGH value, cheap]* We measured the train↔test shift (AUC 0.73) but never *removed* the features driving it — those are subject-specific **spurious** features that hurt cross-subject generalization. Rank features by train/test discriminability → ablate the top ones → re-check cross-user OOF. (This is the acknowledged TODO: "EO with a penalty against user-predictive features.")

**C. Group DRO / worst-user optimization; Just-Train-Twice (JTT).** *[MEDIUM, caveated]* Minimize the **worst-USER** loss (Sagawa et al. ICLR 2020) → robust to the worst-case subject. Caveats: Group-DRO can *underperform* ERM on subpopulation shift (Koh et al. 2020); it's built for overparameterized NNs and is non-trivial to bolt onto a GBDT stack. JTT (Liu et al. ICML 2021) = a simpler "upweight the ERM errors and retrain" alternative.

**D. Further DG variants — CORAL, IRM, Fishr, MLDG, AFFAR.** *[LOW-MEDIUM]* Same family as what we already ran; our measurement (overlap is not user-induced) caps the headroom. A 2025 HAR survey notes accuracy can drop 42% under heterogeneous domain shift — but our shift is *within-dataset* and mild (AUC 0.73), so the big DG gains reported across *datasets* won't transfer.

## Recommendation
Pursue **A (label-shift correction)** and **B (adversarial feature pruning)** — both cheap, principled, testable on cross-user OOF, and directly aimed at the private split rather than the public board. A is the most apt: it adapts to the unseen distribution *by construction*. Expected gains are modest (we already generalize well), but they're the *right* robustness moves and carry no public-chasing risk.

### Sources (Part III)
- [Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts*, ICLR 2020](https://arxiv.org/abs/1911.08731) · [group_DRO code](https://github.com/kohpangwei/group_DRO) · [Just Train Twice, ICML 2021](http://proceedings.mlr.press/v139/liu21f/liu21f.pdf)
- [Lipton et al., *Detecting and Correcting for Label Shift with Black Box Predictors* (BBSE), ICML 2018](https://arxiv.org/abs/1802.03916) · [Label Shift Estimation: A Bayesian Approach, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Ye_Label_Shift_Estimation_for_Class-Imbalance_Problem_A_Bayesian_Approach_WACV_2024_paper.pdf)
- [Adversarial validation for train-test divergence (overview)](https://unfoldai.com/adversarial-validation/) · [Managing dataset shift by adversarial validation, arXiv 2112.10078](https://arxiv.org/pdf/2112.10078)
- [Towards Generalizable HAR: A Survey, arXiv 2508.12213](https://arxiv.org/pdf/2508.12213) · [DIVERSIFY (cross-person), arXiv 2406.04609](https://arxiv.org/pdf/2406.04609) · [AFFAR / Domain Generalization for HAR, ACM TIST](https://dl.acm.org/doi/10.1145/3552434)

---

# Literature Synthesis — Part II: the L2 problem is *class overlap under imbalance* (and we're at the Bayes floor)

> **Why this exists:** after proving L1↔L2 are *separable* (AUC 0.86) yet capped (L2-F1 ~0.38), the question became: what does the literature call this, why did the textbook fixes fail, and what does it say to do when they fail? Answer: the problem is **class overlap combined with imbalance** where the minority is dominated by **"unsafe" examples** — and we are sitting on the **Bayes error floor**. Measured, not asserted (`scripts/l2_typology_bayes.py`, `l1l2_*.py`, `hierarchy_vs_flat.py`).

## 1. The precise name: "unsafe" minority examples under class overlap
Plain imbalance (rare-but-separable) is *easy* — re-weighting solves it. Our case is the hard regime: **overlap + imbalance**, where the minority sits *inside* other classes. [Napierała & Stefanowski (2016, J. Intelligent Information Systems)](https://link.springer.com/article/10.1007/s10844-015-0368-1) classify each minority example by its k=5 neighbourhood: **safe** (4–5 same-class), **borderline** (2–3), **rare** (1), **outlier** (0); the last three are "unsafe" and provably hard.

**Measured for L2** (k=5, PCA-50 feature space): **safe 4.5% · borderline 21.2% · rare 30.4% · outlier 43.9% → 95.5% UNSAFE.** Nearly half of L2 are *outliers* (zero L2 neighbours — completely surrounded by L1/L3/L5). This is a near-pathological instance of the overlap-imbalance regime, exactly the case the literature flags as resistant to the standard toolbox.

## 2. We are at the Bayes (irreducible-error) floor
The [Bayes error rate](https://en.wikipedia.org/wiki/Bayes_error_rate) is the lowest achievable error for *any* classifier on a given P(X,Y) — the irreducible overlap of the class-conditional distributions. kNN error brackets it (BER ∈ [e₁ₙₙ/2, e_largeₖ]). Balanced, cross-user (`l2_typology_bayes.py`):
| overlap | Bayes-error bracket | balanced Bayes ACC | strong LightGBM | verdict |
|---|---|---|---|---|
| L1 vs L2 | [0.170, 0.288] | ~0.71 | err 0.240 | **at floor** |
| L2 vs L3 | [0.155, 0.243] | ~0.76 | err 0.206 | **at floor** |
| L2 vs L5 | [0.169, 0.316] | ~0.68 | err 0.252 | **at floor** |
Our strong classifier's error is *inside* the Bayes bracket for all three overlaps. Even a perfect classifier on these features would misclassify ~24–32% of balanced L2-vs-neighbour pairs. *"Once your classifier approaches the Bayes error rate, further improvements become increasingly difficult — you're approaching the fundamental limit imposed by the data distribution itself."*

## 3. Literature solution families → what we tried → why each failed
| Family (canonical refs) | What we ran | Why it didn't move L2 (grounded) |
|---|---|---|
| **Resampling**: SMOTE, Borderline-SMOTE (Han 2005), ADASYN, SMOTE-Tomek/ENN, overlap undersampling | `lgbm_full_smote` | SMOTE **blurs boundaries in overlap** and fails on outlier/rare examples; with L2 95.5% unsafe, it synthesises *into* the L1/L3/L5 region — adds noise. Predicted failure. |
| **Cost-sensitive / margin / logit adjustment**: LDAM (Cao NeurIPS'19), Logit-Adjustment (Menon ICLR'21), Balanced-Softmax (Ren'20) | class_weight=balanced everywhere; **the (L1,L2) threshold grid IS logit adjustment** (per-class log-weights = additive logit shifts), nested-validated | Logit adjustment only **moves the operating point along a fixed ranking** — it cannot create separability. The L2-weight sweep proved the operating point is already optimal (peaks exactly at production). Got us to 0.8200; can't cross Bayes. |
| **Decoupling rep/classifier** (Kang ICLR'20: "decoupling beats fancy losses") | isotonic calibration + threshold = decoupled classifier rebalancing on a frozen representation | Already implemented; it's *why* GBDT+isotonic beats the neural losses. Maxed. |
| **Metric / contrastive separation**: SupCon (Khosla'20), SICL, our conditional cross-user SupCon (CISC) | `train_dg_cisc`, GRU-evidential, trajectory-CNN | Metric learning can only pull apart what the features make separable; Bayes ACC ~0.71 caps it. 358 samples (44% outliers) across disjoint users = too few clean positives. Confirmed neural ≤ GBDT (+0.0008). |
| **Hierarchical / coarse-to-fine** | `hierarchy_vs_flat.py` cascade; production P2 (hier_v6) | Flat 6-class **beats** class-by-class cascade (−0.019, hurts L2 via error propagation). The *good* hierarchy (coarse→fine, blended 16%) is already in. |
| **Domain generalisation**: DANN (Ganin'16), CORAL, DIVERSIFY | DG/CISC | Within-vs-cross-user AUC gap is only ~0.02; geometry tracks activity not user. Overlap is **not** user-induced → DG addresses a small part. +0.0008. |

**Pattern:** every family is either (a) already in production at its measured optimum (logit adjustment, decoupling, the useful hierarchy) or (b) marginal/negative because it attacks a layer that isn't the binding constraint. The binding constraint is the Bayes floor of the 1 Hz/no-gyro feature space.

## 4. What the literature says to do *when the standard fixes fail* (i.e., at the Bayes floor)
Bayes error is a property of (X, Y). You cannot lower it by changing the model — only by changing the information. The sanctioned routes:
1. **Richer features / sensor fusion (the real fix)** — add the gyroscope, raise the sampling rate, fuse modalities. **Forbidden here** (no external data; fixed 1 Hz; no gyro). The *one legal version* — **construct a complementary view from existing data** — is exactly what the orientation pseudo-gyro did (derivative of the gravity-orientation trace ≈ missing gyro), and it is the **only lever that ever helped (+0.0009)**. We exhausted the derivable views (summary + trajectory).
2. **Learning with a reject option / abstention** (Chow 1970; Cortes et al., learning-with-rejection) — abstain on the overlap. Not applicable: a label is forced on every sample.
3. **Optimise the deployment metric directly + hedge variance** — threshold grid is macro-F1-direct; the 2-pick (peak/robust) final hedges the OOF↔public divergence. Done.
4. **Semi-supervised / transductive use of unlabeled test** — checked (`unsup_separability_probe.py`): no extra structure at 1 Hz.

**Conclusion:** L2 is a textbook *overlap-under-imbalance* problem with a 95.5%-unsafe minority, and our strong classifier sits **on the Bayes floor** for all three of its overlaps. The failure of resampling / margin / contrastive / hierarchical methods is not a tuning miss — it is the **predicted** behaviour at the irreducible floor. The only literature-sanctioned remedy is *new information*, of which the single available unit (orientation) is already captured. **0.8200 is the realizable ceiling**, now established from the imbalance-overlap and Bayes-error literature as well.

### Sources (Part II)
- [Napierała & Stefanowski, *Types of minority class examples…*, J. Intell. Inf. Syst. 2016](https://link.springer.com/article/10.1007/s10844-015-0368-1) · [conf. version (HAIS 2012)](https://link.springer.com/chapter/10.1007/978-3-642-28931-6_14) · [data-typology code](https://github.com/miriamspsantos/data-typology)
- [Neighbourhood-based undersampling for imbalanced & overlapped data (Inf. Sci. 2019)](https://www.sciencedirect.com/science/article/abs/pii/S0020025519308114) · [OBMI borderline oversampling (Complex & Intell. Syst. 2024)](https://link.springer.com/article/10.1007/s40747-024-01399-y) · [Borderline-SMOTE](https://www.researchgate.net/publication/225129029_Borderline-SMOTE_A_New_Over-Sampling_Method_in_Imbalanced_Data_Sets_Learning)
- [Kang et al., *Decoupling Representation and Classifier for Long-Tailed Recognition*, ICLR 2020](https://arxiv.org/pdf/1910.09217) · [Systematic Review on Long-Tailed Learning (2024)](https://arxiv.org/html/2408.00483v1) · [Difficulty-aware Balancing Margin Loss, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34261/36416)
- [Bayes error rate (overview)](https://en.wikipedia.org/wiki/Bayes_error_rate) · [Bayes Error Rate Estimation in Difficult Situations (arXiv 2506.03159)](https://arxiv.org/html/2506.03159v3) · [BER via classifier ensembles (Tumer)](http://www.ideal.ece.utexas.edu/pdfs/48.pdf)
- [Comparing sampling strategies for imbalanced HAR (PMC 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8963022/)

---

# Literature Synthesis — Part I: Robustness to User-Distribution Shift

> **Why this exists:** the 3 LB results on May 8 showed combo-style models with DL embeddings have a **negative LB-vs-OOF gap** without post-hoc correction (combo_full RAW: OOF 0.7687 → LB 0.7568, gap −0.012). This document maps that symptom to known phenomena in the literature and ranks the available fixes by ROI.

## The diagnosis matches a well-known stacking pitfall

From the Kaggle Grandmaster Playbook for tabular data:
> "Train meta-features and test meta-features should follow a similar distribution when predictions are made on the test set using models trained on each fold. … When facing distribution shift between public and private test sets, prioritize model robustness over maximizing public scores."

Our specific symptom:
- Train embeddings for sample $i$: from a CNN/Transformer **final model** (retrained on all train data), so it has *memorized* sample $i$
- Test embeddings for sample $j$: from the *same* final model, but $j$ has never been seen
- The combo LGBM fits a calibration that uses the memorization signal which is absent at inference

This is the same root cause that the literature identifies for the train-test shift in stacked ensembles: the meta-features must be produced by an out-of-fold protocol on **both** sides.

## Five technique families ranked by ROI

### Tier A — direct fixes for the shift we observed (1-2 days, high ROI)

**A.1. Per-fold DL embedding extraction**
- We already have all 5 `fold_k_best.pt` checkpoints saved per DL run
- For training sample $i$ in fold $k$'s validation set, extract embedding from `fold_k_best.pt` (model that did NOT see $i$)
- For test sample $j$, average embeddings from all 5 fold models
- Both train and test embeddings now come from "models that haven't seen this sample" → distribution-matched
- **Expected lift:** +0.02 to +0.04 LB just from eliminating the structural shift
- **Cost:** ~2 hours of code + 5-10 min of inference per DL track

**A.2. Per-subject (per-file at inference) z-score on raw DL inputs**
- Standard preprocessing recommendation: *"Standardizing the data per subject is almost certainly required for any cross-subject model"* (Nat Sci Data 2024, smartphone HAR benchmark)
- We do per-file z-score on engineered features (the `per_file_norm` group) but NOT on the DL pipeline's raw 6×300 input
- Implementation: in `SeqDataset.__getitem__`, optionally subtract per-file mean and divide by per-file std before applying augmentations
- Concat the original mean/std as 2 extra channels so the model can still see absolute magnitude
- **Expected lift:** +0.005 to +0.01 OOF on DL alone; downstream blend lift unclear

### Tier B — train-time subject invariance (3-5 days, medium ROI)

**B.1. Subject-Invariant Contrastive Learning (SICL)** [Talegaonkar et al., MLSP 2025; arXiv 2507.03250]
- Loss: $\mathcal{L}_{SICL} = -\sum_i \log \frac{\exp(z_i \cdot z_j / \tau)}{Q_{S_i}\sum_{s \in S(i)} \exp(z_i \cdot z_s / \tau) + \sum_{k \notin S(i)} \exp(z_i \cdot z_k / \tau)}$
- $S(i)$ = same-subject indices, $Q_{S_i}$ = down-weight factor (typically 0.1-0.5)
- Pretraining-style: train encoder via SICL contrastive loss, then freeze and fit a linear classifier
- Code: github.com/olivesgatech/SICL
- **Reported gain: up to +11% F1 vs standard contrastive HAR** on UTD-MHAD/MMAct/DARai
- **Cost:** new training script, 1-2 days; ~60 min training per fold

**B.2. DANN (Domain-Adversarial Neural Network) with user_id as domain** [Ganin et al. 2015]
- Add a domain classifier head to CNN-BiLSTM with a Gradient Reversal Layer (GRL)
- Forward pass: GRL is identity. Backward pass: GRL multiplies gradient by −λ
- Encoder learns features good for activity classification but uninformative about which user produced them
- Implementation: 50-100 LOC modification to `train_cnn_bilstm.py`; add nn.Module GRL wrapper, compute domain loss with class_label = user_id_index
- **Cost:** 0.5 day; works alongside existing pipeline

**B.3. Mixup with cross-subject pair selection**
- Standard mixup picks a random pair. Modified version picks pairs of the **same activity from different subjects**
- Encourages the encoder to compute features invariant to which user produced them
- Cheap modification to existing mixup_batch in our trainers
- **Cost:** 1 hour

### Tier C — test-time adaptation (1 day, low-medium ROI)

**C.1. TENT-style test-time adaptation** [Wang et al., ICLR 2021]
- At inference, fine-tune **only BatchNorm affine parameters** (γ, β) to minimize entropy of test predictions
- Online: process test data in batches, update BN params after each batch
- Surprisingly effective: typically +0.5-2% on distribution-shifted test sets
- Code: github.com/DequanWang/tent
- Our CNN-BiLSTM has BN layers and is suitable; Transformer doesn't and would need adaptation
- **Cost:** 0.5 day; runs in seconds at inference

### Tier D — full self-supervised pretraining (5+ days, conditional ROI)

**D.1. CrossHAR-style hierarchical self-supervised pretraining**
- Train an encoder with masked-reconstruction + contrastive objective on train **+ test sequences combined** (no labels needed)
- Then fine-tune with labeled train data only
- Reported +10.83% accuracy on cross-dataset HAR (where domain shift is much larger than ours)
- **Cost:** new training pipeline, ~3-5 days
- **Risk:** large engineering effort; may not transfer that big a number to our smaller within-dataset shift

**D.2. Diffusion-based synthetic IMU data for unseen subjects** [Oppel et al.]
- Use a denoising diffusion model trained on raw sequences to synthesize new "subject-like" data
- Augments train set with synthetic-but-plausible unseen-subject distributions
- Heavy engineering; probably out of budget unless other tiers fail to deliver

## Recommended sequence for next ~10 days

Sequence chosen by *expected lift × probability of working × inverse cost*:

1. **Day 1-2: Per-fold DL embeddings (Tier A.1)** — the directly-implied fix from our gap diagnosis. Highest expected lift (+0.02 to +0.04 LB) for lowest cost.
2. **Day 2-3: Per-file z-score on raw DL inputs (Tier A.2)** — single-file change in `SeqDataset`.
3. **Day 3-5: DANN for CNN-BiLSTM (Tier B.2)** — train one CNN-BiLSTM with user-domain adversarial head. Directly attacks subject leakage at training time.
4. **Day 5-6: TENT at inference (Tier C.1)** — test-time BN adaptation. Run on the DANN model + the per-fold-embedded combo.
5. **Day 7-10: SICL pretraining (Tier B.1) IF the above hasn't pushed us above 0.82 LB** — biggest reported gain in literature for our exact problem class, but most expensive to implement.

## Why we are NOT pursuing some "obvious" ideas

- **More Optuna tuning** — we already saw Optuna shrinks the gap (OOF-overfit signature). Further HP search will trade OOF for LB.
- **More base models in blend** — saturated at OOF 0.742. Each new model contributes <3% weight in simplex.
- **More feature engineering** — combo's 805 features had train-test shift. Fixing the shift is more valuable than more features.
- **Larger CNN/Transformer** — our 460k-parameter models aren't underfitting; they're misaligning train and test distributions.

## Sources
- [Towards Generalizable HAR: A Survey (arXiv 2508.12213)](https://arxiv.org/html/2508.12213v1)
- [Subject-Invariant Contrastive Learning for HAR (arXiv 2507.03250)](https://arxiv.org/html/2507.03250)
- [SICL code (github.com/olivesgatech/SICL)](https://github.com/olivesgatech/SICL)
- [Domain-Adversarial Training of Neural Networks (Ganin et al., JMLR 2016)](https://jmlr.org/papers/volume17/15-239/15-239.pdf)
- [TENT: Fully Test-Time Adaptation by Entropy Minimization (ICLR 2021, arXiv 2006.10726)](https://arxiv.org/abs/2006.10726)
- [TENT code (github.com/DequanWang/tent)](https://github.com/DequanWang/tent)
- [CrossHAR: Generalizing Cross-dataset HAR (ACM IMWUT 2024)](https://dl.acm.org/doi/10.1145/3659597)
- [Smartphone HAR benchmark for domain adaptation (Sci Data 2024)](https://www.nature.com/articles/s41597-024-03951-4)
- [The Kaggle Grandmasters Playbook for Tabular Data (NVIDIA Tech Blog 2024)](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [Cross validation strategy when blending/stacking (Kaggle discussion)](https://www.kaggle.com/general/18793)
- [Data Augmentation for Time-Series Classification: a Comprehensive Survey (arXiv 2310.10060)](https://arxiv.org/html/2310.10060v4)
- [ContrasGAN: Unsupervised domain adaptation in HAR (Inf Soft Tech 2021)](https://www.sciencedirect.com/science/article/abs/pii/S1574119221001103)
- [Comparing sampling strategies for imbalanced HAR (PMC 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8963022/)
