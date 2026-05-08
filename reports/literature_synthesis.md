# Literature Synthesis — Robustness to User-Distribution Shift

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
