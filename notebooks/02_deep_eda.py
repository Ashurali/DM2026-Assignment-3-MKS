# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
# ---
# %% [markdown]
# # DM2026 Assignment 3 — Deep EDA
#
# Phase-2 analysis going beyond the initial 7-section EDA.
# Covers: distributional shape, per-class signature, confusion structure,
# time-domain patterns, frequency-domain deep dive, feature redundancy,
# hard-example mining (using current best OOF probs), and per-user outliers.
#
# Output: `reports/eda_deep_summary.md` + `reports/figures_deep/*.png`.

# %%
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
# Force non-interactive backend so plt.show() doesn't block in headless runs.
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scistats
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
ROOT = HERE.parent if HERE.name == "notebooks" else HERE
FIG = ROOT / "reports" / "figures_deep"
REPORTS = ROOT / "reports"
FIG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style="whitegrid", context="notebook")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CHANNELS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
N_CLASSES = 6

findings: dict = {}


# %% [markdown]
# ## Load data + best OOF probs

# %%
meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
y = meta_train["label"].values
groups = meta_train["user_id"].values
print(f"Train: {len(meta_train)} files | Test: {len(meta_test)} files")
print(f"Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")

# Best OOF (combo with cal+thresh; OOF probs are pre-cal so use raw)
combo_oof = np.load(ROOT / "oof" / "lgbm_combo_combo_full_oof.npy")
print(f"Best model OOF: {combo_oof.shape}")
print(f"Best model OOF macro F1: {f1_score(y, combo_oof.argmax(axis=1), average='macro'):.4f}")


# %% [markdown]
# ## Section 1 — Per-channel distribution shape
#
# Skewness, kurtosis, and tail behaviour across all 11k files.

# %%
chan_stats = []
for ch in CHANNELS:
    vals = meta_train[f"feat_{ch}"].values
    chan_stats.append({
        "channel": ch,
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "skew": float(scistats.skew(vals)),
        "kurt": float(scistats.kurtosis(vals)),
        "p1": float(np.percentile(vals, 1)),
        "p99": float(np.percentile(vals, 99)),
        "n_outlier_3sigma": int(np.sum(np.abs(vals - np.mean(vals)) > 3 * np.std(vals))),
    })
chan_stats_df = pd.DataFrame(chan_stats)
print(chan_stats_df.round(3).to_string(index=False))

# Plot histograms for each channel
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for i, ch in enumerate(CHANNELS):
    ax = axes[i // 3, i % 3]
    vals = meta_train[f"feat_{ch}"].values
    sns.histplot(vals, bins=80, ax=ax, kde=True, stat="density")
    ax.set(title=f"{ch}  (skew={scistats.skew(vals):.2f}, kurt={scistats.kurtosis(vals):.2f})", xlabel="")
plt.tight_layout()
plt.savefig(FIG / "s1_channel_distributions.png", dpi=110, bbox_inches="tight")
plt.show()

findings["section1_channel_distributions"] = chan_stats


# %% [markdown]
# ### Interpretation §1
#
# Look for: heavy tails, asymmetry, multi-modality. mean_x/y/z encode gravity
# orientation, std_x/y/z encode activity intensity. Distinct shapes drive what
# preprocessing/transformations matter.

# %% [markdown]
# ## Section 2 — Per-class × per-channel statistical signature

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, ch in enumerate(CHANNELS):
    ax = axes[i // 3, i % 3]
    sns.violinplot(data=meta_train, x="label", y=f"feat_{ch}", ax=ax, inner="quartile",
                    palette="Set2", hue="label", legend=False)
    ax.set(title=ch)
plt.suptitle("Per-class violin plots (window-mean of each channel)", y=1.005)
plt.tight_layout()
plt.savefig(FIG / "s2_class_violin.png", dpi=110, bbox_inches="tight")
plt.show()

# Per-class summary table
per_class = meta_train.groupby("label")[[f"feat_{ch}" for ch in CHANNELS]].agg(["mean", "std"])
per_class.columns = [f"{c[0].replace('feat_', '')}_{c[1]}" for c in per_class.columns]
print("Per-class mean/std of each channel:")
print(per_class.round(3))

findings["section2_per_class_summary"] = per_class.round(4).reset_index().to_dict(orient="records")


# %% [markdown]
# ### Interpretation §2
#
# Violins show within-class spread per channel. Look for class pairs with
# overlapping distributions — those are the confusable ones.

# %% [markdown]
# ## Section 3 — Class-pair distance matrix (centroid Euclidean in feature space)

# %%
# Compute centroid in 6-feature space per class
feat_cols = [f"feat_{ch}" for ch in CHANNELS]
centroids = meta_train.groupby("label")[feat_cols].mean().values
distances = np.zeros((N_CLASSES, N_CLASSES))
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])

print("Pairwise centroid distance (smaller = more confusable):")
dist_df = pd.DataFrame(distances, index=[f"L{i}" for i in range(N_CLASSES)],
                        columns=[f"L{j}" for j in range(N_CLASSES)])
print(dist_df.round(3))

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(dist_df, annot=True, fmt=".2f", cmap="viridis_r", ax=ax, cbar_kws={"label": "Euclidean dist (smaller = closer)"})
ax.set(title="Inter-class centroid distance (6-feature simple representation)")
plt.tight_layout()
plt.savefig(FIG / "s3_centroid_distance.png", dpi=110, bbox_inches="tight")
plt.show()

# Closest off-diagonal pairs
pairs = []
for i in range(N_CLASSES):
    for j in range(i + 1, N_CLASSES):
        pairs.append((i, j, distances[i, j]))
pairs.sort(key=lambda x: x[2])
print("\nClosest class pairs (most confusable on these 6 features):")
for i, j, d in pairs[:5]:
    print(f"  L{i}-L{j}: dist = {d:.3f}")

findings["section3_centroid_distance"] = dist_df.round(4).to_dict()
findings["section3_closest_pairs"] = [{"a": int(i), "b": int(j), "dist": float(d)} for i, j, d in pairs]


# %% [markdown]
# ## Section 4 — Confusion matrix from current best OOF

# %%
best_oof_preds = combo_oof.argmax(axis=1)
cm = confusion_matrix(y, best_oof_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
            xticklabels=[f"L{i}" for i in range(N_CLASSES)],
            yticklabels=[f"L{i}" for i in range(N_CLASSES)])
axes[0].set(title=f"Raw counts — combo OOF\n(macro F1 = {f1_score(y, best_oof_preds, average='macro'):.4f})",
            xlabel="predicted", ylabel="true")

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Reds", ax=axes[1], cbar=False,
            xticklabels=[f"L{i}" for i in range(N_CLASSES)],
            yticklabels=[f"L{i}" for i in range(N_CLASSES)])
axes[1].set(title="Row-normalized (recall view)", xlabel="predicted", ylabel="true")
plt.tight_layout()
plt.savefig(FIG / "s4_confusion_matrix.png", dpi=110, bbox_inches="tight")
plt.show()

print("\nMost-confused class pairs (off-diagonal cells with largest counts):")
off_diag = []
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if i != j:
            off_diag.append((i, j, cm[i, j], cm_norm[i, j]))
off_diag.sort(key=lambda x: -x[2])
for i, j, c, frac in off_diag[:8]:
    print(f"  true L{i} → pred L{j}: {c} files ({frac:.1%} of L{i})")

findings["section4_confusion"] = {
    "matrix": cm.tolist(),
    "matrix_normalized": cm_norm.round(4).tolist(),
    "top_confusions": [{"true": int(i), "pred": int(j), "count": int(c), "fraction": float(f)} for i, j, c, f in off_diag[:8]],
}


# %% [markdown]
# ## Section 5 — Feature correlation analysis (271 engineered features)

# %%
# Use cached feature parquet
feat_train = pd.read_parquet(ROOT / "data" / "feat_train_none.parquet")
fcols = [c for c in feat_train.columns if c != "file_id"]
print(f"Engineered features: {len(fcols)}")

# Subsample features for visualization (271 × 271 is big)
# Use full matrix for redundancy detection though
corr = feat_train[fcols].corr().abs()
print("Computing correlation matrix...", corr.shape)

# Top redundant pairs
redundant = []
for i in range(len(fcols)):
    for j in range(i + 1, len(fcols)):
        c = corr.iloc[i, j]
        if c > 0.95:
            redundant.append((fcols[i], fcols[j], float(c)))
redundant.sort(key=lambda x: -x[2])
print(f"\nFound {len(redundant)} feature pairs with |corr| > 0.95")
print("Top 10:")
for a, b, c in redundant[:10]:
    print(f"  {a}  ↔  {b}  : {c:.3f}")

# Visualize correlation matrix (downsample to 80 features for clarity)
np.random.seed(42)
sample_fcols = sorted(np.random.choice(fcols, size=min(80, len(fcols)), replace=False))
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr.loc[sample_fcols, sample_fcols], cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1,
            xticklabels=False, yticklabels=False, cbar_kws={"label": "|corr|"})
ax.set(title=f"Feature correlation matrix (random {len(sample_fcols)} of {len(fcols)})")
plt.tight_layout()
plt.savefig(FIG / "s5_feature_corr_heatmap.png", dpi=110, bbox_inches="tight")
plt.show()

# Histogram of all pairwise |corr| values
upper = corr.values[np.triu_indices_from(corr.values, k=1)]
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(upper, bins=80, color="#4c72b0", alpha=0.85)
ax.axvline(0.95, color="r", linestyle="--", label=">0.95 threshold")
ax.set(title=f"All-pairs |corr| histogram — {len(upper)} feature pairs",
       xlabel="|Pearson r|", ylabel="frequency")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "s5_feature_corr_hist.png", dpi=110, bbox_inches="tight")
plt.show()

findings["section5_feature_correlation"] = {
    "n_features": len(fcols),
    "n_pairs_above_0.95": len(redundant),
    "top_redundant_pairs": [{"a": a, "b": b, "corr": c} for a, b, c in redundant[:20]],
    "median_pairwise_abs_corr": float(np.median(upper)),
    "p95_pairwise_abs_corr": float(np.percentile(upper, 95)),
}


# %% [markdown]
# ## Section 6 — Hard-example detection
#
# Files where the best model is uncertain (low max prob) or confidently wrong.
# These are the targets for further model attention.

# %%
oof_max_prob = combo_oof.max(axis=1)
oof_is_correct = (best_oof_preds == y)

# Per-class confidence distribution
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for c in range(N_CLASSES):
    ax = axes[c // 3, c % 3]
    mask = (y == c)
    correct = oof_max_prob[mask & oof_is_correct]
    wrong = oof_max_prob[mask & ~oof_is_correct]
    ax.hist(correct, bins=30, alpha=0.6, label=f"correct ({len(correct)})", color="#55a868")
    ax.hist(wrong, bins=30, alpha=0.6, label=f"wrong ({len(wrong)})", color="#c44e52")
    ax.set(title=f"L{c} — N={int(mask.sum())}", xlabel="max softmax prob", xlim=(0, 1))
    ax.legend(fontsize=8)
plt.suptitle("Confidence distribution per class (combo OOF)", y=1.005)
plt.tight_layout()
plt.savefig(FIG / "s6_confidence_per_class.png", dpi=110, bbox_inches="tight")
plt.show()

# Confidence statistics per class
conf_stats = []
for c in range(N_CLASSES):
    mask = (y == c)
    correct_mask = mask & oof_is_correct
    wrong_mask = mask & ~oof_is_correct
    conf_stats.append({
        "class": int(c),
        "n_total": int(mask.sum()),
        "n_correct": int(correct_mask.sum()),
        "n_wrong": int(wrong_mask.sum()),
        "mean_conf_correct": float(oof_max_prob[correct_mask].mean()) if correct_mask.sum() > 0 else 0.0,
        "mean_conf_wrong": float(oof_max_prob[wrong_mask].mean()) if wrong_mask.sum() > 0 else 0.0,
        "wrong_rate": float(wrong_mask.sum() / max(1, mask.sum())),
    })
conf_df = pd.DataFrame(conf_stats)
print(conf_df.round(3).to_string(index=False))

# "Confidently wrong" — where the model is very sure but still wrong
confidently_wrong = (~oof_is_correct) & (oof_max_prob > 0.85)
print(f"\nConfidently wrong (prob > 0.85, prediction != truth): {int(confidently_wrong.sum())} files")
print("Per-class breakdown of confidently-wrong:")
for c in range(N_CLASSES):
    n = int(np.sum(confidently_wrong & (y == c)))
    print(f"  L{c}: {n}")

findings["section6_hard_examples"] = {
    "per_class_confidence": conf_stats,
    "n_confidently_wrong": int(confidently_wrong.sum()),
}


# %% [markdown]
# ## Section 7 — t-SNE on engineered feature space (deep version)

# %%
# Stratified subsample
SUB_PER_CLASS = 400
sub_idx = []
rng = np.random.default_rng(RANDOM_STATE)
for c in range(N_CLASSES):
    pool = np.where(y == c)[0]
    pick = rng.choice(pool, size=min(SUB_PER_CLASS, len(pool)), replace=False)
    sub_idx.extend(pick.tolist())

X_for_tsne = feat_train.iloc[sub_idx][fcols].values
y_sub = y[sub_idx]
groups_sub = groups[sub_idx]

# PCA→t-SNE for stability with high-dim input
print("Running PCA(50)...")
pca = PCA(n_components=50, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_for_tsne)
print(f"  Explained var ratio (cumulative top 50): {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

print("Running t-SNE (this may take ~1-2 min)...")
tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto", init="pca", random_state=RANDOM_STATE, max_iter=1000)
emb = tsne.fit_transform(X_pca)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# (a) colored by class
for c in range(N_CLASSES):
    mask = y_sub == c
    axes[0].scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6, label=f"L{c}")
axes[0].legend(markerscale=2, fontsize=9)
axes[0].set(title="t-SNE on 271 engineered features — by class", xticks=[], yticks=[])

# (b) colored by user
unique_users = np.unique(groups_sub)
cmap = plt.cm.tab20
for i, u in enumerate(unique_users[:20]):  # first 20 users for legibility
    mask = groups_sub == u
    axes[1].scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.5, color=cmap(i % 20))
axes[1].set(title=f"t-SNE — by user (first 20 of {len(unique_users)})", xticks=[], yticks=[])
plt.tight_layout()
plt.savefig(FIG / "s7_tsne_full_features.png", dpi=110, bbox_inches="tight")
plt.show()

findings["section7_tsne"] = {
    "n_points": len(sub_idx),
    "pca_explained_top50": float(pca.explained_variance_ratio_.cumsum()[-1]),
}


# %% [markdown]
# ## Section 8 — Per-user signature outliers

# %%
# Compute per-user mean profile (mean of each feat per user)
user_profile = feat_train.copy()
user_profile["user_id"] = groups
user_means = user_profile.groupby("user_id")[fcols].mean()
print(f"User profiles: {user_means.shape}")

# Outlier detection: each user's distance to the global mean
global_mean = user_means.mean()
global_std = user_means.std() + 1e-9
zscores = ((user_means - global_mean) / global_std).abs()
user_outlier_score = zscores.mean(axis=1)
print("\nTop 5 outlier users (highest mean |z| across features):")
print(user_outlier_score.sort_values(ascending=False).head(5).round(3))
print("\nMost-typical 5 users (lowest mean |z|):")
print(user_outlier_score.sort_values(ascending=True).head(5).round(3))

# Plot user outlier score
fig, ax = plt.subplots(figsize=(11, 4))
sorted_scores = user_outlier_score.sort_values()
ax.bar(range(len(sorted_scores)), sorted_scores.values, color="#4c72b0")
ax.set_xticks(range(len(sorted_scores)))
ax.set_xticklabels(sorted_scores.index, rotation=90, fontsize=7)
ax.set(title="Per-user outlier score (mean |z| across features)", ylabel="mean |z|")
ax.axhline(sorted_scores.mean(), color="r", linestyle="--", label="mean")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "s8_user_outliers.png", dpi=110, bbox_inches="tight")
plt.show()

findings["section8_user_outliers"] = {
    "top_outliers": user_outlier_score.sort_values(ascending=False).head(5).round(3).to_dict(),
    "most_typical": user_outlier_score.sort_values(ascending=True).head(5).round(3).to_dict(),
}


# %% [markdown]
# ## Section 9 — L2 deep dive (the bottleneck class)

# %%
l2_mask = y == 2
l2_files = meta_train[l2_mask]
print(f"L2 files: {l2_mask.sum()}")
print(f"L2 users (have ≥1 L2 file): {l2_files['user_id'].nunique()}")
print(f"L2 user distribution (top 5):")
print(l2_files["user_id"].value_counts().head(5))

# Where does the model think L2 files belong?
l2_oof = combo_oof[l2_mask]
l2_preds = l2_oof.argmax(axis=1)
print("\nWhat does the model predict for true-L2 files?")
for c in range(N_CLASSES):
    n = int(np.sum(l2_preds == c))
    print(f"  → L{c}: {n} files ({n/l2_mask.sum():.1%})")

# Average probability vector for true-L2 files (showing where the mass goes)
avg_prob = l2_oof.mean(axis=0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(N_CLASSES), avg_prob, color="#4c72b0")
ax.set_xticks(range(N_CLASSES))
ax.set_xticklabels([f"L{i}" for i in range(N_CLASSES)])
ax.set(title="Average softmax prob across all true-L2 files", ylabel="mean P(class)")
for i, v in enumerate(avg_prob):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "s9_l2_average_prob.png", dpi=110, bbox_inches="tight")
plt.show()

findings["section9_l2_deep_dive"] = {
    "n_l2_files": int(l2_mask.sum()),
    "n_l2_users": int(l2_files["user_id"].nunique()),
    "l2_predictions": {int(c): int(np.sum(l2_preds == c)) for c in range(N_CLASSES)},
    "l2_avg_softmax": [float(x) for x in avg_prob],
}


# %% [markdown]
# ## Section 10 — Time-domain pattern analysis (sample)

# %%
# Load actual sequences for a small per-class sample (skipped if cache absent)
seq_path = ROOT / "data" / "seq_train.npy"
if not seq_path.exists():
    print(f"Skipping §10 — {seq_path.name} not present locally (server-only).")
    findings["section10_time_domain"] = None
else:
    seq_train = np.load(seq_path)
    print("Sample 50 files per class; compute trend (slope of mean over time).")
    trend_data = []
    for c in range(N_CLASSES):
        pool = np.where(y == c)[0]
        pick = rng.choice(pool, size=min(50, len(pool)), replace=False)
        for idx in pick:
            seq = seq_train[idx]
            mag = np.sqrt(seq[0]**2 + seq[1]**2 + seq[2]**2)
            slope = np.polyfit(np.arange(300), mag, 1)[0]
            std_intensity = np.sqrt(seq[3]**2 + seq[4]**2 + seq[5]**2).mean()
            trend_data.append({"label": c, "mag_slope": float(slope), "std_intensity": float(std_intensity)})

    trend_df = pd.DataFrame(trend_data)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.boxplot(data=trend_df, x="label", y="mag_slope", ax=axes[0], palette="Set2", hue="label", legend=False)
    axes[0].set(title="Trend (slope of |mean| over 300s) per class", ylabel="slope")
    sns.boxplot(data=trend_df, x="label", y="std_intensity", ax=axes[1], palette="Set2", hue="label", legend=False)
    axes[1].set(title="Mean std-channel intensity per class", ylabel="mean(|std|)")
    plt.tight_layout()
    plt.savefig(FIG / "s10_time_domain.png", dpi=110, bbox_inches="tight")
    plt.show()
    findings["section10_time_domain"] = trend_df.groupby("label").agg(["mean", "std"]).round(4).reset_index().to_dict(orient="records")


# %% [markdown]
# ## Save findings

# %%
with open(REPORTS / "eda_deep_summary.json", "w", encoding="utf-8") as f:
    json.dump(findings, f, indent=2, default=str)
print(f"Wrote {REPORTS / 'eda_deep_summary.json'}")
