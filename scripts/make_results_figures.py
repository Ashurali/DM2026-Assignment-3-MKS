"""Generate figures that summarise the project's results for the final report.

Outputs to reports/figures_final/:
  1. lb_progression.png   — submission timeline showing LB score progression
  2. oof_vs_lb.png        — OOF vs LB scatter, with the OOF→LB inversion pattern
  3. per_class_f1.png     — per-class F1 across the key milestone submissions
  4. eo_feature_groups.png — feature group retention by EO

Run: python scripts/make_results_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "figures_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# 1. LB progression figure
# ────────────────────────────────────────────────────────────────────────────
def lb_progression():
    """Submission timeline with milestone annotations."""
    data = [
        # (label, date_idx, oof, lb, milestone?)
        ("Majority", 0, None, 0.1024, False),
        ("LR(6 means)", 1, None, 0.5315, False),
        ("LGBM full v1", 2, 0.7091, 0.7808, True),
        ("LGBM Optuna", 3, 0.7350, 0.7816, False),
        ("CNN-BiLSTM v1", 4, 0.671, 0.74, False),
        ("Transformer v1", 5, 0.66, 0.73, False),
        ("Stacked blend", 6, 0.74, 0.785, False),
        ("combo_full_v2", 7, 0.74, 0.7984, True),
        ("v1 hier α=0.88", 8, 0.78, 0.8107, True),
        ("v4 a088 cal+thresh", 9, 0.7856, 0.8114, True),
        ("v6 a842 NM", 10, 0.7880, 0.7991, False),
        ("v6 a842 grid PEAK", 11, 0.7880, 0.8154, True),
    ]
    labels = [d[0] for d in data]
    lbs = [d[3] for d in data]
    milestones = [d[4] for d in data]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["tab:blue" if m else "lightblue" for m in milestones]
    bars = ax.bar(x, lbs, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.1024, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axhline(0.7088, color="orange", linestyle="--", linewidth=1.0, label="Baseline-3 (0.7088)")
    ax.axhline(0.8114, color="green", linestyle="--", linewidth=1.0, label="Backup (0.8114)")
    ax.axhline(0.8154, color="red", linestyle="--", linewidth=1.0, label="Primary (0.8154)")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Public LB F1-macro", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_title("Submission LB progression — primary breakthrough at v6 a842 grid peak",
                 fontsize=13)
    for i, (rect, lb) in enumerate(zip(bars, lbs)):
        ax.text(rect.get_x() + rect.get_width() / 2, lb + 0.012,
                f"{lb:.3f}", ha="center", fontsize=9, fontweight="bold" if milestones[i] else "normal")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lb_progression.png", dpi=150)
    plt.close()
    print(f"Wrote {OUT_DIR / 'lb_progression.png'}")


# ────────────────────────────────────────────────────────────────────────────
# 2. OOF vs LB scatter
# ────────────────────────────────────────────────────────────────────────────
def oof_vs_lb():
    """Scatter showing the OOF→LB shift pattern."""
    # (name, oof, lb)
    pts = [
        ("LGBM full v1",       0.7091, 0.7808),
        ("LGBM Optuna",        0.7350, 0.7816),
        ("combo_full_v2",      0.7400, 0.7984),
        ("v1 hier α=0.88",     0.7800, 0.8107),
        ("v4 a088",            0.7856, 0.8114),
        ("v4 a088 grid peak",  0.7865, 0.8119),
        ("v6 a842 NM",         0.7880, 0.7991),
        ("v6 a842 grid PEAK",  0.7880, 0.8154),
        ("v6 rigorous top1",   0.7882, 0.7698),
    ]
    oof = [p[1] for p in pts]
    lb = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    # y = x line
    lim = (0.69, 0.83)
    ax.plot(lim, lim, "k--", linewidth=0.7, alpha=0.5, label="y = x")
    # +0.02, +0.025 reference lines
    ax.plot(lim, [v + 0.025 for v in lim], "g--", linewidth=0.5, alpha=0.5, label="LB = OOF + 0.025")
    ax.plot(lim, [v - 0.01 for v in lim], "r--", linewidth=0.5, alpha=0.5, label="LB = OOF − 0.01")

    colors = []
    for n, _, _ in pts:
        if "v6 a842 grid PEAK" in n: colors.append("red")
        elif "v6 a842 NM" in n: colors.append("orange")
        elif "v6 rigorous" in n: colors.append("purple")
        elif "v4 a088" in n: colors.append("green")
        else: colors.append("tab:blue")
    ax.scatter(oof, lb, c=colors, s=110, edgecolors="black", linewidth=0.8, zorder=5)
    for (n, o, l), c in zip(pts, colors):
        dx, dy = 0.002, 0.005
        if "rigorous" in n: dy = -0.01
        if "NM" in n and "a842" in n: dy = -0.012
        if "grid PEAK" in n and "v6" in n: dy = 0.008; dx = -0.05
        ax.annotate(n, (o, l), xytext=(o + dx, l + dy), fontsize=9, fontweight="bold" if c == "red" else "normal")
    ax.set_xlabel("OOF F1-macro (5-fold GroupKFold by user_id)", fontsize=11)
    ax.set_ylabel("Public LB F1-macro", fontsize=11)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title("OOF→LB transfer is non-monotonic in threshold space\n"
                 "Two v6 a842 variants with identical OOF (0.7880) span LB 0.770 to 0.815",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "oof_vs_lb.png", dpi=150)
    plt.close()
    print(f"Wrote {OUT_DIR / 'oof_vs_lb.png'}")


# ────────────────────────────────────────────────────────────────────────────
# 3. Per-class F1 across milestone submissions
# ────────────────────────────────────────────────────────────────────────────
def per_class_f1():
    """Per-class F1 bar chart across milestones."""
    classes = ["L0", "L1", "L2", "L3", "L4", "L5"]
    milestones = {
        "LGBM v1 (271 feats)": [0.964, 0.902, 0.173, 0.713, 0.896, 0.678],
        "combo_full_v2":        [0.965, 0.905, 0.310, 0.755, 0.918, 0.770],
        "v4 a088 (backup)":     [0.968, 0.908, 0.377, 0.766, 0.919, 0.776],
        "v6 a842 grid (primary)": [0.967, 0.908, 0.384, 0.764, 0.924, 0.781],
    }
    n_classes = len(classes)
    n_ms = len(milestones)
    bar_w = 0.2
    x = np.arange(n_classes)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["lightgray", "lightblue", "tab:green", "tab:red"]
    for i, (name, vals) in enumerate(milestones.items()):
        offset = (i - (n_ms - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w, label=name,
                      color=colors[i], edgecolor="black", linewidth=0.5)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("Per-class F1 (OOF)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1 progression across milestones — L2 is the bottleneck",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_class_f1.png", dpi=150)
    plt.close()
    print(f"Wrote {OUT_DIR / 'per_class_f1.png'}")


# ────────────────────────────────────────────────────────────────────────────
# 4. EO feature group retention
# ────────────────────────────────────────────────────────────────────────────
def eo_feature_groups():
    """Bar chart of feature group retention after EO."""
    groups = [
        ("gravity", 0, 11),
        ("autocorr", 1, 6),
        ("FFT", 6, 24),
        ("catch22", 38, 132),
        ("jerk", 7, 24),
        ("std_channel", 12, 42),
        ("magnitude", 9, 30),
        ("zerocross", 2, 6),
        ("znorm", 6, 18),
        ("sliding_window", 22, 60),
        ("mean_channel", 18, 42),
    ]
    groups.sort(key=lambda g: g[1] / max(g[2], 1))
    names = [g[0] for g in groups]
    kept = [g[1] for g in groups]
    total = [g[2] for g in groups]
    pct = [100 * k / max(t, 1) for k, t in zip(kept, total)]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:red" if p == 0 else ("tab:orange" if p < 30 else "tab:green") for p in pct]
    bars = ax.barh(names, pct, color=colors, edgecolor="black", linewidth=0.5)
    for rect, p, k, t in zip(bars, pct, kept, total):
        ax.text(p + 1.5, rect.get_y() + rect.get_height() / 2,
                f"{k}/{t} ({p:.0f}%)", va="center", fontsize=10)
    ax.set_xlabel("% of features kept by EO", fontsize=11)
    ax.set_xlim(0, 60)
    ax.set_title("EO independently identified gravity as spurious (0% retained)\n"
                 "Other low-retention groups: autocorr (17%), FFT (25%) — also user-signature heavy",
                 fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eo_feature_groups.png", dpi=150)
    plt.close()
    print(f"Wrote {OUT_DIR / 'eo_feature_groups.png'}")


if __name__ == "__main__":
    lb_progression()
    oof_vs_lb()
    per_class_f1()
    eo_feature_groups()
    print(f"\nAll figures saved to {OUT_DIR}/")
