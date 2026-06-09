"""Generate the figures for the IEEE report:
  report/figures/kaggle_progression.png  — public-LB climb by milestone (real Kaggle scores)
  report/figures/pipeline.png             — final-method framework diagram
  report/figures/per_class_f1.png         — per-class F1, baseline vs final
All numbers are the verified Kaggle public scores (see TASK_STATE.md). No invented data.
Run: python scripts/make_report_figures.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

    # ---------------- Figure 1: Kaggle progression ----------------
    # (label, public LB, group) chronological; group drives colour
    mile = [
        ("Majority baseline", 0.1024, "base"),
        ("LightGBM, basic feats", 0.7473, "base"),
        ("LightGBM, 271-feat catalog", 0.7808, "feat"),
        ("Combo stack + isotonic + thresh", 0.7984, "feat"),
        ("Hierarchical + EO + blend", 0.8154, "feat"),
        ("+ Orientation pseudo-gyro", 0.8200, "win"),
        ("+ Test-prior corr. (β=1)", 0.8220, "win"),
        ("+ Test-prior corr. (β=2.0)", 0.8234, "best"),
    ]
    colmap = {"base": "#9aa0a6", "feat": "#4285f4", "win": "#34a853", "best": "#ea4335"}
    labels = [m[0] for m in mile][::-1]
    vals = [m[1] for m in mile][::-1]
    cols = [colmap[m[2]] for m in mile][::-1]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    bars = ax.barh(labels, vals, color=cols, edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("Kaggle public-leaderboard macro-F1")
    ax.set_title("Public-LB progression by milestone", fontsize=10)
    ax.axvline(0.8234, color="#ea4335", ls="--", lw=0.7, alpha=0.6)
    ax.grid(axis="x", ls=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(FIG / "kaggle_progression.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote kaggle_progression.png")

    # ---------------- Figure 2: pipeline / framework diagram ----------------
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")

    def box(x, y, w, h, text, fc="#e8f0fe", ec="#4285f4"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.2)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color="#5f6368", lw=1.1))

    box(5, 11.2, 8.2, 1.0, "RAW  6 channels × 300 s  (1 Hz, gravity-laden, wrist)", fc="#f1f3f4", ec="#9aa0a6")
    arrow(5, 10.7, 5, 10.2)
    box(5, 9.7, 8.2, 1.0, "FEATURES:  271 stats + catch22 + 11 families  +  orientation pseudo-gyro dynamics")
    arrow(5, 9.2, 3.2, 8.6); arrow(5, 9.2, 6.8, 8.6)
    box(3.0, 8.1, 3.6, 1.0, "P1: combo LGBM\nstacker (805 cols)")
    box(7.0, 8.1, 3.6, 1.0, "P2: hierarchical\n(coarse→fine)")
    arrow(3.0, 7.6, 5, 7.1); arrow(7.0, 7.6, 5, 7.1)
    box(5, 6.6, 6.2, 0.85, "BLEND   0.842·P1 + 0.158·P2", fc="#e6f4ea", ec="#34a853")
    arrow(5, 6.18, 5, 5.75)
    box(5, 5.3, 6.2, 0.85, "Per-class ISOTONIC calibration (5-fold OOF)", fc="#e6f4ea", ec="#34a853")
    arrow(5, 4.88, 5, 4.45)
    box(5, 4.0, 7.4, 0.9, "TEST-PRIOR CORRECTION  × (test/train prior)$^{2.0}$  (Saerens, label-free)",
        fc="#fce8e6", ec="#ea4335")
    arrow(5, 3.55, 5, 3.12)
    box(5, 2.7, 7.4, 0.9, "ORIENTATION L2-injection  (w=0.15)", fc="#fce8e6", ec="#ea4335")
    arrow(5, 2.25, 5, 1.82)
    box(5, 1.4, 6.2, 0.85, "ROBUST per-class threshold → argmax", fc="#fef7e0", ec="#fbbc04")
    arrow(5, 0.97, 5, 0.6)
    ax.text(5, 0.35, "Public LB  0.8234", ha="center", va="center", fontsize=10, fontweight="bold", color="#ea4335")
    ax.set_title("Final pipeline (test-time stages in red)", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG / "pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote pipeline.png")

    # ---------------- Figure 3: per-class F1 (baseline vs final) ----------------
    classes = ["L0", "L1", "L2", "L3", "L4", "L5"]
    base_f1 = [0.964, 0.902, 0.173, 0.713, 0.896, 0.678]
    final_f1 = [0.967, 0.908, 0.384, 0.764, 0.924, 0.781]
    x = np.arange(len(classes)); w = 0.38
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.bar(x - w / 2, base_f1, w, label="271-feat baseline", color="#9aa0a6", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, final_f1, w, label="final pipeline", color="#34a853", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(classes); ax.set_ylabel("per-class F1"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1: baseline vs final pipeline", fontsize=9)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(axis="y", ls=":", alpha=0.4)
    for xi, b, f in zip(x, base_f1, final_f1):
        ax.text(xi + w / 2, f + 0.02, f"{f:.2f}", ha="center", fontsize=6)
    plt.tight_layout()
    fig.savefig(FIG / "per_class_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote per_class_f1.png")
    print("done:", [p.name for p in FIG.glob("*.png")])


if __name__ == "__main__":
    main()
