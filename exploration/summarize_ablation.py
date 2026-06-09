"""Read all ablation sidecars and write reports/ablation_features.md.

Feeds Report Q4 directly: each row is "all features minus group X" with the
delta vs. the full-catalog baseline (lgbm_full_v1).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "oof"
OUT = ROOT / "reports" / "ablation_features.md"

# Order rows by expected importance (rough prior).
GROUPS_ORDERED = [
    "basic_stats", "subwindow", "magnitude", "fft", "gravity",
    "jerk", "autocorr", "per_file_norm", "crossaxis", "zerocross", "quality",
]


def load_meta(name: str):
    p = OOF / f"lgbm_full_{name}_meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    base = load_meta("v1")
    if base is None:
        raise SystemExit("Missing oof/lgbm_full_v1_meta.json — run baseline first.")

    base_f1 = base["cv_f1_mean"]
    base_per_class = base["per_class_f1"]

    rows: list[dict] = []
    for g in GROUPS_ORDERED:
        meta = load_meta(f"abl_no_{g}")
        if meta is None:
            continue
        d = meta["cv_f1_mean"] - base_f1
        rows.append({
            "group": g,
            "n_features": meta["n_features"],
            "cv_f1": meta["cv_f1_mean"],
            "delta": d,
            "per_class_f1": meta["per_class_f1"],
        })

    # Sort by delta so the most-important groups are at the top
    rows.sort(key=lambda r: r["delta"])

    lines: list[str] = []
    lines.append("# Phase-4 Feature-Group Ablation\n")
    lines.append(
        "Each row removes one feature group from the full catalog and re-runs the "
        "5-fold GroupKFold CV. Negative ΔCV F1 means the group was helping; the "
        "more negative, the more important.\n"
    )
    lines.append(f"**Baseline (full catalog, `lgbm_full_v1`):** {base['n_features']} features, CV F1-macro = **{base_f1:.4f}**.\n")
    lines.append(
        "Per-class F1 baseline: "
        + ", ".join(f"`L{i}={f:.3f}`" for i, f in enumerate(base_per_class))
        + "\n"
    )
    lines.append("")

    lines.append("| Removed group | n features | CV F1 | Δ vs full | L0 | L1 | L2 | L3 | L4 | L5 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        pcs = r["per_class_f1"]
        lines.append(
            f"| `{r['group']}` | {r['n_features']} | "
            f"{r['cv_f1']:.4f} | {r['delta']:+.4f} | "
            + " | ".join(f"{f:.3f}" for f in pcs) + " |"
        )

    if not rows:
        lines.append("| _no ablation runs found_ | | | | | | | | | |")

    lines.append("")
    lines.append("## Reading the table")
    lines.append("")
    lines.append("- **ΔCV F1 < -0.005** means the group is meaningfully important.")
    lines.append("- **Δ ≈ 0** means the group's signal is redundant with other groups.")
    lines.append("- **Per-class deltas matter for the report:** if removing FFT tanks label-2 F1 by 0.10 but only moves macro by 0.01, that's a story worth telling — FFT is critical *for the bottleneck class*.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}  ({len(rows)} ablation rows)")


if __name__ == "__main__":
    main()
