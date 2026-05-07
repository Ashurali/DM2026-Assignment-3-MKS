"""Print a one-page summary of every lgbm_full_*_meta.json sidecar.

Useful at the end of a Phase-3 batch to see which run won on which class.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "oof"


def main() -> None:
    metas = sorted(OOF.glob("lgbm_full_*_meta.json"))
    if not metas:
        print("No sidecars found.")
        return

    print(f"\n{'Run':30s}  {'CV F1':7s}  {'L0':6s}  {'L1':6s}  {'L2':6s}  {'L3':6s}  {'L4':6s}  {'L5':6s}  Notes")
    print("-" * 110)
    for p in metas:
        d = json.loads(p.read_text())
        name = d.get("model", p.stem)
        f1 = d.get("cv_f1_mean", float("nan"))
        per = d.get("per_class_f1", [float("nan")] * 6) + [float("nan")] * 6
        notes = []
        if d.get("smote"):
            notes.append("SMOTE")
        if d.get("tuned"):
            notes.append("tuned")
        if d.get("exclude_groups"):
            notes.append(f"-{','.join(d['exclude_groups'])}")
        notes.append(f"n_feat={d.get('n_features')}")
        print(
            f"{name:30s}  {f1:.4f}   "
            + "  ".join(f"{x:.3f}" for x in per[:6])
            + f"   {' '.join(notes)}"
        )

    # Highlight the best so far
    best = max(metas, key=lambda p: json.loads(p.read_text()).get("cv_f1_mean", -1))
    best_d = json.loads(best.read_text())
    print(f"\nBest by CV F1-macro: {best_d.get('model')} = {best_d.get('cv_f1_mean'):.4f}")


if __name__ == "__main__":
    main()
