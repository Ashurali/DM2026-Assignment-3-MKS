"""Average multiple DG-CISC seed OOF/test-prob arrays into one combined source.

Usage:  python scripts/avg_dg_seeds.py <in_name1> <in_name2> ... <out_name>
Example: python scripts/avg_dg_seeds.py v2s42 v2s7 v2s23 v2ms
  -> writes oof/dg_cisc_v2ms_oof.npy + _test_probs.npy (mean of the inputs).
Multi-seed averaging reduces OOF variance -> cleaner calibrated L2 probs for the
gated injection.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
*names, out = sys.argv[1:]
if not names:
    raise SystemExit("need >=1 input name + 1 output name")

oofs = [np.load(ROOT / "oof" / f"dg_cisc_{n}_oof.npy") for n in names]
tests = [np.load(ROOT / "oof" / f"dg_cisc_{n}_test_probs.npy") for n in names]
oof = np.mean(oofs, axis=0).astype(np.float32)
test = np.mean(tests, axis=0).astype(np.float32)
np.save(ROOT / "oof" / f"dg_cisc_{out}_oof.npy", oof)
np.save(ROOT / "oof" / f"dg_cisc_{out}_test_probs.npy", test)
print(f"averaged {names} -> dg_cisc_{out}: oof{oof.shape} test{test.shape}")
