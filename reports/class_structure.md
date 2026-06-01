# Class structure investigation

Embedding source: `cnn_bilstm_v1_emb_train.npy` (256-d penultimate-layer features).


## Cross-class proximity (smoking gun for entangled boundaries)

| Class | nearest-other | % closer to other than self | median d_self | median d_other |
|---|---|---|---|---|
| L2 | L1 | 76.5% | 1.503 | 1.243 |
| L2 | L3 | 31.8% | 1.503 | 1.715 |
| L3 | L1 | 24.2% | 1.442 | 1.819 |
| L3 | L2 | 18.3% | 1.442 | 2.110 |
| L5 | L1 | 30.4% | 1.098 | 1.667 |
| L5 | L3 | 14.3% | 1.098 | 2.480 |

## L2

- n = 358
- Best K by BIC: **1**, sizes [358]
- BIC by K: K=1:-79775, K=2:57538, K=3:223963, K=4:402773

  **VERDICT:** L2 prefers K=1 → it's a single coherent cluster. The hardness comes from overlap with another class (see proximity table), not from internal multimodality. Decomposition wouldn't help; performance ceiling is near current.


## L3

- n = 656
- Best K by BIC: **1**, sizes [656]
- BIC by K: K=1:-247770, K=2:-145815, K=3:23667, K=4:198305

  **VERDICT:** L3 prefers K=1 → it's a single coherent cluster. The hardness comes from overlap with another class (see proximity table), not from internal multimodality. Decomposition wouldn't help; performance ceiling is near current.


## L5

- n = 526
- Best K by BIC: **1**, sizes [526]
- BIC by K: K=1:-219065, K=2:-75047, K=3:88318, K=4:271368

  **VERDICT:** L5 prefers K=1 → it's a single coherent cluster. The hardness comes from overlap with another class (see proximity table), not from internal multimodality. Decomposition wouldn't help; performance ceiling is near current.

