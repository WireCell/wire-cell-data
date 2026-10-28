# DUNE FD-HD 1x2x6: learned cascade deghosting, W9 single-detector fine-tune (TorchScript)

The five levels of the `e2-projq` cascade (charge-only conservation GNN, 64 hidden x 4 layers, `CascadeGNNLean` `_v2`
forward), **fine-tuned on the W9 response-matched truth labels**, seed 0 fold 0, exported at SDCC by
`wcfm/scripts/xdet/export.py` (wcp-porting-img `wcfm/docs/53_cross-detector-deghosting-plan.md` §10.2 part 38, the
"export/w9_ft/fdhd" set). Owner decision 2026-10-26 17:10 EDT (doc 53 §10.3): the single-detector fine-tuned model on
all four detectors. Loaded by the toolkit `img` component `CascadeDeghosting` through `TorchTensorSetService` (libtorch
2.8, CPU), unchanged C++ (15 blob + 4 wire base columns; no cross-detector columns). Integration, parity and the
imaging-level comparison: wcp-porting-img `pr_ml/docs/27_sdcc_deghoster_integration.md`; the imaging configs select
these files with the TLA `deghost_w9ft`. Detector: 12 APAs here (1x2x6), 4.67 / 4.67 / 4.79 mm pitch.

| file | level | cut width (wires) | super-wire k | threshold (logit) | role | best epoch | source checkpoint sha256 |
|---|---|---|---|---|---|---|---|
| `e2c_L0_U.ts` | 0 | uncut | 16 | -2.6525 | prune below | 13 | `aed56fccc67d459d` |
| `e2c_L1_32.ts` | 1 | 32 | 8 | -2.2468 | prune below | 11 | `f2113aa21a166fd2` |
| `e2c_L2_16.ts` | 2 | 16 | 4 | -2.1785 | prune below | 13 | `be9c15f65fcade10` |
| `e2c_L3_8.ts` | 3 | 8 | 2 | -2.4514 | prune below | 11 | `03921bbdcb266d06` |
| `e2c_L4_4.ts` | 4 | 4 | 1 | +1.0751 (exported: -0.8929) | keep at or above | 14 | `3877880d89ec63a2` |

`levels.json` is the same ladder in the `ml_levels` form `[level, width, k, threshold]` the imaging configs take; the
L4_4 entry is the keep logit (ghost-fraction-0.2 rule, `tables/keep_rule.md`; `tables/keep_threshold.*` records the ghost-fraction-0.2
search, which saturates on SBND / PDHD / PDVD and is not shipped there). The L0-L3 prune thresholds are the parent
models' (SDCC part 38: not re-derived on the fine-tuned logits).

- **Fine-tune:** `ft/fdhd_w8_w9 (parent baseline/fdhd_w8)`, 16 epochs from the parent's fold models on the W9 hybrid relabel of the parent's level graphs.
- **Held-out (SDCC, the FD-HD gate w9/fdhd/graphs_gate, 900 read-once events):** L4 AP single-detector fine-tuned / fine-tuned pooled / native A0 = 0.8918 / 0.8869 / 0.8745 (FD-HD gate, seed 0, 5 folds; A0 = the W8 baseline); charge recall /
  ghost fraction at the keep logit 0.9941 / 0.311 (gate).
- **Export:** torch 2.10.0+cu126; loads bitwise-identically under torch 2.8.0 and 2.5.1 (SDCC check) and loads in the
  toolkit's libtorch 2.8 (`check_load_tsmodel`, pr_ml doc 27).
- **Parity (pr_ml doc 27 §3):** C++ CascadeDeghosting against the same `.ts` in python on the C++ level graphs, relative
  |d logit| <= 1e-5 x max(|logit|, 1), 0 flips, 0 decision mismatches (`wcfm/scripts/d53_w9ft_parity.py`).

- **FD-HD: no head stage.** The keep logit is on the plain cascade's L4_4 output (doc 53 §3.5 contract, SDCC agent 2026-10-26), so the imaging config runs these with `ml_head=false`; the doc 32 head stage (`../deghost_head/`) belongs to the `../deghost/` models.

`SHA256SUMS` (from the export) covers the five `.ts`.

- **FD-HD running condition (2026-10-28, Xin decisions 3/4, wcp pr_ml doc 30 secs 6-8):** the `levels.json` L4_4 keep is
  **+1.0751** (was the exported -0.8929): the registered after-clustering operating point, at which the pooled S1 + W6 nu_e
  post-clustering ghost-point share equals the doc 32 production chain's; confirmed on R (2001-2550). These models without the
  head stage are the FD-HD imaging default (`wcfm/wct-img-all.jsonnet` `deghost_w9ft` true). The L0-L3 thresholds are unchanged.
