# SBND: learned cascade deghosting, W9 single-detector fine-tune (TorchScript)

The five levels of the `e2-projq` cascade (charge-only conservation GNN, 64 hidden x 4 layers, `CascadeGNNLean` `_v2`
forward), **fine-tuned on the W9 response-matched truth labels**, seed 0 fold 0, exported at SDCC by
`wcfm/scripts/xdet/export.py` (wcp-porting-img `wcfm/docs/53_cross-detector-deghosting-plan.md` §10.2 part 38, the
"export/w9_ft/sbnd" set). Owner decision 2026-10-26 17:10 EDT (doc 53 §10.3): the single-detector fine-tuned model on
all four detectors. Loaded by the toolkit `img` component `CascadeDeghosting` through `TorchTensorSetService` (libtorch
2.8, CPU), unchanged C++ (15 blob + 4 wire base columns; no cross-detector columns). Integration, parity and the
imaging-level comparison: wcp-porting-img `pr_ml/docs/27_sdcc_deghoster_integration.md`; the imaging configs select
these files with the TLA `deghost_w9ft`. Detector: 2 anodes (APAs), 3.0 mm pitch, +-60 deg induction.

| file | level | cut width (wires) | super-wire k | threshold (logit) | role | best epoch | source checkpoint sha256 |
|---|---|---|---|---|---|---|---|
| `e2c_L0_U.ts` | 0 | uncut | 16 | -2.2452 | prune below | 11 | `7eaeebfe34ef7baf` |
| `e2c_L1_32.ts` | 1 | 32 | 8 | -2.2407 | prune below | 15 | `af12725f7ee07cf7` |
| `e2c_L2_16.ts` | 2 | 16 | 4 | -1.7655 | prune below | 15 | `3f4c15c0de153c1e` |
| `e2c_L3_8.ts` | 3 | 8 | 2 | -1.9008 | prune below | 14 | `2b353e33257f8bcd` |
| `e2c_L4_4.ts` | 4 | 4 | 1 | -0.6936 (exported: -3.1877) | keep at or above | 10 | `fa666961f541f902` |

`levels.json` is the same ladder in the `ml_levels` form `[level, width, k, threshold]` the imaging configs take; the
L4_4 entry is the keep logit (plateau rule, `tables/keep_rule.md`; `tables/keep_threshold.*` records the ghost-fraction-0.2
search, which saturates on SBND / PDHD / PDVD and is not shipped there). The L0-L3 prune thresholds are the parent
models' (SDCC part 38: not re-derived on the fine-tuned logits).

- **Fine-tune:** `ft/sbnd_w9 (parent baseline/sbnd)`, 16 epochs from the parent's fold models on the W9 hybrid relabel of the parent's level graphs.
- **Held-out (SDCC, W9 held-out trees w9/sbnd/graphs_w4, events 526-679 and 1263-1415):** L4 AP single-detector fine-tuned / fine-tuned pooled / native A0 = 0.9941 / 0.9931 / 0.9903; charge recall /
  ghost fraction at the keep logit 0.9985 / 0.152.
- **Export:** torch 2.10.0+cu126; loads bitwise-identically under torch 2.8.0 and 2.5.1 (SDCC check) and loads in the
  toolkit's libtorch 2.8 (`check_load_tsmodel`, pr_ml doc 27).
- **Parity (pr_ml doc 27 §3):** C++ CascadeDeghosting against the same `.ts` in python on the C++ level graphs, relative
  |d logit| <= 1e-5 x max(|logit|, 1), 0 flips, 0 decision mismatches (`wcfm/scripts/d53_w9ft_parity.py`).

`SHA256SUMS` (from the export) covers the five `.ts`.

- **Decided keep (2026-10-28):** the `levels.json` L4_4 keep is **-0.6936** (was the exported -3.1877), the keep logit Xin chose
  for SBND on 2026-10-27 (wcfm doc 53 §10.3 decision 2, option C; the pr_ml doc 27 §8 operating point) and the value the
  ML-PR recipes `pr_ml/recipes/mlpr_v1*_sbnd.json` pass. `-S deghost_w9ft=true` without a keep
  now runs at it; an explicit `deghost_w9ft_keep` or `ml_levels` still wins. Compiled-config gate (wcp pr_ml doc 27 §8.9):
  knob on == the old knob on + the decided keep; recipe, knob-off and explicit-keep configs byte-identical. This is not the SBND production chain: SBND production keeps the traditional deghosting (owner 2026-10-27); these models serve the R&D imaging fork `sbnd/sbnd_xin/d53` and the ML-PR branch.
  The L0-L3 thresholds and the models are unchanged.
