# ProtoDUNE-HD: learned cascade deghosting, W9 single-detector fine-tune (TorchScript)

The five levels of the `e2-projq` cascade (charge-only conservation GNN, 64 hidden x 4 layers, `CascadeGNNLean` `_v2`
forward), **fine-tuned on the W9 response-matched truth labels**, seed 0 fold 0, exported at SDCC by
`wcfm/scripts/xdet/export.py` (wcp-porting-img `wcfm/docs/53_cross-detector-deghosting-plan.md` §10.2 part 38, the
"export/w9_ft/pdhd" set). Owner decision 2026-10-26 17:10 EDT (doc 53 §10.3): the single-detector fine-tuned model on
all four detectors. Loaded by the toolkit `img` component `CascadeDeghosting` through `TorchTensorSetService` (libtorch
2.8, CPU), unchanged C++ (15 blob + 4 wire base columns; no cross-detector columns). Integration, parity and the
imaging-level comparison: wcp-porting-img `pr_ml/docs/27_sdcc_deghoster_integration.md`; the imaging configs select
these files with the TLA `deghost_w9ft`. Detector: 4 APAs, 4.67 / 4.67 / 4.80 mm pitch, +-35.7 deg induction.

| file | level | cut width (wires) | super-wire k | threshold (logit) | role | best epoch | source checkpoint sha256 |
|---|---|---|---|---|---|---|---|
| `e2c_L0_U.ts` | 0 | uncut | 16 | -2.4070 | prune below | 14 | `ccb70752df0b6fed` |
| `e2c_L1_32.ts` | 1 | 32 | 8 | -1.9587 | prune below | 14 | `026a119ff51bae96` |
| `e2c_L2_16.ts` | 2 | 16 | 4 | -1.9075 | prune below | 14 | `bc82b34e2cb9aa12` |
| `e2c_L3_8.ts` | 3 | 8 | 2 | -1.9576 | prune below | 14 | `a7337f9755f7d9ca` |
| `e2c_L4_4.ts` | 4 | 4 | 1 | +1.8003 (exported: -2.3495) | keep at or above | 11 | `d8ae7da3d022e046` |

`levels.json` is the same ladder in the `ml_levels` form `[level, width, k, threshold]` the imaging configs take; the
L4_4 entry is the keep logit (plateau rule, `tables/keep_rule.md`; `tables/keep_threshold.*` records the ghost-fraction-0.2
search, which saturates on SBND / PDHD / PDVD and is not shipped there). The L0-L3 prune thresholds are the parent
models' (SDCC part 38: not re-derived on the fine-tuned logits).

- **Fine-tune:** `ft/pdhd_r2_w9 (parent pdhd/graphs_r2/train, the round-2 models of ../deghost)`, 16 epochs from the parent's fold models on the W9 hybrid relabel of the parent's level graphs.
- **Held-out (SDCC, W9 held-out tree w9/pdhd/graphs_r2/graphs, events 40-49):** L4 AP single-detector fine-tuned / fine-tuned pooled / native A0 = 0.9969 / 0.9977 / 0.9941; charge recall /
  ghost fraction at the keep logit 0.9997 / 0.112.
- **Export:** torch 2.10.0+cu126; loads bitwise-identically under torch 2.8.0 and 2.5.1 (SDCC check) and loads in the
  toolkit's libtorch 2.8 (`check_load_tsmodel`, pr_ml doc 27).
- **Parity (pr_ml doc 27 §3):** C++ CascadeDeghosting against the same `.ts` in python on the C++ level graphs, relative
  |d logit| <= 1e-5 x max(|logit|, 1), 0 flips, 0 decision mismatches (`wcfm/scripts/d53_w9ft_parity.py`).

`SHA256SUMS` (from the export) covers the five `.ts`.

- **Decided keep (2026-10-28):** the `levels.json` L4_4 keep is **+1.8003** (was the exported -2.3495), the keep logit Xin chose
  for PDHD on 2026-10-27 (wcfm doc 53 §10.3 decision 2, option C; the pr_ml doc 27 §8 operating point) and the value the
  ML-PR recipes `pr_ml/recipes/mlpr_v1*_pdhd.json` pass. `-S deghost_w9ft=true` without a keep
  now runs at it; an explicit `deghost_w9ft_keep` or `ml_levels` still wins. Compiled-config gate (wcp pr_ml doc 27 §8.9):
  knob on == the old knob on + the decided keep; recipe, knob-off and explicit-keep configs byte-identical. This is the PDHD production imaging stays traditional (`pdhd/d31` `deghost_w9ft` false); these models serve the knob and the ML-PR branch.
  The L0-L3 thresholds and the models are unchanged.
