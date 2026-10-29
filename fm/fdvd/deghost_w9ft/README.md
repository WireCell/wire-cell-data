# DUNE FD-VD 1x8x14: learned cascade deghosting, FD-VD fine-tune (TorchScript)

The five levels of the `e2-projq` cascade (charge-only conservation GNN, 64 hidden x 4 layers, `CascadeGNNLean` `_v2`
forward, the same architecture and C++ as `../../protodunevd/deghost_w9ft` and `../../dune10kt-1x2x6/deghost_w9ft`),
**fine-tuned on FD-VD truth labels** on wcgpu1, initialised from the shipped FD-HD W9 fold checkpoints, seed 0, fold 0
exported by `wcfm/scripts/xdet/export.py`. Owner decision 2026-10-29 (Xin: "We can adopt m560 now"): this is the FD-VD
deghoster of the ML-PR campaign lane, both in imaging (wcp `pr_ml/fdvd/img/wct-img-all.jsonnet`, `deghost_w9ft` true by
default there) and in the PR job's gap-fill retile (`pr_ml/recipes/mlpr_v1c_fdvd.json`). Write-up:
wcp-porting-validation `pr_ml/docs/34_fdvd_in_the_campaign.md` sec 8. Loaded by the toolkit `img` component
`CascadeDeghosting` through `TorchTensorSetService` (libtorch 2.8, CPU), unchanged C++ (15 blob + 4 wire base columns;
no cross-detector columns, `xdet_mip` false). Detector: 112 single-face top-drift CRMs, wires
`../../dunevd10kt_3view_30deg_v7_refactored_1x8x14.json.bz2`.

| file | level | cut width (wires) | super-wire k | threshold (logit) | role | best epoch | source checkpoint sha256 |
|---|---|---|---|---|---|---|---|
| `e2c_L0_U.ts` | 0 | uncut | 16 | -1.8546 | prune below | 9 | `4cf4d645d56f3bd8` |
| `e2c_L1_32.ts` | 1 | 32 | 8 | -1.2329 | prune below | 9 | `23da6146b5f09cc0` |
| `e2c_L2_16.ts` | 2 | 16 | 4 | -1.1023 | prune below | 9 | `1c0cbd7f50acbe9a` |
| `e2c_L3_8.ts` | 3 | 8 | 2 | -1.1429 | prune below | 9 | `155a9c82295bc5f1` |
| `e2c_L4_4.ts` | 4 | 4 | 1 | **+1.0** (exported: -0.3599) | keep at or above | 13 | `d8f541ca6da13306` |

`levels.json` is the same ladder in the `ml_levels` form `[level, width, k, threshold]`; its L4_4 entry is the decided
keep **+1.0** (Xin 2026-10-29, chosen from the after-clustering keep scan below). The `.meta.json` files keep the
exported thresholds; `tables/keep_threshold.*` records the ghost-fraction-0.2 rule that gave -0.3599. The L0-L3
prune thresholds are this fine-tune's own (the cascade builder's coverage-guard rule on its fold models).

- **Training set ("m560"):** 700 FD-VD events simulated by the campaign lane's `fdvd)` arm (Stage A GENIE / MPV-MPR /
  single-particle, Stage B fdvd_sim, imaging with `-X` truth tiers, W9 hybrid relabel): numu 260, nue 260, nc 60,
  mpvmpr 60, single 60; 560 train (5-fold CV split by event) / 140 held out (20 % per kind by sha256 of the tag).
  Manifest `manifest_fdvd_m.json` (event list in each `.meta.json`).
- **Held-out (140 events, 5-fold mean):** L4 AP 0.915 (nue 0.870, numu 0.962); real-cell charge recall at ghost
  fraction 3 / 5 / 10 % = 0.937 / 0.953 / 0.971. The same recipe on 80 events from the PDVD export gave 0.894.
- **After clustering (10 smoke events, none in training; `tables/after_clustering_keep_scan.tsv`):**
  real charge kept / ghost-point share = 0.964 / 4.6 % at +1.0, 0.978 / 7.5 % at +0.5, 0.995 / 16.3 % at -0.36;
  traditional imaging 27.2 % ghosts; the PDVD export zero-shot at its +1.7122 kept 0.845 at 3.9 %.
- **Export:** torch 2.10.0+cu128. The same `.ts` files (sha256 in `SHA256SUMS`) ran in the toolkit's libtorch 2.8
  through the FD-VD imaging job on the smoke events (wcp pin15, toolkit 8074ae11) for the table above.
- **Input scale:** the input standardiser is FD-HD's (`e2_train.py` keeps the init model's); a MIP-rescaled
  variant (FD-VD Q0 3958 e vs FD-HD 2687 e) gave no measurable gain (doc 34 sec 8) and is not used.

`SHA256SUMS` (from the export) covers the five `.ts`.
