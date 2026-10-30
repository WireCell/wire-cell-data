# DUNE FD-VD 1x8x14: learned cascade deghosting, FD-VD fine-tune on 1,500 events ("x1500", TorchScript)

The five levels of the `e2-projq` cascade (charge-only conservation GNN, 64 hidden x 4 layers, `CascadeGNNLean` `_v2`
forward, the same architecture and C++ as `../deghost_w9ft` (m560), `../../protodunevd/deghost_w9ft` and
`../../dune10kt-1x2x6/deghost_w9ft`), **fine-tuned on FD-VD truth labels** on wcgpu1, initialised from the shipped
FD-HD W9 fold checkpoints, seed 0, fold 0 exported by `wcfm/scripts/xdet/export.py`. It supersedes `../deghost_w9ft`
(m560, left unchanged) as the FD-VD deghoster of the ML-PR campaign lane, in imaging (wcp
`pr_ml/fdvd/img/wct-img-all.jsonnet`) and in the PR job's gap-fill retile (`pr_ml/fdvd/wct-pr-perevt-fdvd.jsonnet`), through
the model-dir setting only. Owner decision 2026-10-30 (Xin): adopt at the iso-ghost keep **+1.35**. Write-up:
wcp-porting-validation `pr_ml/docs/34_fdvd_in_the_campaign.md` sec 8.12. Loaded by the toolkit `img` component
`CascadeDeghosting` through `TorchTensorSetService` (libtorch 2.8, CPU), unchanged C++ (15 blob + 4 wire base columns;
no cross-detector columns, `xdet_mip` false). Detector: 112 single-face top-drift CRMs, wires
`../../dunevd10kt_3view_30deg_v7_refactored_1x8x14.json.bz2`.

**Trained and evaluated on graphs built with the RayGrid `Blob::add` exact_center fix** (toolkit f6331be0, knobs
`GridTiling.exact_center` / `CascadeDeghosting.cut_exact_center`, set true in the FD-VD lane by wcp a4d5a77c,
WireCell/wire-cell-toolkit#542). Use it with exact_center on.

| file | level | cut width (wires) | super-wire k | threshold (logit) | role | best epoch | source checkpoint sha256 |
|---|---|---|---|---|---|---|---|
| `e2c_L0_U.ts` | 0 | uncut | 16 | -1.7321 | prune below | 13 | `c1222e1d494f67c3` |
| `e2c_L1_32.ts` | 1 | 32 | 8 | -1.3594 | prune below | 13 | `86b0d3fc806ced24` |
| `e2c_L2_16.ts` | 2 | 16 | 4 | -1.0245 | prune below | 15 | `317bff49228ac5b2` |
| `e2c_L3_8.ts` | 3 | 8 | 2 | -1.1973 | prune below | 15 | `90da9d700a09c99f` |
| `e2c_L4_4.ts` | 4 | 4 | 1 | **+1.35** (exported: -0.2246) | keep at or above | 13 | `53b9200d9a98054d` |

`levels.json` is the same ladder in the `ml_levels` form `[level, width, k, threshold]`; its L4_4 entry is the decided
keep **+1.35**. The `.meta.json` files keep the exported thresholds; `tables/keep_threshold.*` records the
ghost-fraction-0.2 rule that gave -0.2246. The L0-L3 prune thresholds are this fine-tune's own (the cascade builder's
coverage-guard rule on its fold models). A fixed keep logit is not a fixed operating point across retrains: x1500 at
m560's +1.0 keeps more charge **and** more ghosts, so the keep was set where x1500's ghost share matches m560's.

- **Training set ("x1500"):** 1,500 FD-VD events simulated by the campaign lane's `fdvd)` arm (Stage A GENIE / MPV-MPR /
  single-particle, Stage B fdvd_sim, imaging with `-X` truth tiers and exact_center on, W9 hybrid relabel): m560's 700
  events re-imaged plus 800 more numu / nue; 1,200 train (5-fold CV split by event) / 300 held out (numu 368/92, nue
  688/172, nc / mpvmpr / single 48/12 each). m560's 140 held-out events stay held out. Manifest `manifest_fdvd_m.json`
  (event list in each `.meta.json`; batch-5-8 nue events with J >= 100 are numbered N + 10000).
- **Held-out cell level (m560's 140 events, same exact_center graphs, 5-fold mean):** L4 AP 0.9204 vs m560 0.9155
  (nue 0.8775 vs 0.8707, numu 0.9645 vs 0.9617); better at every level L0-L4 and in every kind.
- **After clustering (the same 140 held-out events, exact_center on; `tables/after_clustering_held140_*.tsv`):**
  real charge kept / ghost-point share, pooled, [16, 84] % per-event bootstrap:

  | arm | charge kept | ghost share | nue charge kept | nue ghost share |
  |---|---|---|---|---|
  | traditional | - | 0.260 | - | 0.262 |
  | m560 at +1.0 (`../deghost_w9ft`) | 0.9624 [0.9551, 0.9701] | 0.0497 | 0.9457 | 0.0684 |
  | **x1500 at +1.35 (this ladder)** | **0.9709** [0.9653, 0.9773] | **0.0482** | **0.9573** | **0.0665** |
  | x1500 at +1.5 | 0.9669 | 0.0418 | 0.9518 | 0.0581 |
  | x1500 at +1.0 | 0.9793 | 0.0666 | 0.9689 | 0.0901 |
  | x1500 at +0.5 | 0.9893 | 0.1030 | 0.9829 | 0.1354 |
  | x1500 at -0.2246 | 1.0002 | 0.1757 | 0.9986 | 0.2210 |

  Paired per-event bootstrap, x1500 at +1.35 minus m560 at +1.0: charge kept +0.0085 (95 % [+0.0059, +0.0113]),
  ghost share -0.0015 (95 % [-0.0040, +0.0007]), nue charge kept +0.0117 (95 % [+0.0081, +0.0156]). Worst event
  `nue_b16_e3` 0.815 -> 0.850.
- **Export:** torch 2.10.0+cu128. The same `.ts` files (sha256 in `SHA256SUMS`) ran in the toolkit's libtorch 2.8
  through the FD-VD imaging job (pin with toolkit f6331be0, wcp a4d5a77c) for the tables above.
- **Input scale:** the input standardiser is FD-HD's (`e2_train.py` keeps the init model's), as m560.

`SHA256SUMS` (from the export) covers the five `.ts`.
