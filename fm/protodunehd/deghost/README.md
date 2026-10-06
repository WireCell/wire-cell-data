# ProtoDUNE-HD: coarse-to-fine learned deghosting models (TorchScript), round 2

The five levels of the `e2-projq` cascade (the charge-only conservation GNN of wcfm doc 09, the cascade of wcfm
doc 12) **retrained on ProtoDUNE-HD** (4 APAs, 2,560 channels each, 4.67 / 4.67 / 4.80 mm pitch, ±35.7° induction
wires), loaded by the toolkit `img` component `CascadeDeghosting` through the `pytorch` service
`TorchTensorSetService` (libtorch 2.8, CPU). Training, deployment and validation: `wcp-porting-img`
`pdhd/docs/31_ml-deghosting-training-proposal.md` section 12 (2026-10-06; round 2 is §12.9). The configs are
`pdhd/d31/{img,wct-img-all}.jsonnet`, the runner `pdhd/d31/run_img_evt.sh -M -m fm/protodunehd/deghost:` with
`-S "ml_levels=$(cat pdhd/d31/ml_levels.json)"`. **The production PDHD chain does not load these files**; the
cascade is an opt-in knob of the d31 fork.

The FD-HD models in `../../dune10kt-1x2x6/deghost/` share the architecture and the I/O contract (their README).
PDHD has the FD-HD wire geometry, and the FD-HD weights do work on it, but worse: average precision 0.922–0.949 on
the nodes where these models reach 0.976–0.982, and 2.4 % of the true charge lost against 0.3 %.

| file | level | cut width (wires) | super-wire k | threshold (logit) | role |
|---|---|---|---|---|---|
| `e2c_L0_U.ts` | 0 | uncut | 16 | -2.4070 | prune below |
| `e2c_L1_32.ts` | 1 | 32 | 8 | -1.9587 | prune below |
| `e2c_L2_16.ts` | 2 | 16 | 4 | -1.9075 | prune below |
| `e2c_L3_8.ts` | 3 | 8 | 2 | -1.9576 | prune below |
| `e2c_L4_4.ts` | 4 | 4 | 1 (real wires) | -2.0 | keep at or above |

`levels.json` holds the same ladder in the `ml_levels` form the d31 `img.jsonnet` takes.

- **Training sample:** 240 CORSIKA cosmic events: 40 of the DNN-ROI campaign's Stage A deposits
  (`/home/xqian/work/data/pdhd/generated`, events 0–39) and 200 generated for this study
  (`pdhd/work/d31/depos`, events 50–249), simulated by `pdhd/d31/wct-sim-depo-nf-sp-dnnroi.jsonnet` (DNN-ROI SP,
  electron lifetime 35 ms, `elecGain` 14), 960 anode-graphs. Events 40–49 are held out. Labels from `BlobDepoFill`
  (time offset 314 µs) on the 4-wire cells. **Cosmics only: no beam particles.**
- **Level graphs:** the toolkit's own `CascadeDeghosting` dumps (every level kept, `run_img_evt.sh -X`), so the
  training inputs are the deployed C++ graph builder's.
- **Models:** one per level, seed 0, fold 0 of five event folds (`/home/xqian/tmp/d31r2/train/<level>/models/
  e2-projq_s0_f0.pt`; sha256 in each `.meta.json`); held-out AP pooled over the five folds 0.976 / 0.982 / 0.982 / 0.979 / 0.977 (levels 0–4), fold 0 alone 0.977 / 0.982 / 0.982 / 0.978 / 0.975.
  Prune thresholds: 0.075 % of the dev true charge per level. Keep threshold −2.0, the conservative setting also
  used for PDVD; the ghost-fraction-0.2 rule of FD-HD gives −1.9536 here and the same gate numbers.
- **Export:** `pdhd/scripts/d31/export.py` = wcfm `d15_export.CascadeGNNLean` (the lower-memory forward of the
  FD-HD `_v2` files), `torch.jit.script` with torch 2.5.1; `set_keep.py` then writes the −2.0.
- **Parity:** C++ (libtorch, inside wire-cell) against the scripted model in python on the same dumped graphs,
  held-out event 40, 4 APAs, 317,192 nodes: max |Δ logit| 1.1e-5, 0 decision flips (`31_tables/r2_parity.md`).
- **Held-out gate** (events 40–49, doc 31 §12.9): completeness 0.9971 ± 0.0005 against 0.9930 for the production
  chain; ghost area removed 0.724 ± 0.032 against 0.319. On APA 1 the ghost area removed equals production's.
- **Charge unit:** `qhat` in units of 1e4 e as on FD-HD; `charge_scale` 0.25 and `repair_q_floor` 1e4 e, the FD-HD
  values, in the d31 config.
- **Frame time:** the simulated SP frames of the training sample were rewritten to frame time 0 (PDHD simulation
  writes −250 µs). Data SP frames have frame time 0. For frames with a non-zero time use the toolkit knob
  `slice_start_relative` of `CascadeDeghosting` (default false).
- **On data** (run 29107, 30 events, doc 31 §12.9): kept area and solved charge match simulation APA by APA. On
  APA 0 the solved charge is 0.79 of the production chain's (0.81 in simulation, where the true charge kept is
  99.4 % and production's solved charge is 1.67 times the truth). Not validated after clustering.
- **Round 1** (40 events, AP 0.953–0.971, ghost area removed 0.589) is superseded and not in this repository.

Input / output contract: identical to `../../dune10kt-1x2x6/deghost/README.md`
(`forward(xb, wq, wplane, bw_src, bw_dst, bw_w, bb, bb_in, ww) -> (logit, qhat)`; `bw_w` read on levels 0–3).
