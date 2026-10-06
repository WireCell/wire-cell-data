# ProtoDUNE-VD: coarse-to-fine learned deghosting models (TorchScript), round 2

The five levels of the `e2-projq` cascade (the charge-only conservation GNN of wcfm doc 09, the cascade of wcfm
doc 12) **retrained on ProtoDUNE-VD** (8 CRP half-anodes, 60° strips, 7.65 / 7.65 / 5.10 mm pitch), loaded by the
toolkit `img` component `CascadeDeghosting` through the `pytorch` service `TorchTensorSetService` (libtorch 2.8,
CPU).  Training, deployment and validation: `wcp-porting-img` `pdvd/docs/nf_sp_img_clus/121_ml-deghosting-training-proposal.md`
section 12 and `122_ml-deghosting-fdhd-vs-pdvd-and-data.md` section 8 (round 2, 2026-10-06; **round 1 of 10-05 was trained
without the wire charge, a frame-time defect in the training dumps, and is replaced by these files**); the configs are `pdvd/d121/{img,wct-img-all}.jsonnet`, the runner
`pdvd/d121/run_img_evt.sh -M`.  The FD-HD models in `../../dune10kt-1x2x6/deghost/` share the architecture and the
I/O contract (their README) and nothing else: on PDVD they score at AP 0.59–0.62 (random 0.50).

| file | level | cut width (strips) | super-wire k | threshold (logit) | role |
|---|---|---|---|---|---|
| `e2c_L0_U.ts` | 0 | uncut | 16 | -1.8312 | prune below |
| `e2c_L1_32.ts` | 1 | 32 | 8 | -1.0290 | prune below |
| `e2c_L2_16.ts` | 2 | 16 | 4 | -0.7475 | prune below |
| `e2c_L3_8.ts` | 3 | 8 | 2 | -1.1095 | prune below |
| `e2c_L4_4.ts` | 4 | 4 | 1 (real strips) | -2.0 | keep at or above |

`levels.json` holds the same ladder in the `ml_levels` form the d121 `img.jsonnet` takes.

- **Training sample:** 200 CORSIKA cosmic events of the DNN-ROI campaign's Stage A deposits (`/home/xqian/work/data/
  pdvd/generated`, q > 0 rows dropped), simulated by `pdvd/d121/wct-sim-depo-nf-sp-dnnroi.jsonnet` (data transport
  constants, DNN-ROI + L1SP, wire file `protodunevd-wires-larsoft-v7-uvwfit`), 1,600 half-anode graphs, both drift
  volumes; labels from `BlobDepoFill` (time offset 372 µs) on the 4-strip cells.  **No beam particles, no rotated or
  gun events yet** (doc 121 §12.10): the keep threshold is set on the pooled held-out
  cosmic cells, not on 0° muon slabs.
- **Level graphs:** the toolkit's own `CascadeDeghosting` dumps (every level kept, `pdvd/d121/run_img_evt.sh -X`),
  so the training inputs are the deployed C++ graph builder's.
- **Models:** one per level, seed 0, fold 0 of five event folds (`/home/xqian/tmp/d121r2/train/<level>/models/
  e2-projq_s0_f0.pt`; sha256 in each `.meta.json`); held-out AP 0.981 / 0.983 / 0.987 / 0.990 / 0.986 (levels 0–4).
  Prune thresholds: 0.075 % of the dev true charge per level; keep threshold -2.0: the conservative point of
  doc 122 §9.2 (99.83 % charge recall on the held-out cells, 0.01 % below keeping every arriving cell, ghost fraction
  0.127). The ghost-fraction-0.2 rule of FD-HD gives -7.3968 here (a flat part of the curve); -0.830 = ghost fraction
  0.1 keeps 99.48 %.
- **Export:** `pdvd/scripts/d121/export.py` = wcfm `d15_export.CascadeGNNLean` (the lower-memory forward of the
  FD-HD `_v2` files), `torch.jit.script` with torch 2.5.1.
- **Parity:** C++ (libtorch, inside wire-cell) vs the scripted model in python on the same dumped graphs, held-out
  event 200, 8 half-anodes, 182,868 nodes: max |Δ logit| 1.1e-5, 0 decision flips (`122_tables/r2_parity.md`).
- **Charge unit:** `qhat` in units of 1e4 e as on FD-HD (the standardiser was refit on PDVD); the C++
  `repair_q_floor` is set to 1.6e4 e in the d121 config.

Input / output contract: identical to `../../dune10kt-1x2x6/deghost/README.md`
(`forward(xb, wq, wplane, bw_src, bw_dst, bw_w, bb, bb_in, ww) -> (logit, qhat)`; `bw_w` read on levels 0–3).

- **Frame time:** the simulated SP frames of the training sample carry frame time 0. `CascadeDeghosting` reads its
  charge at (slice start − frame time) / tick while `MaskSlice` slice starts are frame-relative, so a frame with a
  non-zero time is read at the wrong ticks (doc 122 §3). Data SP frames have frame time 0. Toolkit knob `slice_start_relative` (default false) of
  `CascadeDeghosting` / `CascadeDeghostingFM` reads such frames correctly; d121 `ml_slice_start_relative`.
- **On data** (doc 122 §8.4): quiet half-anodes match simulation; on the busiest half-anodes zero-charge cells remain
  in isochronous slices and the solved charge is 7–22 % below the production chain's on four of them. Not validated
  after clustering.
