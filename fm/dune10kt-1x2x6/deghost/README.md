# DUNE FD-HD 1x2x6: coarse-to-fine learned deghosting models (TorchScript)

The five levels of the `e2-projq` cascade (charge-only conservation GNN, wcfm doc 09; cascade and training
wcfm doc 12), loaded by the toolkit `img` component `CascadeDeghosting` through the `pytorch` service
`TorchTensorSetService` (libtorch 2.8, CPU).  Deployment and validation: `wcp-porting-img` `wcfm/docs/14`.

| file | level | cut width (wires) | super-wire k | threshold (logit) | role |
|---|---|---|---|---|---|
| `e2c_L0_U.ts` | 0 | uncut | 16 | -3.1384 | prune below |
| `e2c_L1_32.ts` | 1 | 32 | 8 | -1.9868 | prune below |
| `e2c_L2_16.ts` | 2 | 16 | 4 | -2.1615 | prune below |
| `e2c_L3_8.ts` | 3 | 8 | 2 | -2.2894 | prune below |
| `e2c_L4_4.ts` | 4 | 4 | 1 (real wires) | -0.1877 | keep at or above |

- **Training:** doc 12 T2, one model per level, seed 0, fold 0 of `train_final_projq`'s five interaction-group
  folds (`/home/xqian/tmp/wcfm-d12/train/<level>/models/e2-projq_s0_f0.pt`; sha256 in each `.meta.json`).
  The prune thresholds are doc 12's (0.075 % of the dev true charge per level); the keep threshold is doc 14's
  (ghost fraction 0.2 on the pooled dev 0-degree muon-slab cells, L4 held-out logits).
- **Export:** `wcfm/scripts/d14_export.py export` with torch 2.5.1 (the toolkit direnv python; <= 2.8 for the
  libtorch shim), `torch.jit.script` of an inference-only wrapper that loads the state dict unchanged and holds
  the standardiser, `pi`, the graph preprocessing and the lean chunked forward.
- **Parity:** scripted vs the trainer's lean forward, same torch, 320 fold-0 test graphs per level, 5.7 M
  cells: max |d logit| 5.3e-6, 0 decision flips at the thresholds (`wcfm/docs/14_tables/export_parity.md`).

## Input / output contract

`forward(xb, wq, wplane, bw_src, bw_dst, bw_w, bb, bb_in, ww) -> (logit, qhat)`:

| arg | dtype, shape | meaning |
|---|---|---|
| `xb` | f4 [N, 15] | per blob: for U, V, W (nch, nch with charge > 0, log1p sum q, log1p mean q), then the three log-sum differences |
| `wq` | f4 [M] | per (super-)wire node: the charge, 0.25 x the 4-tick sum of the `gauss` trace (summed over the bin) |
| `wplane` | i8 [M] | plane 0, 1, 2 |
| `bw_src`, `bw_dst`, `bw_w` | i8, i8, f4 [E] | blob -> wire edges; `bw_w` = the blob's channels in the bin (read on levels 0-3 only) |
| `bb`, `bb_in`, `ww` | i8 [., 2] | cross-slice blob pairs, in-slice blob pairs, wire pairs, each pair once |
| `logit` | f4 [N] | the real-charge logit |
| `qhat` | f4 [N] | charge estimate, units of 1e4 e (not calorimetric: doc 11 sec 4.2) |

The model must not write to its inputs (the service wraps them without a copy).
