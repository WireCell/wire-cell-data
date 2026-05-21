# PDVD DNN-ROI TorchScript models

TorchScript (`.ts`) models loaded by the wire-cell-toolkit DNN-ROI node
`DNNROIFindingMultiPlane` for ProtoDUNE Vertical Drift. All are exported with
`DNN_ROI_SP/scripts/to_torchscript.py` from the full-corpus 6-channel SDCC
training campaign (DAGMan 287, 2026-05-20/21) and output `sigmoid`
probabilities in `[0, 1]` (no extra sigmoid needed in Wire-Cell).

## Production deployables

The three files actively wired by the toolkit. The current default in
`simulation/toolkit/pdvd/wct-nf-sp-dnnroi.jsonnet` is the INT8 primary
(`pipe_qat_nestedunet_6ch_ep0_int8.ts`); flip `dnnroi_model` to a different
row to swap.

| file | input ch | precision | size | role | run with |
|---|---|---|---|---|---|
| `pipe_base_mbv3_6ch.ts`              | 6 | FP32       | 20.4 MB | FP32 baseline (no KD)         | `run_nf_sp_dnnroi_evt.sh -M dnnroi/pdvd/pipe_base_mbv3_6ch.ts` |
| `pipe_distill_nestedunet_6ch.ts`     | 6 | FP32       | 20.4 MB | FP32 best KD                  | `run_nf_sp_dnnroi_evt.sh -M dnnroi/pdvd/pipe_distill_nestedunet_6ch.ts` |
| `pipe_qat_nestedunet_6ch_ep0_int8.ts`| 6 | INT8 (QAT) | 10.8 MB | **INT8 primary (default)**    | `run_nf_sp_dnnroi_evt.sh -D cpu -M dnnroi/pdvd/pipe_qat_nestedunet_6ch_ep0_int8.ts` |

## Staged / diagnostic

Not wired by default; kept so the user can re-run the §11 / §12.4
comparisons without re-exporting from checkpoints. Both originate from the
Transformer-teacher chain of DAGMan 287 — kept as a reference companion to
the production NestedUNet-teacher chain above.

| file | input ch | precision | size | role |
|---|---|---|---|---|
| `pipe_distill_transformer_6ch.ts`        | 6 | FP32       | 20.4 MB | FP32 KD-Tx, used in §11 as a same-architecture FP32 reference for INT8-Tx |
| `pipe_qat_transformer_6ch_ep3_int8.ts`   | 6 | INT8 (QAT) | 10.8 MB | Tx INT8 candidate at epoch 3; narrows the §11 top-CRP regression but did not clear the strict §12.4 ≤10 % nzpx gate (an5 12.57 %). Held back from production pending an explicit decision; not the canonical Tx INT8. |

(The previous canonical Tx INT8, `pipe_qat_transformer_6ch_int8.ts`, was
derived from epoch 19 — the last-epoch fakequant that the un-patched
`scripts/qat_kd_finetune.py` shipped by default. Per
`DNN_ROI_SP/docs/qat_deployable_diagnostic_2026-05-21.md`, that ep19 ckpt
was strictly dominated by ep3 on labeled Dice (0.7550 vs 0.7772) **and** on
top-CRP over-emission. The ep19 `.ts` was removed in the 2026-05-21
cleanup; if the user ever wants it back for a controlled comparison,
re-export from `checkpoints/pdvd_qat_transformer_6ch/qat_int8_state.pth.ep19`
on wcgpu1.)

## Provenance

All exports trace back to DAGMan cluster **287** on SDCC
(`sgpu0004`, 2× L40S, 2026-05-20 22:12 → 2026-05-21 06:51 EDT,
~8 h 40 min wall total). The training corpus is the 6-channel PDVD
mix (1 000 train + 200 val + 400 held-out test, 125/25/50 events per
anode × 8 anodes, `pdvd_anode{0..7}_6ch_th150_pad3.h5`).

### Production deployables

| field | `pipe_base_mbv3_6ch` | `pipe_distill_nestedunet_6ch` | `pipe_qat_nestedunet_6ch_ep0_int8` |
|---|---|---|---|
| Architecture | MobileNetV3-large UNet | MobileNetV3-large UNet | QuantizableMobileNetV3-UNet, INT8 |
| Run-id | `pdvd_mobilenetv3_all_6ch` | `pdvd_distill_nestedunet_6ch` | `pdvd_qat_nestedunet_6ch` |
| Checkpoint | `CP97.pth` (best-val ep 97) | `CP35.pth` (best-val ep 35) | `qat_int8_state.pth.ep0` (best-by-post-convert) |
| Training | 6-ch baseline, no KD, 100 ep | NestedUNet teacher + feature-map KD, 100 ep | QAT-KD INT8, 20 ep, warm-started from KD-NU (`pdvd_distill_nestedunet_6ch`) |
| TorchScript mode | trace | trace | trace |
| Held-out test (400 ev) Dice | 0.7538 | **0.7816** | 0.7797 |
| Held-out test eff_roi / pur_roi | 0.7135 / 0.8594 | 0.7520 / 0.8537 | 0.7533 / 0.8490 |

The KD-NestedUNet student (`pdvd_distill_nestedunet_6ch`) is the strongest
FP32 model on test (Dice 0.7816, eff_roi 0.7520 — both #1 of the 5 FP32
runs). The INT8 primary (`pdvd_qat_nestedunet_6ch` epoch 0) keeps 99.7 % of
that FP32 Dice (0.7797 = −0.27 % vs FP32 KD parent) and clears the toolkit
§12.4 ≤10 % nzpx gate on all 8 anodes (worst case 9.34 % on anode 5). The
direct-MBV3 baseline is shipped as the no-KD reference.

The INT8 primary's epoch choice (ep 0) is governed by post-convert dice
peaking early in QAT, not the trainer's fakequant `val_dice`. See
`DNN_ROI_SP/docs/qat_deployable_diagnostic_2026-05-21.md` for the full
diagnostic and `DNN_ROI_SP/scripts/qat_kd_finetune.py`'s
best-by-post-convert tracking that lands canonically going forward.

### Staged / diagnostic

| field | `pipe_distill_transformer_6ch` | `pipe_qat_transformer_6ch_ep3_int8` |
|---|---|---|
| Architecture | MobileNetV3-large UNet | QuantizableMobileNetV3-UNet, INT8 |
| Run-id | `pdvd_distill_transformer_6ch` | `pdvd_qat_transformer_6ch` |
| Checkpoint | `CP99.pth` (best-val ep 99) | `qat_int8_state.pth.ep3` |
| Held-out test Dice | 0.7680 | 0.7772 |
| Notes | Diagnostic FP32 anchor for §11 / §12.4 same-arch INT8 comparison. Not the shipped FP32 deployable. | Tx-chain INT8 candidate. +1.20 % vs FP32 KD-Tx parent on labeled test Dice. Narrows the §11 top-CRP regression (an4 11.27 %→7.60 %, an5 17.63 %→12.57 %) but an5 still exceeds the §12.4 10 % strict gate; ep19 was deleted as superseded but no Tx INT8 .ts is wired as canonical until the residual is resolved (see `DNN_ROI_SP/memory/pdvd_int8_top_crp_oversegmentation.md`). |

`to_torchscript.py` falls back to `torch.jit.trace` (the encoder `break`
and the INT8 graph cannot be scripted); each export is verified by an
eager-vs-TorchScript `allclose` (max abs diff 0.00e+00 for all five files).

## Input layout

C++ tensor order is `(batch=1, ntags, nchannels, nticks)`:

- `ntags` = **6**.
- `nchannels` = **952** = the two induction planes stacked (U 476 + V 476).
  `DNNROIFindingMultiPlane` is configured with `planes:[0,1]`; the W collection
  plane is not consumed (passed through from standard SP gauss).
- `nticks` = **1600**, from PDVD's raw `6400` ticks after `tick_per_slice=4`
  downsampling inside the C++ node.

This `(1, 6, 952, 1600)` toolkit input matches the PDVD training shape exactly
(`x_range=[0,952]`, `y_range=[0,1600]`).

**6-channel input** — `ntags=6`, trace tags in order:

```
loose_lf{A}, mp2_roi{A}, mp3_roi{A}, tight_lf{A}, decon_charge{A}, gauss{A}
```

The two new tags relative to the previous 4-ch deployment (`tight_lf` and
`decon_charge`) must be emitted by PDVD's `OmnibusSigProc` chain in debug +
multi-plane-protection mode — the same way PDHD 6-ch deployment works. The
order matches the PDHD 6-ch sibling exactly.

## Per-channel normalization

The 6-ch models are trained on inputs divided by **per-channel** z-scales.
Wire-Cell's `DNNROIFindingMultiPlane` applies one scalar `input_scale`, so the
per-channel division is baked into each `.ts` as a fixed normalization layer;
the models run with `input_scale = 1.0` (set by
`protodunevd/dnnroi_mp.jsonnet`).

A single set is baked into all `.ts` files — the **cross-anode mean**:

```
[766.1332, 4000.0, 4000.0, 762.4834, 1679.252, 11827.907]
        (loose_lf, mp2_roi, mp3_roi, tight_lf, decon_charge, gauss)
```

The per-anode z-scales differ — z[0] runs ~50 % higher on the top CRP
(anodes 4-7) than the bottom CRP (anodes 0-3). The cross-anode mean is an
approximation; the unified vs split study
(`DNN_ROI_SP/docs/pdvd_unified_vs_split_study.md`) shows one mixed model
matches the per-half specialists, so a single set is shipped.

## Consumer

Loaded by the toolkit C++ node `DNNROIFindingMultiPlane`. Wired by
`cfg/pgrapher/experiment/protodunevd/dnnroi_mp.jsonnet`; driven by
`DNN_ROI_SP/simulation/toolkit/pdvd/run_nf_sp_dnnroi_evt.sh` (`-M <model>`
selects the `.ts`).

## Limitations

- Trained on **all 8 PDVD anodes** (bottom CRP = anodes 0-3, top CRP =
  anodes 4-7) — the previous 4-channel deployment's "anodes 4-7
  out-of-domain" caveat no longer applies. See
  `DNN_ROI_SP/docs/pdvd_unified_vs_split_study.md` for evidence one model
  handles both halves.
- The W collection plane is not processed; the toolkit jsonnet routes it
  through a `PlaneSelector` passthrough of standard SP gauss.
- INT8 QAT models run on **CPU only** (x86 quantized backend); they cannot
  be placed on a GPU device.
- The INT8 primary (NU ep0) is the result of a deployable-selection patch
  (best-by-post-convert dice). For runs that pre-date the patch, the
  trainer's last-epoch `qat_int8_state.pth` should not be assumed to be
  the best post-convert deployable — see the §12.6 follow-up in
  `DNN_ROI_SP/docs/sdcc_full_training_campaign.md` for details.
- Cross-anode-mean z-scales are an approximation — see
  *Per-channel normalization* above.
