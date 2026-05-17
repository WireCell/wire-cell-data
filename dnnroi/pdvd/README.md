# PDVD DNN-ROI TorchScript models

TorchScript (`.ts`) models loaded by the wire-cell-toolkit DNN-ROI node
`DNNROIFindingMultiPlane` for ProtoDUNE Vertical Drift. All are exported with
`DNN_ROI_SP/scripts/to_torchscript.py` from the PDVD overnight training
campaign (Transformer-teacher lineage) and output `sigmoid` probabilities in
`[0, 1]` (no extra sigmoid needed in Wire-Cell).

| file | input ch | precision | size | run with |
|---|---|---|---|---|
| `base_mbv3_transformer_4ch.ts` | 4 | FP32 | 20.4 MB | `run_nf_sp_dnnroi_evt.sh -M dnnroi/pdvd/base_mbv3_transformer_4ch.ts` |
| `kd_distill_transformer_4ch.ts` | 4 | FP32 | 20.4 MB | `run_nf_sp_dnnroi_evt.sh` (default model) |
| `qat_distill_transformer_4ch_int8.ts` | 4 | INT8 (QAT) | 10.8 MB | `run_nf_sp_dnnroi_evt.sh -D cpu -M dnnroi/pdvd/qat_distill_transformer_4ch_int8.ts` |

## Provenance

| field | `base_mbv3_transformer_4ch` | `kd_distill_transformer_4ch` | `qat_distill_transformer_4ch_int8` |
|---|---|---|---|
| Architecture | MobileNetV3-large UNet | MobileNetV3-large UNet | QuantizableMobileNetV3-UNet, INT8 |
| Source repo | `DNN_ROI_SP/` | `DNN_ROI_SP/` | `DNN_ROI_SP/` |
| Run-id | `pdvd_mobilenetv3_all_4ch` | `pdvd_distill_transformer_4ch` | `pdvd_qat_transformer_4ch` |
| Checkpoint | `CP64.pth` | `CP83.pth` | `qat_int8_state.pth` |
| Training | 4-ch baseline, no KD | Transformer teacher + bottleneck-feature KD | QAT-KD, warm-started from the distillation |
| Best-epoch val Dice | 0.9671 | 0.9738 | 0.9736 |
| Best-epoch val loss | 0.003560 | 0.002895 | 0.002930 |
| ROI eff / pur | 0.8894 / 0.9853 | 0.9006 / 0.9822 | 0.8995 / 0.9804 |

QAT metrics are validation of the *fake-quantized* model during finetuning, not
a separate evaluation of the converted INT8 graph (see
`DNN_ROI_SP/docs/pdvd_overnight_results.md`). `to_torchscript.py` falls back to
`torch.jit.trace` (the encoder `break` and the INT8 graph cannot be scripted);
each export is verified by an eager-vs-TorchScript `allclose` (max abs diff
0.00e+00 for all three).

## Input layout

C++ tensor order is `(batch=1, ntags, nchannels, nticks)`:

- `ntags` = **4**.
- `nchannels` = **952** = the two induction planes stacked (U 476 + V 476).
  `DNNROIFindingMultiPlane` is configured with `planes:[0,1]`; the W collection
  plane is not consumed (passed through from standard SP gauss).
- `nticks` = **1600**, from PDVD's raw 6400 ticks after `tick_per_slice=4`
  downsampling inside the C++ node.

This `(1, 4, 952, 1600)` toolkit input matches the PDVD training shape exactly
(`x_range=[0,952]`, `y_range=[0,1600]`).

**4-channel input** — `ntags=4`, trace tags in order:

```
loose_lf{A}, mp2_roi{A}, mp3_roi{A}, gauss{A}
```

NOTE: the 4th channel is `gauss` — not PDHD's `tight_lf`. All four tags are
emitted by the standard PDVD `OmnibusSigProc` chain in debug +
multi-plane-protection mode and require no SP-config change.

## Per-channel normalization

The PDVD models are trained on inputs divided by **per-CRP** z-scales. Wire-Cell
applies one scalar `input_scale`, so the per-channel division is baked into the
`.ts` as a fixed normalization layer; the models run with `input_scale = 1.0`
(set by `protodunevd/dnnroi_mp.jsonnet`).

A single set is baked — the **cross-CRP mean**:

```
[635.6828, 4000.0, 4000.0, 260.5]   (loose_lf, mp2_roi, mp3_roi, gauss)
```

This is an approximation: per-CRP z[0] spans 609–670 (±5%) and z[3] spans
210–318 (±20%). If a CRP1/2-vs-CRP3/4 accuracy gap matters, the future option
is four per-CRP `.ts` files per model.

## Consumer

Loaded by the toolkit C++ node `DNNROIFindingMultiPlane`. Wired by
`cfg/pgrapher/experiment/protodunevd/dnnroi_mp.jsonnet`; driven by
`wcp-porting-img/pdvd/run_nf_sp_dnnroi_evt.sh` (`-M <model>` selects the `.ts`).

## Limitations

- Trained on **crp1–4 = toolkit anodes 0–3** (the bottom drift volume). Run
  039324's anodes 4–7 (top drift volume) are geometrically mirror-equivalent
  but not in the training corpus — inference on anode4–7 is out-of-domain.
- The W collection plane is not processed; the toolkit jsonnet routes it
  through a `PlaneSelector` passthrough of standard SP gauss.
- The INT8 QAT model runs on **CPU only** (x86 quantized backend); it cannot be
  placed on a GPU device.
