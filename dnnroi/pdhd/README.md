# PDHD DNN-ROI TorchScript models

TorchScript (`.ts`) models loaded by the wire-cell-toolkit DNN-ROI nodes
(`DNNROIFinding` / `DNNROIFindingMultiPlane`). All are exported with
`DNN_ROI_SP/scripts/to_torchscript.py` and output `sigmoid` probabilities
in `[0, 1]` (no extra sigmoid needed in Wire-Cell).

| file | input ch | precision | size | run with |
|---|---|---|---|---|
| `CP43.ts` | 3 | FP32 | 20.4 MB | `run_nf_sp_dnnroi_evt.sh -n 3` (default) |
| `kd_mbv3_transformer_bnKD_6ch.ts` | 6 | FP32 | 20.4 MB | `run_nf_sp_dnnroi_evt.sh -n 6` |
| `qat_mbv3_transformer_bnKD_6ch_int8.ts` | 6 | INT8 (QAT) | 10.8 MB | `run_nf_sp_dnnroi_evt.sh -n 6 -D cpu` |
| `pipe_base_mbv3_6ch.ts` | 6 | FP32 | 20.4 MB | `run_nf_sp_dnnroi_evt.sh -n 6` |
| `pipe_distill_transformer_6ch.ts` | 6 | FP32 | 20.4 MB | `run_nf_sp_dnnroi_evt.sh -n 6` |
| `pipe_qat_transformer_6ch_int8.ts` | 6 | INT8 (QAT) | 10.8 MB | `run_nf_sp_dnnroi_evt.sh -n 6 -D cpu` |

## Provenance

| field | `CP43.ts` | `kd_..._6ch.ts` | `qat_..._6ch_int8.ts` |
|---|---|---|---|
| Architecture | MobileNetV3-large UNet | MobileNetV3-large UNet | QuantizableMobileNetV3-UNet, INT8 |
| Source repo | `DNN_ROI_SP/` | `DNN_ROI_SP/` | `DNN_ROI_SP/` |
| Run-id | `bs1_20260511-210525` | `distill_mbv3_transformer_bnKD_6ch_th150_ep100_l40s_ddp2` | `qat_distill_mbv3_transformer_bnKD_6ch_th150_ep20_l40s_ddp2` |
| Checkpoint | `CP43.pth` | `CP70.pth` | `qat_int8_state.pth` |
| Training | 3-ch baseline | Transformer teacher + bottleneck-feature KD | QAT-KD-C, warm-started from `CP70.pth` |
| TorchScript mode | trace | trace | trace |
| Held-out test Dice / ROI-eff | — | 0.9118 / 0.7609 | 0.8932 / 0.7274 |

`to_torchscript.py` falls back to `torch.jit.trace` because `torch.jit.script`
hits the `break` in the encoder loop; the INT8 quantized graph also cannot be
scripted. The traced UNets are fully convolutional and run at both the
per-plane (`800`) and stacked (`1600`) channel heights.

### Pipeline-reproduced models (2026-05-16)

`pipe_base_mbv3_6ch.ts`, `pipe_distill_transformer_6ch.ts`, and
`pipe_qat_transformer_6ch_int8.ts` are the three models deployed by the
end-to-end run documented in `DNN_ROI_SP/docs/full_pipeline.md` — a baseline,
the best distillation, and its QAT INT8 model, all 6-channel and trained on the
same corpus and split.

| field | `pipe_base_mbv3_6ch.ts` | `pipe_distill_transformer_6ch.ts` | `pipe_qat_transformer_6ch_int8.ts` |
|---|---|---|---|
| Architecture | MobileNetV3-large UNet | MobileNetV3-large UNet | QuantizableMobileNetV3-UNet, INT8 |
| Run-id | `pipe_base_mbv3_6ch` | `pipe_distill_transformer_6ch` | `pipe_qat_transformer_6ch` |
| Training | 6-ch baseline, no KD | Transformer teacher + bottleneck-feature KD | QAT-KD, warm-started from the distillation |
| Held-out test Dice / ROI-eff | 0.9120 / 0.7474 | 0.9107 / 0.7454 | 0.8900 / 0.7305 |

All three pass the toolkit-vs-standalone replay validation (max abs diff
< 1.4e-6; the INT8 model bit-exact) — see `full_pipeline.md` §4.3.

## Input layout

C++ tensor order is `(batch=1, ntags, nchannels, nticks)`:

- `nchannels` = `800` per plane in per-plane (`pp`) mode, or `1600`
  (U+V stacked) in stacked (`mp`) mode. The W collection plane is not consumed.
- `nticks` = `1500`, from PDHD's raw `6000` after `tick_per_slice=4`
  downsampling inside the C++ node.

**3-channel model** (`CP43.ts`) — `ntags=3`, in order:

```
loose_lf{APA}, mp2_roi{APA}, mp3_roi{APA}
```

**6-channel models** — `ntags=6`, in order:

```
loose_lf{APA}, mp2_roi{APA}, mp3_roi{APA}, tight_lf{APA}, decon_charge{APA}, gauss{APA}
```

All six tags are emitted by the standard PDHD `OmnibusSigProc` chain
(debug + multi-plane-protection mode) and require no SP-config change.

## Per-channel normalization (6-ch models)

The 6-ch models are trained on inputs divided by **per-channel** z-scales:

```
[944.6256, 4000.0, 4000.0, 803.7348, 1927.6997, 530.75]
```

Wire-Cell's `DNNROIFinding` can only apply one **scalar** `input_scale` to all
channels, so the per-channel division is **baked into the `.ts` module** as a
fixed normalization layer. Consequently the 6-ch models must run with
`input_scale = 1.0` — the `run_nf_sp_dnnroi_evt.sh -n 6` path sets this
automatically (`dnnroi_pp.jsonnet`). `CP43.ts` keeps the C++ default
`input_scale = 1/4000`.

## Tick padding

The C++ node rebins the time axis by `tick_per_slice=4` before inference and
needs the input tick count to be a multiple of the model's stride alignment.
For the PDHD MobileNetV3-large UNet (no deep stride-2 cascade in the tick
axis: post-rebin width 1500 = 4·375 is not divisible by 8 or higher powers
of 2), the alignment requirement is just `nticks % tick_per_slice == 0`,
i.e. **`nticks` must be a multiple of 4**. PDHD's standard `nticks=6000`
satisfies this with no padding.

The `dnnroi_pp.jsonnet` for PDHD leaves `tick_pad_multiple` unset (defaults
to `tick_per_slice=4`). The C++ node pads to the next 4-multiple before
inference, then crops back to `input_ticks` — a no-op for any
`nticks % 4 == 0` (including 6000, 6400, 8000).

## Consumer

Loaded by the toolkit C++ node `DNNROIFinding` (per-plane sequential: U
and V each run their own forward call sharing one TorchService). Wired by
`cfg/pgrapher/experiment/pdhd/dnnroi_pp.jsonnet`; driven by
`wcp-porting-img/pdhd/run_nf_sp_dnnroi_evt.sh` (`-n 3|6` selects the
input-channel set, `-M <model>` selects the `.ts`).

## Limitations

- Trained on **APA0 only**. Inference on APAs 1–3 is out-of-domain.
- W plane is not processed; downstream jsonnet routes it through a
  `PlaneSelector` passthrough.
- The INT8 QAT model runs on **CPU only** (x86/fbgemm quantized backend);
  it cannot be placed on a GPU device.
