# `pipe_distill_transformer_6ch.ts` — PDHD DNN-ROI (FP32, KD-Transformer)

Single-file companion to the directory-level [`README.md`](README.md), which is
the authoritative source for the full PDHD DNN-ROI model set, input layout,
normalization, and tick-padding rules. This note documents only this one file.

| field | value |
|---|---|
| file | `dnnroi/pdhd/pipe_distill_transformer_6ch.ts` |
| size | 21,410,681 bytes (≈20.4 MB) |
| architecture | MobileNetV3-large UNet |
| precision | FP32 |
| input channels | 6 |
| output | per-pixel `sigmoid` probability in `[0, 1]` (no extra sigmoid in Wire-Cell) |
| TorchScript mode | `torch.jit.trace` |
| role | pipeline-reproduced FP32 distillation model |

## What it is

The FP32 knowledge-distillation **Transformer-teacher** student for PDHD — the
"best distillation" leg of the end-to-end run documented in
`DNN_ROI_SP/docs/full_pipeline.md`. Trained 6-channel on the same corpus and
split as its sibling baseline (`pipe_base_mbv3_6ch.ts`) and its QAT INT8 model
(`pipe_qat_transformer_6ch_int8.ts`).

| metric (held-out test) | value |
|---|---|
| Dice / ROI-eff | 0.9107 / 0.7454 |
| run-id | `pipe_distill_transformer_6ch` |
| training | Transformer teacher + bottleneck-feature KD |

Passes the toolkit-vs-standalone replay validation (max abs diff < 1.4e-6) —
see `full_pipeline.md` §4.3.

## Input / output

C++ tensor order `(batch=1, ntags=6, nchannels, nticks)`. `nchannels = 800`
per-plane (`pp` mode) or `1600` (U+V stacked, `mp` mode); the traced UNet is
fully convolutional and runs at both heights. `nticks = 1500` (PDHD raw 6000
after `tick_per_slice=4`). The 6 trace tags, in order:

```
loose_lf, mp2_roi, mp3_roi, tight_lf, decon_charge, gauss
```

All six tags come from the standard PDHD `OmnibusSigProc` chain (debug +
multi-plane-protection mode) — no SP-config change needed. Per-channel z-scale
normalization is **baked into the `.ts`**; run with `input_scale = 1.0`. Tick
padding for PDHD only requires `nticks % 4 == 0`. Full details in the directory
[`README.md`](README.md).

## Run with

```
run_nf_sp_dnnroi_evt.sh -n 6
```

(then `-M dnnroi/pdhd/pipe_distill_transformer_6ch.ts` to select this `.ts`).
Wired by `cfg/pgrapher/experiment/pdhd/dnnroi_pp.jsonnet`; loaded by the toolkit
C++ node `DNNROIFinding`.

## Limitations

Trained on **APA0 only** — inference on APAs 1–3 is out-of-domain. The W plane
is not processed (routed through a `PlaneSelector` passthrough).
