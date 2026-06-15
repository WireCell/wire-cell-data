# `pipe_distill_transformer_6ch.ts` — PDVD DNN-ROI (FP32, KD-Transformer)

Single-file companion to the directory-level [`README.md`](README.md), which is
the authoritative source for the full PDVD DNN-ROI model set, input layout,
normalization, and tick-padding rules. This note documents only this one file.

| field | value |
|---|---|
| file | `dnnroi/pdvd/pipe_distill_transformer_6ch.ts` |
| size | 21,407,103 bytes (≈20.4 MB) |
| architecture | MobileNetV3-large UNet |
| precision | FP32 |
| input channels | 6 |
| output | per-pixel `sigmoid` probability in `[0, 1]` (no extra sigmoid in Wire-Cell) |
| TorchScript mode | `torch.jit.trace` (re-traced 2026-05-23 at per-plane shape) |
| role | **staged / diagnostic — not wired by default** |

## What it is

The FP32 knowledge-distillation **Transformer-teacher** student for PDVD,
exported from DAGMan cluster 287 (SDCC, 2026-05-20/21). It is the
same-architecture FP32 reference used in the §11 / §12.4 INT8-vs-FP32
comparisons against the Transformer INT8 candidate
(`pipe_qat_transformer_6ch_ep3_int8.ts`). It is **not** the shipped FP32
deployable — that is `pipe_distill_nestedunet_6ch.ts` (NestedUNet teacher,
stronger on held-out test). See the directory README's *Staged / diagnostic*
and *Provenance* sections.

| metric (400-event held-out test) | value |
|---|---|
| Dice | 0.7680 |
| run-id | `pdvd_distill_transformer_6ch` |
| checkpoint | `CP99.pth` (best-val ep 99) |

## Input / output

C++ tensor order `(batch=1, ntags=6, nchannels=476, nticks=1600)`, processed
per-plane (U then V) by `DNNROIFinding`. The 6 trace tags, in order:

```
loose_lf, mp2_roi, mp3_roi, tight_lf, decon_charge, gauss
```

Per-channel z-scale normalization is **baked into the `.ts`**; run with
`input_scale = 1.0`. Tick padding must use a multiple of
`tick_per_slice·32 = 128`. Full details (the 5-level stride-2 cascade, the
cross-anode-mean z-scales, the 2026-05-23 per-plane re-trace) are in the
directory [`README.md`](README.md).

## Run with

```
run_nf_sp_dnnroi_evt.sh -M dnnroi/pdvd/pipe_distill_transformer_6ch.ts
```

Wired (when selected) by
`cfg/pgrapher/experiment/protodunevd/dnnroi_pp.jsonnet`. Loaded by the toolkit
C++ node `DNNROIFinding`.
