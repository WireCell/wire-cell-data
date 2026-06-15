# `l1sp_dnn_pdhd_v1.ts` — PDHD L1SP DNN ROI tagger

TorchScript (`.ts`) model loaded by the wire-cell-toolkit L1SP deep-learning
ROI tagger for ProtoDUNE-HD. It is a **per-ROI binary classifier** (not a
per-pixel segmentation U-Net like the `dnnroi/` models): for each candidate ROI
it consumes a short waveform window plus 29 hand-engineered scalar features and
emits a single `sigmoid` score in `[0, 1]`, which is cut at a default threshold
to keep or drop the ROI.

Full machine-readable spec: [`l1sp_dnn_pdhd_v1.meta.json`](l1sp_dnn_pdhd_v1.meta.json).

| field | value |
|---|---|
| file | `l1sp/pdhd/l1sp_dnn_pdhd_v1.ts` |
| size | 917,502 bytes (≈896 KB) |
| task | per-ROI binary classification (keep / drop) |
| output | `score` = `sigmoid` in `[0, 1]`, cut at `default_threshold` |
| **default threshold** | **0.9945** |
| precision | FP32 |

## Inputs

The model `forward` takes **two** tensors (C++ `Pytorch::from_itensor` 4-D
convention, batch `B`):

| input | shape | dtype | contents |
|---|---|---|---|
| `waveform` | `(B, 1, 2, 256)` | float32 | channel 0 = `raw/scale`, channel 1 = `decon/scale`, where `scale = max(|raw|.max, |decon|.max, 1.0)`. Window = full ROI right-padded to 256, **or** ±128 ticks centered on `argmax(|decon|)` clamped to ROI bounds. The dim-1 axis is a dummy to satisfy WCT's 4-D requirement. |
| `scalars` | `(B, 1, 1, 29)` | float32 | the 29 scalar features in `scalar_feature_order` (see meta JSON) |

`nbin = 256`, `amp_floor = 1.0`.

### Scalar feature order (29)

```
nbin_fit, temp_sum, temp1_sum, temp2_sum, max_val, min_val, prev_gap, next_gap,
flag, ratio, temp_sum_pos, temp_sum_neg, n_above_pos, n_above_neg, argmax_tick,
argmin_tick, sig_peak, sig_integral, gmax, gauss_fill, gauss_fwhm_frac,
roi_energy_frac, raw_asym_wide, core_lo, core_hi, core_length, core_fill,
core_fwhm_frac, core_raw_asym_wide
```

The 30th feature (`vae_kl`, `kl_index = 29`) appears in the full
`feature_order` but is **not** part of the model's scalar input — it is the KL
term from the stage-B VAE (`model_n16.pt`, `vae_n_lat = 16`) used in training,
not consumed at inference.

## Output

`score` of shape `(B, 1, 1, 1)`, a `sigmoid` probability in `[0, 1]`. An ROI is
kept when `score ≥ default_threshold = 0.9945`. The threshold convention is the
p99.9 of the data-corpus score distribution from the training run; see the
experiment dir's `notes.md` for the promoted value.

## Provenance

| field | value |
|---|---|
| experiment dir | `/nfs/data/1/xqian/toolkit-dev/l1sp_dl_tagger/experiments/stage_a_pu_round4` |
| VAE checkpoint | `…/experiments/stage_b_vae/model_n16.pt` (`vae_n_lat = 16`) |
| git sha | `708b942b199e2cc7395e9e3468b926b8146e171b` |

## Note on the PDVD sibling

`l1sp/pdvd/l1sp_dnn_pdvd_v1.ts` is the same architecture and I/O layout. The
differences are the training corpus / detector (PDVD `stage_a_pu_round2_pdvd`)
and a much lower `default_threshold` (**0.16** vs 0.9945 here) — the two are
**not** interchangeable; always use the model matching the detector and its
own threshold.
