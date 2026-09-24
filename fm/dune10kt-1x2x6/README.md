# DUNE FD-HD 1x2x6 foundation-model (FM) TorchScript models

TorchScript (`.ts`) models loaded by the wire-cell-toolkit `TorchService`
(`pytorch/src/TorchService.cxx`, `torch::jit::load`, libtorch 2.8) for the
Wire-Cell FM integration campaign (`wcp-porting-img/wcfm/docs/01`, `03`).
Exported with `WC_FM_DINO/sdcc/export_mbv3_ts.py`; each `.ts` carries a
`fm_meta.json` extra file whose copy sits beside it as `<name>.meta.json`.

| file | model | params | precision | size | mode |
|---|---|---|---|---|---|
| `kd_uni_mbv3_a1_inf01.ts` | unified cross-plane dense MobileNetV3-Large U-Net student (WC_FM_DINO docs/37 "M2"), 128-d trunk | 3.45 M deployed (5.20 M in training incl. the never-run encoder tail and the charge/occupancy heads) | FP32 | 14.2 MB | `torch.jit.script` |

## Provenance

| field | `kd_uni_mbv3_a1_inf01.ts` |
|---|---|
| Source repo | `WC_FM_DINO` (`lastgeorge/WC_FM_DINO`), `models/mobilenetv3_unet.py` + `models/dense_mae_adapter.py` |
| Training | KD from the PoLAr-MAE `pm4_{U,V,W}` teachers (PCA-64 targets) + teacher-free instance InfoNCE (lambda 0.1), one plane-blind student for U, V, W; `sdcc/config_kd_uni_mbv3_a1_inf01.json`, 36 000 steps, SDCC Condor cluster 1304 |
| Checkpoint | `CONDOR_OUT/kd_campaign/checkpoints/kd_uni_mbv3_a1_inf01/checkpoint_step36000.pt` on SDCC, sha256 `5ed9c29053bea52142ddf9510e7b2836495d728414c9798b452b779d771e1440` (49 126 754 B), copied to `WC_FM_DINO/kd_checkpoints/` 2026-09-24 |
| Export env | torch 2.5.1+cu121, torchvision 0.20.1 (the toolkit direnv python), 2026-09-24 |
| TorchScript mode | `script` of a wrapper (`FMDenseStudent`) that re-expresses `MobileNetV3_UNet._forward_body` with the encoder cut into its five recorded stages (indices 0, 2, 4, 7, 13), so the `UpBlock` shape branch survives (a traced model would bake one canvas size) |
| Parity | 60 events (20 per plane, `packed_numu_truth_apa0_{U,V,W}_20k.npz`): scripted vs the real torch-2.10 adapter forward max abs diff 2.0e-5, min cosine 0.99999964; scripted vs eager in the export env bit-identical; GPU (RTX 4090, cuDNN defaults) vs CPU reference max abs 9.2e-3, min cosine 0.999992 |
| Load checks | `build/pytorch/check_load_tsmodel` (libtorch 2.8.0 shim): loaded; `torch.jit.load` under torch 2.10: forward OK |
| CPU latency (this box, load 20, 60 full-anode canvases, median 760 x 1343) | 1372 / 487 / 423 ms per canvas at 1 / 8 / 16 threads; GPU 14 ms |

## Input / output contract

Tensor order is `(batch=1, 2, H, W)` float32 NCHW, single output `(1, 128, H, W)` float32.

- `H` = anode-local channel index (U: 0-799, V: 800-1599 rebased to 0, W: 1600-2559 rebased to 0
  with columns 0-479 = WCT face 1 (-x) and 480-959 = face 0 (+x)); `W` = time slice of 4 ticks.
- channel 0 = `FeatureLogTransform(q)` with the plane's `VIEW_NORM`
  (`y = 2 (log10(q+m) - log10 m) / (log10(M+m) - log10 m) - 1`, no clip; U (2.77, 144977),
  V (2.97, 157944), W (3.75, 83861.2)); channel 1 = 1.0 at active pixels, 0 elsewhere.
- canvas = tight bounding box of the active pixels, floored at 64 in both dimensions by high-side
  zero padding; one canvas per event and plane; gather the output at the active pixels.
- the pixel value is `0.25 x` the sum of the 4 `gauss` ticks (doc 01 sec 2.1; the exact scale is
  fixed by the F2 oracle).  No normalisation and no sigmoid are inside the model.

The sparse CAP student `kd_uni_a1_inf01` (WarpConvNet + flash-attn) is not scriptable; it is kept
as a checkpoint in `WC_FM_DINO/kd_checkpoints/` for the Triton path (doc 01 F6).
