# PDHD DNN-ROI TorchScript model

`CP43.ts` — MobileNetV3-Large UNet, trained on PDHD APA0 simulation
data, exported via `torch.jit.trace` at the inference shape used by
the wire-cell-toolkit deployment.

## Provenance

| field | value |
|---|---|
| Source repo | `/nfs/data/1/xqian/toolkit-dev/DNN_ROI_SP/` |
| Source run-id | `bs1_20260511-210525` |
| Source checkpoint | `checkpoints/bs1_20260511-210525/CP43.pth` |
| Export script | `scripts/to_torchscript.py` |
| TorchScript mode | trace (script fails due to `break` in encoder loop) |
| Export shape | `(1, 3, 1600, 1500)` |
| Output activation | `sigmoid` (probabilities in [0, 1]) |

## Input layout

The model expects, in C++ tensor order:

```
(batch=1, ntags=3, nchannels=1600, nticks=1500)
```

- `ntags=3` comes from the trace tags `loose_lf{APA}`,
  `mp2_roi{APA}`, `mp3_roi{APA}` produced by the standard PDHD
  signal-processing chain.
- `nchannels=1600` is **U + V planes stacked** (800 wires + 800
  wires) — the first 1600 of PDHD's 2560 channels per APA. The W
  collection plane is not consumed.
- `nticks=1500` comes from PDHD's raw `nticks=6000` after
  `tick_per_slice=4` downsampling inside the C++ node.

The shape is **locked** by the trace — feeding any other shape
will produce wrong outputs (or fatal) at the model call.

## Consumer

This `.ts` is loaded by the wire-cell-toolkit C++ node
`DNNROIFindingMultiPlane` (see
`toolkit/pytorch/src/DNNROIFindingMultiPlane.cxx`). The jsonnet
template that wires it in is
`cfg/pgrapher/experiment/pdhd/dnnroi_mp.jsonnet`.

## Limitations

- Trained on **APA0 only**. Inference on APAs 1–3 is out-of-domain;
  quality is not guaranteed and not benchmarked yet.
- W plane is not processed (model has no W-plane head). Downstream
  jsonnet routes the W plane through a `PlaneSelector` passthrough.
