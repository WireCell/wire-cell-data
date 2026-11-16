# SBND: DV2-MAX neutrino-vertex model, `aurora_space`

The model bundle of the DV2-MAX vertexer, the deep-learning neutrino-vertex step of the toolkit's
`clus` pattern recognition (`TaggerCheckNeutrino`, `vertex_method` `dv2_snap` / `dv2_split` /
`dv2_split_move`). Select it with the TLA / config
`dv2_bundle = 'dv2max/sbnd/aurora_space/bundle.json'`, found through `WIRECELL_PATH`. The Python
inference code is in the toolkit, `pyutil/python/DV2_Vertex.py` and `pyutil/python/dv2max_vertexer/`
(its README documents the inputs, outputs and settings). CPU only, 1 thread, fp32.

Unlike `r7_cloud`, this model also takes the call's **space points** (format `dv2max-bundle-v2`, the
`space` section of `bundle.json`). These are the Bee clustering-layer points of every in-scope imaging
cluster that has a point within 1 cm of the DL input cloud. The toolkit collects them at the DL call
(`PatternAlgorithms::dv2_collect_space`), with the same functions as the `T_dlvtx_space` writer of the
training data. A toolkit without space-point support refuses this bundle at load (unknown format).

| file | what |
|---|---|
| `bundle.json` | manifest: format `dv2max-bundle-v2`, the `space` section, every inference setting, the ranker feature names in order, provenance, and the sha256 of every other file (checked at load) |
| `pass1.pt` | pass-1 network (coarse vertex, 1 cm voxels), fp32 SWA state dict, 20.1 M parameters |
| `pass2.pt` | pass-2 network (fine vertex, 0.3 cm voxels), fp32 SWA state dict, 20.1 M parameters |
| `ranker_fold0.txt` | LightGBM candidate ranker (text model) |
| `isotonic_fold0.json` | isotonic calibration of the ranker score to P(vertex < 1 cm) |

- **Input:** the cloud points (x, y, z, charge) of one DL-vertex call, plus the space points of the
  clusters touching it, in the raw reco frame. The model works in the label frame,
  x -= sign(x) * 0.29 cm; the returned vertex is shifted back. Space-point charge is the Bee charge;
  cloud charge is dQ x 0.2474433061809911. Both constants were measured on the training events.
- **Training:** SBND MC (BNB and intrinsic nue CC) from the Aurora production `aurora-r2` with
  `T_dlvtx_space`. 477,997 training events (fold 1); the networks are the same architecture as
  `r7_cloud`. Pass 1: 140,000 steps, pass 2: 70,000 steps, SWA, seed 0. Ranker: LightGBM LambdaMART on
  300,000 training calls, re-fit from the saved training inputs (same union rows and settings; the re-fit
  is not bit-identical to the first fit, so the numbers below are the re-fit's own). Built by
  `dv2max_vertexer/make_bundle.py` on 2026-11-14.
- **Validation (hold-out fold 0, 15,000 labeled events never trained on: 7,397 BNB + 7,603 nue CC;
  pass-0 calls, no containment cut):** vertex within 1 cm of truth for 0.929 of events (nu_e CC 0.951,
  nu_mu CC 0.955, NC 0.717); the production final vertex gets 0.612 on the same events.
- **Toolkit check (2026-11-14):** all 30,000 hold-out calls (both passes) run through the toolkit entry point
  (`DV2_Vertex`, CPU fp32, 1 thread) give < 1 cm 0.9290 vs 0.9286 for the offline (GPU) predictions; 98.7 % of
  the picks are within 0.1 cm of the offline pick, the rest are near-ties. In the chain the space points the
  toolkit collects are byte-identical to the `T_dlvtx_space` rows (339 / 339 calls). CPU per call: median 1.7 s
  (`r7_cloud` 0.53 s on the same calls and node). Details in the toolkit's `dv2max_vertexer/README.md`.
- **Size:** about 165 MB, mostly the two `.pt` files (about 80 MB each).

`SHA256SUMS` covers all five files; `bundle.json` carries the same hashes for the four model files.
