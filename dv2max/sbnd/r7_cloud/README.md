# SBND: DV2-MAX neutrino-vertex model, `r7_cloud`

The model bundle of the DV2-MAX vertexer, the deep-learning neutrino-vertex step of the toolkit's
`clus` pattern recognition (`TaggerCheckNeutrino`, `vertex_method` `dv2_snap` / `dv2_split` /
`dv2_split_move`). The toolkit finds it through `WIRECELL_PATH` with the TLA / config
`dv2_bundle = 'dv2max/sbnd/r7_cloud/bundle.json'` (the SBND default). The Python inference code is in the
toolkit, `pyutil/python/DV2_Vertex.py` and `pyutil/python/dv2max_vertexer/` (its README documents the
inputs, outputs and settings). CPU only, 1 thread, fp32.

| file | what |
|---|---|
| `bundle.json` | manifest: format `dv2max-bundle-v1`, every inference setting, the ranker feature names in order, provenance, and the sha256 of every other file (checked at load) |
| `pass1.pt` | pass-1 network (coarse vertex, 1 cm voxels), fp32 SWA state dict, 20.1 M parameters |
| `pass2.pt` | pass-2 network (fine vertex, 0.3 cm voxels), fp32 SWA state dict, 20.1 M parameters |
| `ranker_fold0.txt` | LightGBM candidate ranker (text model) |
| `isotonic_fold0.json` | isotonic calibration of the ranker score to P(vertex < 1 cm) |

- **Input:** the cloud points (x, y, z, charge) of one DL-vertex call, in the raw reco frame. The model
  works in the label frame, x -= sign(x) * 0.31 cm; the returned vertex is shifted back.
- **Training:** SBND MC (BNB and intrinsic nue CC), the r7 DL-vertex dump, outer family f0: trained on
  folds 1-4, fold 0 held out. Seed 0. Built by `dv2max_vertexer/make_bundle.py` on 2026-10-28.
- **Validation (held-out fold 0, all 5,038 calls):** vertex within 1 cm of truth for 0.8495 of calls
  (offline GPU pipeline, pass-0 calls: 0.8484); the re-fit ranker reproduces all 5,030 offline fold-0
  top-1 picks.
- **Size:** about 165 MB, mostly the two `.pt` files (about 80 MB each).

`SHA256SUMS` covers all five files; `bundle.json` carries the same hashes for the four model files.
