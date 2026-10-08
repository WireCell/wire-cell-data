# DUNE FD-VD low-energy drift-distance model (three views, deployed 2026-10-07)

A TorchScript regressor that estimates the drift distance of a low-energy cluster from the width of its deconvolved
charge (diffusion), for the drift veto of `FdvdLowEQLMatching` (toolkit `match`). Docs: wcp-porting-validation
`fdvd_sim/docs/` 35 (training sample), 36 (design notes), 37 (toolkit port, gates), 38 (production campaign and the
2σ veto).

| File | Content |
|---|---|
| `e4-fdvd35-fast-s0-bf16.ts` | the model: three views (U, V, W crops of 256 channels × 1024 ticks) with wire-crossing attention, 1.99 M parameters, seed 0, bfloat16 autocast inside. Inputs `x [N,3,256,1024]` float32 and `meta [N,4]` float32 (CRM, absolute first channel of U, V, W); outputs mean and log-variance of the drift distance in cm. Run by `FdvdDriftRegressor3View` (`drift.views = 3`) through `TorchTensorSetService` |
| `fdvd-lowe-qlcal-adjophits-e4fdvd35.json` | matcher calibration for **AdjOpHits flashes** (`flash_finder='adjophits'`, doc 21 point 1.0–1.6 µs, 600 cm) with this model's veto calibration. The production campaign of doc 38 used this file |
| `fdvd-lowe-qlcal-opflash-e4fdvd35.json` | matcher calibration for **our flash finder** (`flash_finder='opflash'`, the doc 16 calibration R20pe) with this model's veto calibration |

Use, from `cfg/pgrapher/experiment/fdvd/lowe-reco.jsonnet` (names resolve through `WIRECELL_PATH`):

```
-A drift_model=fdvd/drift/e4-fdvd35-fast-s0-bf16.ts -A frames='<dir>/sp-anode%d.tar.gz' \
-A calibration=fdvd/drift/fdvd-lowe-qlcal-adjophits-e4fdvd35.json      # with -A flash_finder=adjophits
```

**Training sample:** 202,949 single electrons and photons, 0.5–50 MeV, simulated with the FD-VD 1x8x14 signal
processing over the full drift (0–632 cm, charge clipped at the walls), electron lifetime off; 163,052 / 20,301 /
19,596 train / validation / test crops, split by Geant4 shard. 25 epochs, Gaussian negative log-likelihood, best
epoch 20 by validation.

**Performance:**
- Single particles, test split: bias −0.5 cm, mean absolute error 31.2 cm (the earlier collection-only model: bias
  −42 cm); resolution improves with charge, from about 67 cm at 100–150 thousand electrons to 24 cm at 1.2–1.6 million.
- Solar overlay production (doc 38, 10,900 readouts, AdjOpHits flashes, 2σ veto): 35.8 % efficiency, 99.3 % purity,
  8 background false matches in 8,720 test readouts (the earlier model at 3σ: 36.8 %, 99.0 %, 27).
- Toolkit against Python on 54,469 clusters: median difference 0.2 cm, maximum 3.1 cm.

**Read before use:**
- **The model and a calibration file belong together.** The veto compares the model output with the flash time
  through the `drift` block of the calibration (fitted on 1,621 signal clusters of the tuning split). Another seed or
  model needs its own block.
- **The calibration's light block depends on the flash finder.** Pick the file that matches `flash_finder`.
- Only clusters above 100 thousand electrons were used to fit and to test the veto.
- Trained on single isolated particles without electron lifetime; the production has 10.4 ms and pile-up. The
  calibration absorbs the resulting bias (+14 cm before calibration on production clusters).
- The 2σ operating point was chosen and evaluated on the same test split (doc 38 section 2).
- The photon library the matcher also needs (`fdvd-photlib-vis-comb-10cm`) is not in this repository.
