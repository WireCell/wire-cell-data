# ProtoDUNE-VD charge-only cascade deghoster (round 4, deployed 2026-10-07)

Five TorchScript levels for `CascadeDeghosting` (toolkit `img`); `levels.json` is the ladder with thresholds
(`ml_levels`), final keep logit -2.0. Docs: wcp-porting-validation
`pdvd/docs/nf_sp_img_clus/123_ml-deghosting-round3-busy-isochronous-beam.md` (this round), 121 and 122 (rounds 1-2).

**Training sample:** 4,352 anode-event graphs from 544 simulated events: 200 CORSIKA cosmic events, and built from
them 40 rotated, 24 three-cosmic overlay, 24 cosmic + rotated, 96 beam particle (e+, pi+, K+, mu+, p; 0.5-3 GeV/c,
Geant4 gun at the beam entry) + cosmic, and 160 events with one muon at 0-2 degrees to the CRP planes. Five folds,
fold 0 exported, one seed. Cross-validated AP 0.982 / 0.986 / 0.989 / 0.989 / 0.985.

**Held-out simulation, against round 2 (the previous content of this directory) and the production chain:**

| sample | completeness: production / round 2 / round 4 | ghost area removed: production / round 2 / round 4 |
|---|---|---|
| cosmics (50 events) | 0.9907 / 0.9986 / 0.9988 | 0.386 / 0.656 / 0.654 |
| beam + cosmic (25) | 0.9901 / 0.9985 / 0.9988 | 0.372 / 0.612 / 0.639 |
| beam particle's own charge (25) | 0.9889 / 0.9946 / 0.9971 | - |
| muon at 0-2 degrees to the CRP (50) | 0.9918 / 0.9974 / 0.9972 | 0.392 / 0.468 / 0.818 |

C++ against python on the same graphs: largest logit difference 1.1e-5, no decision flips.

**Read before use:**
- The doc's pre-registered rule was met only at this keep value (-2.0), which was chosen after the rule-selected
  value (-3.296, the plateau rule of doc 122 sec 9.2 on this model's own curve) had been scored; the
  ghost-fraction-0.2 rule value, -4.117, is kept in `e2c_L4_4.meta.json` as `threshold_rule_value`. Adopted by the owner
  2026-10-07.
- On the busiest beam-run data anodes (run 39305) this model removes more solved charge than the simulation
  predicts (round 4 / production 0.83 at 10-20 k production objects, simulation 0.97). Whether that is ghost or
  true charge is not known (doc 123 sec 6). No gate after clustering, no hand scan.
- Frames must carry frame time 0, or set `slice_start_relative` (toolkit e2d4f34c; doc 122 sec 9).
- Round 2 is this directory at wire-cell-data f397e0a.
