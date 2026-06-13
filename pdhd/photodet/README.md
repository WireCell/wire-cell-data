# PDHD photon / optical model for WireCell QLMatching

This directory holds the optical light-prediction model for ProtoDUNE-HD (PDHD)
charge–light matching (`QLMatching`), the analogue of `sbnd/photodet/`.

## TL;DR — do we have the correct settings and files?

The WireCell toolkit's `QLMatching` predicts light with **one** mechanism, the
closed-form **`SemiAnalyticalModel`** (the model SBND/LArSoft uses). It has **no
voxel photon-library loader**. So the relevant file for PDHD is a
`semi-analytical-pdhd.json`, *not* a photon-library ROOT file.

`semi-analytical-pdhd.json` here is now complete **for its scope** (geometry +
Gaisser–Hillas tables + OpDet list). It is **not yet runnable** with the stock
toolkit — two follow-ups are required (see "Remaining work").

## What PDHD uses in DUNE (checked against `duneopdet v10_20_08d00`)

DUNE provides **two** light-prediction routes for PDHD:

1. **Voxel photon library** (`protodune_hd_pdfastsim_pvs`, the LArSoft default in
   `services_protodunehd.fcl`). ROOT visibility grids on
   `/cvmfs/dune.osgstorage.org/.../PhotonPropagation/LibraryData/`:
   - `Photon_library_protoDUNEhd_..._v2_refactored_nonActive.root` — 94 MB, 122×67×93 (default)
   - `lib_protodunehd_v6_refactored_Ar_..._94x87x78_ExternalVolume.root` — 90 MB
   - `libext_protodunehd_v6_Ar_Baseline_..._25x25x25_landau.root` — 30 MB
2. **Semi-analytical** (`protodune_hd_pdfastsim_par`):
   `protodune_hd_vuv_hits_parameterization` — flat X-ARAPUCAs only
   (`DomePDCorr: false`), **no reflected light** (`DoReflectedLight: false`).

The toolkit only supports route 2, so that is what `semi-analytical-pdhd.json`
encodes. The library ROOTs would need new C++ (a grid loader/interpolator) and
are not fetched here.

## Provenance of `semi-analytical-pdhd.json`

| section | source |
|---|---|
| `VUVHits` (GH tables) | DUNE `duneopdet v10_20_08d00` `opticalsimparameterisations_dune.fcl` :: `protodune_hd_vuv_hits_parameterization` (`GH_PARS_flat_protoDUNEhd`) — copied verbatim |
| `VISHits` | empty `{}` — PDHD has no reflected/VIS light |
| `Geometry` | active volume from `wire-cell-bee3/docs/protodune_geometry.md` TPC boxes; central cathode at `x = 0` |
| `OpDets[].x/y/z` | toolkit `cfg/pgrapher/experiment/pdhd/pdhd-opdet-geom.json` (mm→cm), 160 windows |
| `OpDets[].h,w` | `ArapucaSingleAcceptanceWindow` box (y=10, z=47.75 cm), gdml `protodunehd_v2_refactored`; single & double readout share these in-plane dims |
| `OpDets[].type/orientation` | `0` / `0` — all flat X-ARAPUCAs, anode/cathode-facing |

`vuv_absorption_length` is set to `2000` cm (effectively off), matching the SBND
convention: VUV attenuation is already folded into the Gaisser–Hillas fit, so the
extra `exp(-d/λ)` factor is left near-unity to avoid double-counting.

## Remaining work (not data files — toolkit code/config)

1. **Code knob `doReflectedLight`.** `QLMatching::configure` hardcodes
   `doReflectedLight=true` and requires a non-empty `VISHits` object. With
   PDHD's empty `VISHits` this would crash. Add a config flag (default `true` ⇒
   SBND bit-identical; set `false` for PDHD). Until then this file cannot be
   loaded by the stock `QLMatching`.
2. **A PDHD `qlmatching.jsonnet`.** Must set `nchan=160`,
   `active_opdet_types=[0]` (X-ARAPUCA, **not** the SBND PMT default `[1]`),
   `semimodel_file=pdhd/photodet/semi-analytical-pdhd.json`, and the
   visibility→PE factors `VUVEfficiency` (likely uniform for PDHD) + `QtoL`,
   plus any `ch_mask`. These belong in the QLMatching config, not in this file.

The voxel-library route remains an option if higher fidelity is needed, but it is
a larger effort (new toolkit loader + the 90+ MB ROOT grids).
