# FD-HD 1x2x6 cascade deghosting: the stage-A head-score columns at the final level (`img/CascadeDeghostingFM`)

TorchScript models for the `ml_head` stage of the wcfm FD-HD imaging job (`wcp-porting-img/wcfm/img.jsonnet`,
`CascadeDeghostingFM`, docs 30 and 31): the final (4-wire) cascade level takes, besides the 15 charge columns of
`../deghost/e2c_L4_4_v2.ts`, three columns from the stage-A comparison head run on the per-view means of the FM
features (`../kd_uni_mbv3_a1_inf01.ts`) over the cell's active wires: the head score, and the mean and the minimum over
the cell's active wires of the per-wire log-softmax of the scores. Everything else of the cascade (levels 0-3, the
repair) is `../deghost/`.

| file | role | threshold | doc | status |
|---|---|---|---|---|
| `e2c_L4_4_head5.ts` | final-level model, **mean of (logit, qhat) over the five doc 29 AB fold-0 models** (seeds 0-4; 18 input columns) | **−0.387396** (doc 14 dev rule on the seed-mean held-out logits) | 31 | **deployed**: PASS on R (νe completeness at matched ghost share +0.0217 [+0.0135, +0.0325] vs the deployed cascade; consumer gate PASS) |
| `head_f0_AB.ts` | stage-A CrossHead (doc 21, f0 mode), mean of the two halves A and B; inputs eU/eV/eW f4[N,128] + has f4[N,3] → score f4[N] | | 29 | used by both final-level models |
| `e2c_L4_4_head.ts` | final-level model, doc 29 AB seed 0 fold 0 alone | −0.11817049980163574 | 30 | superseded: ties the deployed cascade at matched ghost share (+0.0000 [−0.010, +0.012]) — the lowest of the five seeds |

Forward contract of the final-level models (doc 14): `forward(xb[N,18] f4, wq[M] f4, wplane[M] i8, bw_src[E] i8,
bw_dst[E] i8, bw_w[E] f4 (ignored), bb[B,2] i8, bb_in[I,2] i8, ww[K,2] i8) -> (logit[N] f4, qhat[N] f4)`; the
preprocessing and the standardiser are inside each model. `xb[:, 15:18]` = [score, ls_mean, ls_min] as computed by
`CascadeDeghostingFM::head_columns` (per-view means accumulated in double and rounded to binary16 as the training
graphs; log-sum-exp in double; cells without an active FM pixel take the graph's column minimum).

Provenance: exporters `wcp-porting-img/wcfm/scripts/d31_export.py` (ensemble), `d30_export.py` (single seed),
`d29_export_head.py` (head); torch 2.5.1+cu121, loaded by libtorch 2.8 (`TorchTensorSetService`, CPU). Member
checkpoints and their sha256 are in the `.meta.json` files. Parity: scripted ensemble vs the five trainer-side models
max |Δ logit| 1.9e-6 (60 R graphs, 0 flips); C++ (`CascadeDeghostingFM`) vs python on 23 R anode-events: charge
columns exact, head columns Spearman 1.0000, max |Δ logit| 1.6e-3 (docs 30 §3.5, 31 §4).

Checksums: `SHA256SUMS` (sha256sum format, this directory).
