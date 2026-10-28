# DUNE FD-HD 1x2x6: W9 fine-tuned cascade with projected FM means at L4 ("D3 raw"; record, not production)

The W9 single-detector fine-tuned cascade (`../deghost_w9ft/`) with its final level replaced by a **114-column L4 model**
that reads the per-view means of the FM student features over each cell's active wires, projected on a shared
128 -> 32 PCA basis (SDCC wcfm doc 53 parts 43/44, the "D3 raw" arm, seed 0 fold 0; export `export/w9_ft_fmraw/fdhd`;
Xin approved option C 2026-10-28). Run by the toolkit `img` component `CascadeDeghostingFMRaw` (toolkit c5be2d96) with
the production FM student `../kd_uni_mbv3_a1_inf01.ts`; no head network. Selected by the wcp FD-HD imaging TLA
`deghost_w9ft_fmraw` (default off).

| file | role |
|---|---|
| `e2c_L0_U.ts` .. `e2c_L3_8.ts` | levels 0-3, byte-identical to `../deghost_w9ft/` |
| `e2c_L4_4.ts` | final level: xb[N,114] = 15 charge columns + [x_0 (32), x_1 (32), x_2 (32), has_0, has_1, has_2], standardiser folded in |
| `fmraw_pca_k32.json` / `.npz` | {mean[128], basis[128][32], k, dim, var_frac 0.972}: x_p = ((e_p - mean) @ basis) * has_p |
| `levels.json` | the ladder; L4 keep -1.1163 is the CV gf-0.2 placeholder, not an after-clustering operating point |
| `tables/` | SDCC keep-rule tables |

`SHA256SUMS` (from the export) covers the five `.ts` and the two basis files.

**Not production.** After clustering, at matched post-clustering ghost share against the shipped FD-HD chain
(`deghost_w9ft` at +1.0751) on the S1 and W6 nu_e samples, it is within +-0.002 completeness on every class with every
interval through zero (wcp-porting-validation pr_ml doc 30 sec 9.2; SDCC doc 53 part 45). Its matched pooled keep would
be +1.0719. Kept here as the record of the test.
