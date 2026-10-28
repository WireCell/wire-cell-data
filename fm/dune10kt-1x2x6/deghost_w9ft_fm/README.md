# DUNE FD-HD 1x2x6: learned cascade deghosting, W9 fine-tune with the head-score stage at L4 (TorchScript)

The W9 single-detector fine-tuned cascade (`../deghost_w9ft/`, wcp-porting-img `wcfm/docs/53_cross-detector-deghosting-plan.md`
§10.2 part 38) with its final level replaced by an **18-column L4 model** that reads the production head-score columns
(doc 53 part 41, the "D2" arm `e2-projq-fm`, seed 0 fold 0), exported at SDCC as `export/w9_ft_fm/fdhd`
(Xin's instruction to the SDCC session, 2026-10-27 ~18:40 EDT). Run by the toolkit `img` component `CascadeDeghostingFM`
(the doc 30 head stage) with the production head objects `../kd_uni_mbv3_a1_inf01.ts` (FM student) and
`../deghost_head/head_f0_AB.ts` (stage-A mean head, sha256 `a7cf6a45...`). The imaging config (wcp `wcfm/wct-img-all.jsonnet`)
selects these files with the TLA `deghost_w9ft_fm`.

| file | level | cut width (wires) | super-wire k | threshold (logit) | role |
|---|---|---|---|---|---|
| `e2c_L0_U.ts` | 0 | uncut | 16 | -2.6525 | prune below (byte-identical to `../deghost_w9ft/`) |
| `e2c_L1_32.ts` | 1 | 32 | 8 | -2.2468 | prune below (byte-identical to `../deghost_w9ft/`) |
| `e2c_L2_16.ts` | 2 | 16 | 4 | -2.1785 | prune below (byte-identical to `../deghost_w9ft/`) |
| `e2c_L3_8.ts` | 3 | 8 | 2 | -2.4514 | prune below (byte-identical to `../deghost_w9ft/`) |
| `e2c_L4_4.ts` | 4 | 4 | 1 | -1.1751 | keep at or above; 18 input columns (15 charge + head score, per-wire log-softmax mean / min) |

`levels.json` is the ladder in the `ml_levels` form `[level, width, k, threshold]`; the L4_4 entry is the exported keep logit
(ghost-fraction-0.2 rule on the D2 arm's CV logits; `tables/keep_rule.json` also records the plateau rule, -2.6956). It is
not an after-clustering operating point: if this chain is adopted, the shipped keep comes from the toolkit's matched
post-clustering ghost-share scan (TLA `deghost_w9ft_fm_keep`). The
standardiser (15 charge + 3 head columns) is baked into `e2c_L4_4.ts`. Head columns: computed on the L4 cells, NaN -> the
graph's column minimum (the doc 30 C++ head stage). Export: torch 2.10.0+cu126; `SHA256SUMS` (from the export) covers the
five `.ts`. Integration, parity and the after-clustering comparison: wcp-porting-img pr_ml doc (FD-HD FM deghoster).
