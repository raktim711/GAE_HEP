# Diagnostics of bb3

## Single observables (jet level, box background)

| signal / region | n sig | n bkg | m | n_const | pt | tau21c | mean m_J sig / bkg |
|---|---|---|---|---|---|---|---|
| bb3/full | 3200 | 200000 | 0.626 | 0.639 | 0.714 | 0.488 | 302 / 224 |
| bb3/window | 924 | 8370 | 0.539 | 0.497 | 0.765 | 0.486 | 283 / 246 |
| bb3_dijet/full | 1200 | 200000 | 0.531 | 0.449 | 0.852 | 0.486 | 249 / 224 |
| bb3_dijet/window | 787 | 8370 | 0.495 | 0.443 | 0.758 | 0.501 | 241 / 246 |
| bb3_trijet/full | 2000 | 200000 | 0.682 | 0.754 | 0.631 | 0.489 | 334 / 224 |
| bb3_trijet/window | 137 | 8370 | 0.790 | 0.810 | 0.806 | 0.402 | 525 / 246 |

## Per run

| run | KS(box bkg, R&D bkg) | frac > R&D q90 | ρ(score, m_J) box | ρ(score, m_J) R&D |
|---|---|---|---|---|
| study_deepsets_s1 | 0.024 | 0.107 | +0.75 | +0.74 |
| study_edgeconv_s1 | 0.023 | 0.105 | +0.73 | +0.71 |
| study_transformer_s1 | 0.021 | 0.105 | +0.70 | +0.69 |
| study_gcn_complete_s1 | 0.023 | 0.097 | +0.66 | +0.66 |
| study_efn_s1 | 0.019 | 0.110 | +0.70 | +0.70 |
| study_ewmp_s1 | 0.020 | 0.107 | +0.71 | +0.71 |
| study_ew_attention_s1 | 0.020 | 0.106 | +0.69 | +0.68 |
| fix_hldensity_s1 | 0.019 | 0.091 | +0.02 | -0.08 |
