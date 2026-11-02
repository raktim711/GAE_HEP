# Diagnostics of bb1

## Single observables (jet level, box background)

| signal / region | n sig | n bkg | m | n_const | pt | tau21c | mean m_J sig / bkg |
|---|---|---|---|---|---|---|---|
| bb1/full | 834 | 200000 | 0.819 | 0.529 | 0.811 | 0.184 | 464 / 217 |
| bb1/window | 512 | 16177 | 0.843 | 0.539 | 0.701 | 0.143 | 523 / 249 |

## Per run

| run | KS(box bkg, R&D bkg) | frac > R&D q90 | ρ(score, m_J) box | ρ(score, m_J) R&D |
|---|---|---|---|---|
| study_deepsets_s1 | 0.036 | 0.129 | +0.76 | +0.74 |
| study_edgeconv_s1 | 0.039 | 0.129 | +0.73 | +0.71 |
| study_transformer_s1 | 0.035 | 0.124 | +0.71 | +0.69 |
| study_gcn_complete_s1 | 0.044 | 0.138 | +0.68 | +0.66 |
| study_efn_s1 | 0.034 | 0.124 | +0.72 | +0.70 |
| study_ewmp_s1 | 0.035 | 0.125 | +0.73 | +0.71 |
| study_ew_attention_s1 | 0.035 | 0.124 | +0.70 | +0.68 |
| fix_hldensity_s1 | 0.118 | 0.172 | -0.08 | -0.08 |
