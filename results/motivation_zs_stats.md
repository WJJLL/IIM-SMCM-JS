# Zero-shot motivation figure (ID = ImageNet-100)

| set | n | mean H(π_p) | H(q̄) | JS | cell-JS std |
|---|---|---|---|---|---|
| ID | 5000 | 3.667 | 4.074 | 0.407 | 0.0776 |
| iNaturalist | 10000 | 3.894 | 4.185 | 0.291 | 0.0441 |
| SUN | 10000 | 3.824 | 4.125 | 0.301 | 0.0526 |
| places365 | 10000 | 3.821 | 4.133 | 0.312 | 0.0547 |
| Texture | 5640 | 3.716 | 4.031 | 0.315 | 0.0509 |

Single-feature AUROC (4 OOD sets pooled): mean H(π_p) **79.50**, H(q̄) **59.86**, JS **89.83**, cell-JS std **90.03**
