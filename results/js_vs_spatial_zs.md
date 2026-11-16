# Global JS vs Spatial JS (zero-shot, ID = ImageNet-100)

| statistic | ID | OOD | AUROC | FPR95 |
|---|---|---|---|---|
| Global JS | 0.4067±0.0683 | 0.3035±0.0464 | 89.83 | 38.26 |
| Spatial JS | 0.0776±0.0191 | 0.0505±0.0116 | 90.03 | 43.52 |

## 1) Correlation (low correlation = prerequisite for complementarity)

| scope | Pearson | Spearman |
|---|---|---|
| pooled (ID+OOD) | +0.727 | +0.638 |
| ID only | +0.708 | +0.685 |
| OOD only | +0.561 | +0.530 |

## 2) Fusion s = z(JS) + w·z(Spatial)

| combination | AUROC | FPR95 |
|---|---|---|
| JS only (w=0) | 89.83 | 38.26 |
| Spatial only (w→∞) | 90.03 | 43.52 |
| **best w = 1.2** | **91.94** | **34.46** |

ΔAUROC(best − JS) = +2.10, ΔFPR95 = -3.80

| w | 0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| AUROC | 89.83 | 90.88 | 91.60 | 91.81 | 91.92 | 91.94 | 91.91 | 91.80 |
| FPR95 | 38.26 | 35.58 | 34.50 | 34.20 | 34.32 | 34.46 | 35.18 | 36.28 |

## 2b) OOD detection rate at a fixed ID FPR = 5%

| rule | detection rate | corresponding FPR95 |
|---|---|---|
| Global JS only | 49.51 | 50.49 |
| Spatial only | 56.23 | 43.77 |
| both hit (intersection) | 37.10 | — |
| either hit (union) | 68.64 | — |

union − best single item = +12.40 percentage points (the actual FPR of the union on ID = 7.84%)

## 2c) Per OOD dataset (w = 1, parameter-free)

| OOD set | JS | Spatial | JS+Spatial (w=1) |
|---|---|---|---|
| iNaturalist | 94.10 | 96.70 | 96.77 |
| SUN | 90.20 | 88.12 | 91.32 |
| places365 | 87.61 | 85.41 | 88.71 |
| Texture | 85.56 | 89.74 | 90.11 |

## 3) Conditional AUROC (with Global JS held fixed, how much information is left in Spatial)

| Global JS decile | n_ID | n_OOD | within-bin AUROC of Spatial |
|---|---|---|---|
| D1 | 500 | 24316 | 77.41 |
| D2 | 500 | 5781 | 73.92 |
| D3 | 500 | 2548 | 71.52 |
| D4 | 500 | 1260 | 66.80 |
| D5 | 500 | 712 | 71.05 |
| D6 | 500 | 429 | 68.10 |
| D7 | 500 | 222 | 63.53 |
| D8 | 500 | 137 | 64.66 |
| D9 | 500 | 85 | 62.80 |

mean within-bin AUROC = 68.86 (min 62.80)

## 4) Flexible two-dimensional discrimination (quadratic discriminant analysis, odd/even images split in half)

QDA two-dimensional AUROC (held-out half) = 90.12
