# Why Spatial is needed (zero-shot, ID=ImageNet-100)

| score | AUROC | FPR95 | detection rate@ID-FPR5% |
|---|---|---|---|
| Global JS | 89.83 | 38.26 | 49.51 |
| Spatial JS | 90.03 | 43.52 | 56.23 |
| fusion (w=1) | 91.92 | 34.32 | 61.08 |

## Composition of OOD detections at a fixed ID FPR=5%

| category | share of all OOD |
|---|---|
| hit by both | 37.10% |
| **hit by Global JS only** | 12.40% |
| **hit by Spatial only** | 19.13% |
| hit by neither | 31.36% |

Union = 68.64%; of the 50.49% missed by Global, 19.13 percentage points are recovered by Spatial.

## Within-box AUROC of Spatial after controlling for Global JS = 68.86 (min 62.80)
