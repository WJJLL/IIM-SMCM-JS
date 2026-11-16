# base 4-shot: JS / Spatial hyperparameter analysis (tag=`.hard_w0`, seeds=[1, 2, 3], mean over the 4 OOD sets)

## 0. Reproducing the base (offline vs online)

| seed | S-MCM FPR95 | S-MCM AUROC |
|---|---|---|
| 1 | 29.74 | 93.16 |
| 2 | 27.63 | 93.55 |
| 3 | 30.48 | 93.27 |
| **mean±std** | 29.28±1.48 | 93.33±0.20 |

## 1. γ sweep (a=0.5, g=7 fixed)

| γ | FPR95 | AUROC |
|---|---|---|
| 0.0 | 29.28±1.48 | 93.33±0.20 |
| 0.05 | 28.37±1.43 | 93.51±0.20 |
| 0.1 | 27.78±1.50 | 93.67±0.21 |
| 0.15 | 27.27±1.48 | 93.79±0.21 |
| 0.2 | 26.86±1.39 | 93.90±0.22 **←** |
| 0.25 | 26.58±1.38 | 93.98±0.23 |
| 0.3 | 26.27±1.44 | 94.05±0.23 |
| 0.4 | 25.94±1.54 | 94.14±0.25 |
| 0.5 | 25.71±1.57 | 94.18±0.26 |
| 0.75 | 25.95±1.57 | 94.17±0.30 |
| 1.0 | 26.59±1.93 | 94.05±0.34 |

## 2. a sweep (γ=0.2, g=7 fixed)

| a (JS share) | FPR95 | AUROC |
|---|---|---|
| 0.0 | 26.92±1.70 | 93.67±0.23 |
| 0.1 | 26.78±1.66 | 93.74±0.23 |
| 0.2 | 26.68±1.58 | 93.80±0.22 |
| 0.3 | 26.75±1.45 | 93.84±0.22 |
| 0.4 | 26.83±1.42 | 93.87±0.22 |
| 0.5 | 26.86±1.39 | 93.90±0.22 **←** |
| 0.6 | 26.97±1.33 | 93.91±0.22 |
| 0.7 | 27.00±1.36 | 93.91±0.22 |
| 0.8 | 27.31±1.26 | 93.89±0.22 |
| 0.9 | 27.47±1.26 | 93.87±0.23 |
| 1.0 | 27.74±1.19 | 93.84±0.23 |

## 3. Spatial-grid g sweep (γ=0.2, a=0.5 fixed)

| g | FPR95 | AUROC |
|---|---|---|
| 4 | 27.49±1.43 | 93.81±0.22 |
| 5 | 27.32±1.43 | 93.85±0.21 |
| 7 | 26.86±1.39 | 93.90±0.22 **←** |
| 10 | 26.80±1.41 | 93.95±0.21 |

## 4. Two-dimensional γ × a (FPR95)

| γ \ a | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|----------|----------|----------|----------|----------|
| **0.05** | 28.31 | 28.34 | 28.37 | 28.44 | 28.55 |
| **0.1** | 27.67 | 27.73 | 27.78 | 27.96 | 28.18 |
| **0.2** | 26.92 | 26.67 | 26.86 | 27.10 | 27.74 |
| **0.3** | 26.72 | 26.14 | 26.27 | 26.82 | 28.00 |
| **0.5** | 27.49 | 26.09 | 25.71 | 26.87 | 29.19 |

## 5. Choosing γ by leave-one-OOD-dataset (showing that γ is not tuned on OOD)

| held-out OOD dataset | γ* chosen on the other 3 sets | FPR95 of that set at γ* | FPR95 of that set at γ=0 |
|---|---|---|---|
| iNaturalist | 0.5 | 9.11 | 12.55 |
| SUN | 0.75 | 28.14 | 27.30 |
| places365 | 0.75 | 35.59 | 34.43 |
| Texture | 0.3 | 35.42 | 42.84 |

## 6. Per-OOD breakdown (γ sweep, a=0.5, g=7)

| γ | iNaturalist | SUN | places365 | Texture | AUROC (4 sets) |
|---|--------------|--------------|--------------|--------------|---|
| 0.0 | 12.55 | 27.30 | 34.43 | 42.84 | 93.33 |
| 0.1 | 11.26 | 26.45 | 33.75 | 39.66 | 93.67 |
| 0.2 | 10.45 | 26.21 | 33.58 | 37.19 | 93.90 |
| 0.3 | 9.82 | 26.27 | 33.57 | 35.42 | 94.05 |
| 0.5 | 9.11 | 26.85 | 34.08 | 32.81 | 94.18 |
| 1.0 | 9.11 | 29.80 | 37.28 | 30.17 | 94.05 |

## 7. The two components separately / combined (γ=0.2, g=7)

| configuration | FPR95 | AUROC | ΔFPR vs base |
|---|---|---|---|
| base S-MCM | 29.28 | 93.33 |  |
| JS only (a=1) | 27.74 | 93.84 | -1.54 |
| Spatial only (a=0) | 26.92 | 93.67 | -2.36 |
| JS+Spatial (a=0.5) | 26.86 | 93.90 | -2.42 |

## 8. Honest LOOCV estimate

For each held-out set, γ* is chosen on the other 3 sets and then measured on that set:

| held-out set | γ* | FPR95@γ* on that set | FPR95@γ=0 on that set |
|---|---|---|---|
| iNaturalist | 0.5 | 9.11 | 12.55 |
| SUN | 0.75 | 28.14 | 27.30 |
| places365 | 0.75 | 35.59 | 34.43 |
| Texture | 0.3 | 35.42 | 42.84 |
| **mean** | — | **27.06** | **29.28** |

- The honest LOOCV estimate is **27.06**; fixing γ=0.2 gives **26.86**; fixing γ=0.5 gives **25.71** (but this one was chosen after looking at OOD).
- Findings: **a γ chosen on OOD does not transfer reliably** (SUN / places365 actually get worse at γ*>0.2), so setting γ from ID statistics only (=0.2) is the more stable and more defensible choice.
