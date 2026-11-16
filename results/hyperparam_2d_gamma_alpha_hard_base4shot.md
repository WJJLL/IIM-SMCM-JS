# Two-dimensional (γ, α) hyperparameter grid for S-MCM-JS-Spatial

**ID data**: ImageNet-500-Hard (`ImageNet-500-Hard/new_4shots`, `part=new`, base method, 4-shot)  
**OOD data**: iNaturalist / SUN / places365 / Texture (mean over the 4 OOD sets)  
**Seeds**: [1, 2, 3]  
**Spatial grid**: g = 7 (14×14 patches → 7×7 adaptive pooling)  
**Grid step**: γ and α both in [0, 1] with step 0.1 (11 × 11 = 121 configurations)

## Parameter meaning

| symbol | meaning | code |
|---|---|---|
| **γ** | total budget of the auxiliary terms: the combined fluctuation of JS and Spatial on ID is γ·σ_ID(S-MCM) | `JS_GAMMA` |
| **α** | share of the budget taken by the **Spatial term** (JS takes 1−α) | `JS_SPATIAL_A = 1 − α` |

$$\mathrm{coef}_{js}=(1-\alpha)\,\gamma\,\frac{\sigma_{ID}(S_{MCM})}{\sigma_{ID}(JS)}\qquad \mathrm{coef}_{sp}=\alpha\,\gamma\,\frac{\sigma_{ID}(S_{MCM})}{\sigma_{ID}(JS_{std})}$$

$$S = S_{MCM} - \mathrm{coef}_{js}\,(JS-\mu_{JS}) - \mathrm{coef}_{sp}\,(JS_{std}-\mu_{sp})$$

`(coef, μ)` are calibrated on the **ID set** only and never touch any OOD number; smaller scores are more ID-like.

## Speed-up

The image features are **forwarded only once per seed** (ID 25,000 + 4 OOD sets 35,640 = 60,640 images/seed), and the per-sample components were dumped to disk. All 121 × 3 = 363 evaluations in this table were recomputed on CPU; the GPU is not used and the model is not re-run.

## 0. Base (γ = 0, independent of α)

| seed | S-MCM FPR95 | S-MCM AUROC |
|---|---|---|
| 1 | 29.74 | 93.16 |
| 2 | 27.63 | 93.55 |
| 3 | 30.48 | 93.27 |
| **mean±std** | 29.28±1.48 | 93.33±0.20 |

## 1. Full FPR95 grid (%↓, seed mean)

| γ \ α | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| **0.0** | 29.28 | 29.28 | 29.28 | 29.28 | 29.28 | 29.28 | 29.28 | 29.28 | 29.28 | 29.28 | 29.28 |
| **0.1** | 28.18 | 28.08 | 27.99 | 27.89 | 27.84 | 27.78 | 27.77 | 27.72 | 27.72 | 27.66 | 27.67 |
| **0.2** | 27.74 | 27.47 | 27.31 | 27.00 | 26.97 | **26.86** | 26.83 | 26.75 | 26.68 | 26.78 | 26.92 |
| **0.3** | 28.00 | 27.51 | 26.97 | 26.60 | 26.41 | 26.27 | 26.18 | 26.19 | 26.25 | 26.41 | 26.72 |
| **0.4** | 28.65 | 27.84 | 27.15 | 26.50 | 26.09 | 25.94 | 25.86 | 25.89 | 26.17 | 26.46 | 27.01 |
| **0.5** | 29.19 | 28.19 | 27.34 | 26.43 | 25.98 | 25.71 | 25.72 | 25.88 | 26.34 | 26.77 | 27.49 |
| **0.6** | 29.87 | 28.63 | 27.49 | 26.62 | 26.07 | 25.70 | 25.66 | 25.94 | 26.49 | 27.16 | 28.20 |
| **0.7** | 30.76 | 29.26 | 27.93 | 26.96 | 26.26 | 25.88 | 25.81 | 26.27 | 26.82 | 27.58 | 28.77 |
| **0.8** | 31.76 | 29.88 | 28.34 | 27.23 | 26.53 | 26.06 | 26.01 | 26.48 | 27.23 | 28.16 | 29.67 |
| **0.9** | 32.72 | 30.65 | 28.81 | 27.64 | 26.81 | 26.18 | 26.31 | 26.68 | 27.63 | 28.84 | 30.57 |
| **1.0** | 33.66 | 31.46 | 29.41 | 28.08 | 27.12 | 26.59 | 26.58 | 27.05 | 28.17 | 29.46 | 31.47 |

> Bold = online default (γ=0.2, α=0.5) → 26.86; grid optimum (γ=0.6, α=0.6) → **25.66**

## 2. Full AUROC grid (%↑, seed mean)

| γ \ α | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| **0.0** | 93.33 | 93.33 | 93.33 | 93.33 | 93.33 | 93.33 | 93.33 | 93.33 | 93.33 | 93.33 | 93.33 |
| **0.1** | 93.67 | 93.67 | 93.68 | 93.68 | 93.67 | 93.67 | 93.66 | 93.65 | 93.63 | 93.61 | 93.59 |
| **0.2** | 93.84 | 93.87 | 93.89 | 93.91 | 93.91 | **93.90** | 93.87 | 93.84 | 93.80 | 93.74 | 93.67 |
| **0.3** | 93.89 | 93.97 | 94.02 | 94.05 | 94.06 | 94.05 | 94.01 | 93.95 | 93.87 | 93.77 | 93.64 |
| **0.4** | 93.87 | 93.99 | 94.08 | 94.13 | 94.15 | 94.14 | 94.09 | 94.00 | 93.87 | 93.71 | 93.51 |
| **0.5** | 93.78 | 93.96 | 94.09 | 94.17 | 94.20 | 94.18 | 94.12 | 94.00 | 93.82 | 93.60 | 93.32 |
| **0.6** | 93.65 | 93.89 | 94.06 | 94.17 | 94.22 | 94.20 | 94.11 | 93.96 | 93.74 | 93.45 | 93.09 |
| **0.7** | 93.50 | 93.79 | 94.00 | 94.14 | 94.20 | 94.19 | 94.08 | 93.90 | 93.63 | 93.27 | 92.83 |
| **0.8** | 93.33 | 93.67 | 93.93 | 94.10 | 94.17 | 94.15 | 94.03 | 93.81 | 93.49 | 93.07 | 92.55 |
| **0.9** | 93.15 | 93.54 | 93.84 | 94.04 | 94.13 | 94.11 | 93.97 | 93.72 | 93.35 | 92.86 | 92.26 |
| **1.0** | 92.96 | 93.41 | 93.75 | 93.97 | 94.07 | 94.05 | 93.90 | 93.61 | 93.19 | 92.64 | 91.96 |

> Grid AUROC optimum (γ=0.6, α=0.4) → **94.22**

## 3. Marginal curves

### 3.1 Fixed α = 0.5, sweeping γ

| γ | FPR95 | AUROC | ΔFPR vs base |
|---|---|---|---|
| 0.0 | 29.28 | 93.33 | +0.00 |
| 0.1 | 27.78 | 93.67 | -1.50 |
| 0.2 | 26.86 | 93.90 | -2.42 **←** |
| 0.3 | 26.27 | 94.05 | -3.01 |
| 0.4 | 25.94 | 94.14 | -3.34 |
| 0.5 | 25.71 | 94.18 | -3.57 |
| 0.6 | 25.70 | 94.20 | -3.58 |
| 0.7 | 25.88 | 94.19 | -3.40 |
| 0.8 | 26.06 | 94.15 | -3.22 |
| 0.9 | 26.18 | 94.11 | -3.11 |
| 1.0 | 26.59 | 94.05 | -2.69 |

### 3.2 Fixed γ = 0.2, sweeping α

| α (Spatial share) | FPR95 | AUROC |
|---|---|---|
| 0.0 | 27.74 | 93.84 |
| 0.1 | 27.47 | 93.87 |
| 0.2 | 27.31 | 93.89 |
| 0.3 | 27.00 | 93.91 |
| 0.4 | 26.97 | 93.91 |
| 0.5 | 26.86 | 93.90 **←** |
| 0.6 | 26.83 | 93.87 |
| 0.7 | 26.75 | 93.84 |
| 0.8 | 26.68 | 93.80 |
| 0.9 | 26.78 | 93.74 |
| 1.0 | 26.92 | 93.67 |

### 3.3 Fixed γ = 0.6 (best FPR95 band), sweeping α

| α (Spatial share) | FPR95 | AUROC |
|---|---|---|
| 0.0 | 29.87 | 93.65 |
| 0.1 | 28.63 | 93.89 |
| 0.2 | 27.49 | 94.06 |
| 0.3 | 26.62 | 94.17 |
| 0.4 | 26.07 | 94.22 |
| 0.5 | 25.70 | 94.20 |
| 0.6 | 25.66 | 94.11 |
| 0.7 | 25.94 | 93.96 |
| 0.8 | 26.49 | 93.74 |
| 0.9 | 27.16 | 93.45 |
| 1.0 | 28.20 | 93.09 |

## 4. Region no worse than the default configuration

Cells with FPR95 ≤ 26.86 (default γ=0.2, α=0.5) number **48/121**:

| γ | α interval |
|---|---|
| 0.2 | 0.5 – 0.9 |
| 0.3 | 0.3 – 1.0 |
| 0.4 | 0.3 – 0.9 |
| 0.5 | 0.3 – 0.9 |
| 0.6 | 0.3 – 0.8 |
| 0.7 | 0.4 – 0.8 |
| 0.8 | 0.4 – 0.7 |
| 0.9 | 0.4 – 0.7 |
| 1.0 | 0.5 – 0.6 |

## 5. Key observations

1. **Most stable around α = 0.5.** At γ=0.2 the FPR95 for α from 0.3 to 0.7 is 26.75–27.00, almost a flat line; both ends (pure JS or pure Spatial) degrade. Both components are necessary.
   - Closer look: in the γ=0.2 row the FPR95 minimum sits at α=0.8 (26.68), while the AUROC maximum sits at α=0.4–0.5 (93.91). The two preferences conflict slightly; **α=0.5 is the compromise that serves FPR95 and AUROC at the same time**, so the online version takes 0.5.
2. **γ is the main sensitive direction.** With α=0.5 fixed, FPR95 decreases monotonically from 29.28 at γ=0 to 25.70 at γ=0.6, then rises again. The default γ=0.2 gives 26.86.
3. **Pure JS collapses at large γ.** α=0 (pure JS) gives FPR95 = 33.66 at γ=1.0, worse than the base 29.28; α=0.5 at γ=1.0 gives 26.59. This shows that the Spatial term acts to "keep large budgets stable".
4. **The default γ=0.2 is a conservative but defensible point.** It follows from the criterion "auxiliary-term perturbation ≤ 20% σ_ID", which looks at ID only; see the leave-one-OOD-dataset check in `hyperparam_js_spatial_base4shot_hard.md` — a larger γ chosen on OOD does not transfer across datasets.
5. **The robust region is large.** **48** of the 121 cells are no worse than the default point in FPR95, and they concentrate in a contiguous patch over γ∈[0.3, 0.8] × α∈[0.3, 0.9] (not isolated spikes), showing that the method is insensitive to hyperparameters. If space permits in the paper, the main results can be given in two columns, γ=0.2 (ID criterion) and γ=0.5.

