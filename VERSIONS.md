# Version log

> ## ⚠️ Important: the few-shot split was replaced from 2026-11-11 01:00
>
> The split used for the release (`data/ood/imagenet/split_fewshot/shot_*/seed_*.txt`) was
> replaced at **11-11 01:00** by a version **100.00% identical** to the remote
> `/root/autodl-tmp/IIM-SMCM/data/ood/imagenet/split_fewshot/` (all six files, shot 4/16 × seed 1/2/3, were checked one by one).
>
> **Numbers before 11-11 (the w5 / v6 sections of this file) use the old split and are not comparable with v7.**
> Typical discrepancy: for ImageNet-500-Hard 4-shot, `S-MCM` is **27.51** on the old split
> and **29.29** on the new split (+1.78 FPR). Comparing across splits leads to completely different conclusions,
> so the paper may only cite the v7 (new split) numbers.

Dataset/protocol fixed: ImageNet-500-Hard, 4-shot, ViT-B/16, 50 epoch, evaluation resolution 224.
All metrics are the mean over 3 seeds × 4 OOD sets (iNaturalist / SUN / places365 / Texture),
and the paired statistics are same-seed pairings (first average each seed over the 4 sets, then take mean ± sd over the 3 seeds).

| Version | Training | Evaluation score definition |
|---|---|---|
| w0 | `DIVMEAN_W=0` (original IIM) | double exponential |
| w4 | `DIVMEAN_W=4` | double exponential |
| **w5** | **`DIVMEAN_W=4`** | **single exponential** |
| **v7** | **same as w5** | **single exponential + training-free spatial term** |

---

## v7 — 2026-11-14 (current version, new split)

4 datasets × {4,16}-shot × {base(w0), divmean(w4)} = **16 configurations**, 3 seeds each,
all re-run on the release split with `eval_full9` / `eval_spatial` (9 scores output in one pass).

### Main finding: the training-free score enhancement improves consistently on 16/16 configurations

Score formula (no training, no weight change, only the ranking at evaluation time):

$$S = \underbrace{S_{\text{S-MCM-PC}}}_{\text{base}}
      -\gamma\Big[a\cdot\widehat{JS_g} + (1-a)\cdot\widehat{JS_s}\Big],
\quad \gamma=0.2,\ a=0.5,\ g=7$$

$\widehat{\cdot}$ denotes standardization with the mean/std of the **ID validation set** (note: `eval_ood_detection.py:188`
uses `set_val_loader`, i.e. the ID val/test split, **not** the training set — this must be stated correctly in the paper,
and it must be acknowledged as transductive).

| Dataset | shots | Training | S-MCM | +JS | +Spatial | total ΔFPR |
|---|---|---|---|---|---|---|
| 1k | 4 | base | 23.46 | −1.39 | −1.10 | **−2.48** |
| 1k | 4 | divmean | 22.91 | −1.43 | −0.79 | **−2.21** |
| 1k | 16 | base | 21.77 | −1.69 | −0.52 | **−2.21** |
| 1k | 16 | divmean | 21.03 | −1.63 | −0.41 | **−2.04** |
| Hard | 4 | base | 29.29 | −1.53 | −0.89 | **−2.42** |
| Hard | 4 | divmean | 30.56 | −1.18 | −0.86 | **−2.03** |
| Hard | 16 | base | 27.08 | −1.44 | −0.96 | **−2.40** |
| Hard | 16 | divmean | 25.89 | −1.61 | −0.40 | **−2.01** |
| 100 | 4 | base | 11.42 | −1.09 | −0.94 | **−2.03** |
| 100 | 4 | divmean | 12.32 | −0.41 | −1.05 | **−1.46** |
| 100 | 16 | base | 9.64 | −0.78 | −0.63 | **−1.41** |
| 100 | 16 | divmean | 10.37 | −0.86 | −0.58 | **−1.44** |
| Easy | 4 | base | 7.30 | −0.12 | −0.46 | **−0.58** |
| Easy | 4 | divmean | 7.74 | −0.18 | −0.25 | **−0.42** |
| Easy | 16 | base | 6.48 | −0.31 | −0.27 | **−0.58** |
| Easy | 16 | divmean | 7.07 | −0.27 | −0.12 | **−0.39** |

AUROC also rises on all 16/16 configurations. **This is the core claim the paper can make**:
it does not depend on any training change, so it holds under any training recipe.

### The conclusion for the div_mean training change is weakened: only stably positive on 1k

Change of `S-MCM` from base to divmean under the new split (same-seed pairing):

| Dataset | 4-shot | 16-shot |
|---|---|---|
| ImageNet-1k | **−0.55** (better) | **−0.74** (better) |
| ImageNet-500-Hard | +1.27 (worse) | −1.19 (better) |
| ImageNet-500-Easy | +0.44 (worse) | +0.59 (worse) |
| ImageNet-100 | +0.90 (worse) | +0.73 (worse) |

7 configurations get worse, 4 better, 1 essentially flat. **Do not make div_mean the main contribution**;
at most write it as "helps at the ImageNet-1k scale". The "w4 improves stably" seen on the old split
(the w5 section of this file) does not reproduce on the new split.

### Full four tables (FPR95 / AUROC)

See `python tools/report_final_tables.py` (reads `eval_full9` / `eval_spatial` from `/mnt/mydrive/iim/output{,_1k}`),
which supports `--only {1k,500,100,delta,all}`.

The 9 scores of ImageNet-1k 4-shot base (new split, the most authoritative set):

| score | FPR95 | AUROC |
|---|---|---|
| MCM | 41.86 | 91.00 |
| GL-MCM | 30.80 | 92.51 |
| R-MCM | 28.42 | 93.49 |
| S-MCM | 23.46 | 94.76 |
| S-MCM-PC | 23.42 | 94.81 |
| S-MCM-PC-JS | 22.62 | 95.06 |
| S-MCM-JS | 22.08 | 95.15 |
| S-MCM-PC-JS-Spatial | 21.21 | 95.24 |
| **S-MCM-JS-Spatial** | **20.98** | **95.25** |

### Motivation statistics for the Introduction (500-Hard, w4, 3 seed)

`python tools/report_motivation.py --prefix w4` (pure CPU, reads `/tmp/{sj,jt}_w4_s*.npz`)

| Quantity | Value | Use |
|---|---|---|
| AUROC S-MCM | 94.21 ± 0.24 | baseline |
| AUROC $JS_g$ single feature | 85.23 ± 0.77 | **weaker** than the baseline; cannot claim "stronger" |
| AUROC $JS_s$ single feature | 81.30 ± 0.66 | |
| AUROC $\bar H$ | 81.93 ± 1.18 | |
| AUROC R (direction alignment) | 65.58 ± 0.88 | `alignunit_*` with sharpness contamination removed |
| $\mathrm{Spearman}(S_{\text{MCM}}, JS_g)$ on ID | **−0.306 ± 0.003** | direct evidence of complementarity |
| $\mathrm{Spearman}(JS_g, JS_s)$ on ID | **−0.015 ± 0.005** | the spatial and global terms are **nearly orthogonal** |
| $\text{partial\_corr}(R, JS_g \mid \bar H)$ | **−0.964 ± 0.001** | JS is not a stand-in for sharpness |
| $JS_g$ mean ID / OOD | 1.546 / 1.301 | determines the sign of the subtraction |
| all 4 OOD sets × 3 seed satisfy ID > OOD | **3/3** ✓ | the sign direction was not cherry-picked |

Rescue rate (among the OOD samples missed by S-MCM at 95% TPR, the fraction recovered by the full method):
seed1 **8.2%**, seed2 **8.9%**, seed3 **9.7%**, i.e. the miss rate drops from ~24% to ~16%.

### Same analysis reproduced on ImageNet-1k (w0 base 4-shot, seed1)

`python tools/report_motivation.py --prefix 1k_w0 --seeds 1`
(dumps: `tools/dump_js_mechanism.py` + `dump_spatial_js_v2.py`, each with `--cfg common --part all`)

| Quantity | 1k | 500-Hard | Consistent? |
|---|---|---|---|
| S-MCM AUROC / FPR95 | 94.89 / 23.38 | 94.21 / 25.66 | — |
| $JS_g$ single-feature AUROC | 80.64 | 85.23 | both weaker than the baseline ✓ |
| $JS_s$ single-feature AUROC | 82.18 | 81.30 | ✓ |
| $\bar H$ single-feature AUROC | 82.97 | 81.93 | ✓ |
| S-MCM → +$JS_g$ → +$JS_s$ | 23.38 → 22.22 → **20.87** | 25.66 → 24.63 → **23.58** | monotone improvement ✓ |
| total ΔFPR / ΔAUROC | **−2.51 / +0.50** | −2.08 / +0.55 | end-to-end measured −2.48 ✓ |
| $\mathrm{Spearman}(JS_g, JS_s)$ on ID | **−0.001** | −0.015 | orthogonal ✓ |
| $\mathrm{Spearman}(S_{\text{MCM}}, JS_g)$ on ID | −0.073 | −0.306 | neither is redundant ✓ |
| $\text{partial\_corr}(R, JS_g \mid \bar H)$ | −0.961 | −0.964 | ✓ |
| $JS_g$ mean ID / OOD | 1.470 / 1.275 | 1.546 / 1.301 | ID is higher ✓ |
| all 4 OOD sets satisfy ID > OOD | ✓ | ✓ | ✓ |
| rescue rate | **11.3%** | 8.2–9.7% | more pronounced on 1k |

**All three claims hold on 1k, with a higher rescue rate. Using the 1k set for the motivation figure in the main text is recommended.**

Qualitative figures: `python tools/report_heatmap.py --npz /tmp/hm_w4_s1.npz --outdir figures`
→ `figures/figure_motivation_{dist,maps,heatmap}.png`.
The 7×7 map of a single sample is noisy; **in the main text use `dist` (quantitative) + `maps` (set average)**,
and put the per-image triplets in the appendix.

**Overclaim already ruled out**: one cannot say "the patch consistency of ID has object-centred structure".
The centre-border contrast `center − border` has a mean AUROC over the 4 OOD sets of only **56.65**,
and places365 has a larger contrast (0.070) than ID (0.065), while Texture is negative (−0.037).
The 81.47 of $\sigma(JS_c)$ is the statistic that actually works.

### Implementation points (four things the method section of the paper must state clearly)

1. **$\tau = 2$**. The `logit_scale` raw of CLIP ViT-B/16 is 4.6052 = $\ln 100$,
   which after `exp()` is exactly 100.00000762939453. In evaluation the local logits are first divided by 100 and then multiplied by
   `JS_SCALE=50`, so the softmax temperature inside $JS$ is exactly **2**.
2. **Calibration uses the ID validation set**, not the training set (see above).
3. **$g = 7$**: the 14×14 patches are adaptively pooled into 7×7 regions, 2×2 patches per cell.
4. **`std` is the sample standard deviation (unbiased, $n-1$)**. In `Patch_Spatial_Std_Score`
   `Tensor.std()` defaults to `unbiased=True`, which on the same image differs from `unbiased=False` by
   $\sqrt{49/48}$, but since the normalization uses the ID std afterwards, this factor cancels out.

Also: `S_MCM_PC_Score` itself must be defined earlier in the method section (it is $S_{\text{S-MCM}}$ plus a
"consistency between patches and the global consensus" term), otherwise undefined symbols appear in the formulas.

---

## w5 — 2026-11-10 (**old split, no longer comparable**)

### Definition

Building on w4, the **double exponential** in the denominator of `R-MCM` / `S-MCM-PC` is changed to a **single exponential**:

$$p_{(p,c)}=\frac{e^{z_{pc}/T}}{\sum_{c'}e^{z_{pc'}/T}+\sum_{j}e^{n_{pj}/T}}$$

The old form `exp`s the logit once and then `exp`s the whole thing again ($e^{\,e^{z/T}/T}$), which is an implementation bug:
it compresses the relative differences of the logits twice, making the ranking differ from a standard softmax (the measured
rank correlation between the two forms is only **0.833**).

### Code changes

- `trainers/utils/score_function.py`
  - adds `_pair_ratio(output_local, neg_output_local, T)`, **shared** by `R_MCM_Score` and
    `S_MCM_PC_Score`, guaranteeing a strictly identical denominator formula.
  - adds the `R_MCM_EXP` switch: default `single` (w5); setting it to `double` reproduces
    the old w4 numbers bit-exactly (verified).
- The `DIVMEAN_W` default is already `4`, so w5 is **training + evaluation under the default configuration**.

### Main results (FPR% ↓ / AUROC% ↑)

| score | w0 (single exp) | w4 (single exp) | w4 − w0 (same-seed paired) |
|---|---|---|---|
| MCM | 48.60 / 88.72 | 48.60 / 88.72 | 0.00 / 0.00 (self-check: should be identically 0) |
| GL-MCM | 39.23 / 89.71 | 38.51 / 90.15 | −0.72 / +0.44 |
| R-MCM | 35.66 / 91.28 | 32.38 / 92.41 | −3.28 / +1.13 |
| S-MCM | 27.51 / 93.61 | 26.03 / 94.04 | −1.48 / +0.43 |
| S-MCM-PC | 27.20 / 93.75 | 25.67 / 94.21 | −1.53 / +0.47 |
| **S-MCM-PC-JS** | **26.68 / 94.14** | **24.64 / 94.60** | **−2.04 ± 0.37 / +0.46 ± 0.06** |

Paired statistics for `S-MCM-PC-JS`: ΔFPR $t=-9.68$, ΔAUROC $t=12.55$, same sign in 3/3 seed,
improvement on all 4/4 OOD sets (iNat −2.41/+0.36, SUN −0.72/+0.33, places −1.10/+0.21,
Texture −3.94/+0.93).

ID classification accuracy: w0 = 69.87%, w4 = 69.97% (essentially unchanged).

### Difference from w4 (double exponential)

| score | on w0 (single − double) | on w4 (single − double) |
|---|---|---|
| R-MCM | +0.91 FPR / −0.14 AUROC | +0.39 FPR / −0.02 AUROC |
| S-MCM | −0.00 / +0.07 | −0.14 / +0.11 |
| S-MCM-PC | −0.07 / +0.06 | −0.19 / +0.09 |
| S-MCM-PC-JS | −0.07 / +0.07 | −0.27 / +0.09 |

Conclusion: switching to the single exponential is **not** done to gain points. It is slightly unfavourable to pure R-MCM (the "bug" in the old form happened to help
R-MCM slightly), but has almost no effect on the final score — because the glcm term of `S_MCM_Score` is already
single-exponential and dominates the magnitude ($\lambda\cdot$glcm ≈ 0.14, while the R-MCM sub-term is only ≈ 0.01).
The net effect is that the w4 − w0 effect size is larger with lower variance (ΔAUROC +0.43 ± 0.07 → **+0.46 ± 0.06**;
ΔFPR −1.84 ± 0.57 → **−2.04 ± 0.37**).

### Reproduction

```bash
PY=/home/star/miniconda3/envs/dassl/bin/python
BASE=output/imagenet/IIM/ImageNet-500-Hard_4shots/numneg300_topk50_lamda5_div0.5_mm0.0_beta10.0

# training (w5 = default DIVMEAN_W=4)
SEEDS=$S RUN_TAG=_divmean_w4 PYTHON=$PY \
  sh scripts/iim/train.sh imagenet ImageNet-500-Hard 4 5 0.5 50 new 300 0.0 10.0 max

# evaluation (single exponential = R_MCM_EXP not set)
sh scripts/iim/eval.sh imagenet ImageNet-500-Hard 50 new ${BASE}_divmean_w4/seed${S}_new 300 1.0 256

# reproduce the old (double exponential) w4 numbers
R_MCM_EXP=double sh scripts/iim/eval.sh imagenet ImageNet-500-Hard 50 new <MODEL_DIR> 300 1.0 256
```

Summary script: `python tools/report_single_exp.py` (single/double exponential comparison, reads `/tmp/eval_single_*.log`).

---

# v6 (candidate, **old split, numbers not comparable**): S-MCM-PC-JS-Spatial

## Score formula

One more **spatial organization** statistic is added on top of S-MCM-PC-JS; the two terms share the same gamma budget:

```
js       = H(mean_p q_p) - mean_p H(q_p)                  # global patch divergence
js_std   = std_g( JS(cell_g) ),  g=7, 14x14 -> 7x7 adaptive pooling
S        = base - coef_js *(js    - mu_js )
                - coef_std*(js_std - mu_std)
coef_js  = a     * GAMMA * std_ID(base) / std_ID(js)        a = 0.5
coef_std = (1-a) * GAMMA * std_ID(base) / std_ID(js_std)    GAMMA = 0.2
```

Intuition: the patch consensus of ID concentrates in a few spatial regions -> large inter-cell variance; OOD is diffuse -> small variance.
Implementation: `Patch_Spatial_Std_Score` / `calibrate_js_spatial` / `combine_js_spatial`
(`trainers/utils/score_function.py`); evaluation automatically outputs
`S-MCM-PC-JS-Spatial` and `S-MCM-JS-Spatial`. `JS_SPATIAL_A` and
`JS_SPATIAL_GRID` can override.

## Results (ImageNet-500-Hard 4-shot, mean over the 4 OOD sets, 3 seed, same code)

| score | w0 FPR95 / AUROC | w4 FPR95 / AUROC |
|---|---|---|
| MCM | 48.60 / 88.72 | 48.60 / 88.72 |
| S-MCM-PC | 27.20 / 93.75 | 25.67 / 94.21 |
| S-MCM-PC-JS | 26.68 / 94.14 | 24.64 / 94.60 |
| **S-MCM-PC-JS-Spatial** | **25.47 / 94.21** | **23.80 / 94.62** |
| S-MCM-JS-Spatial | 25.34 / 94.15 | 23.97 / 94.52 |

Gain from the spatial term (relative to the JS version on the same base):

| | dFPR | dAUROC | per-seed dFPR |
|---|---|---|---|
| w0 | **-1.209 ± 0.150** | **+0.069 ± 0.008** | -1.31 / -1.00 / -1.32 |
| w4 | -0.837 ± 0.045 | +0.020 ± 0.029 (ns) | -0.78 / -0.89 / -0.84 |

**Conclusion**: 3/3 seed stably reduce FPR95; but there is no significant AUROC gain on w4 (there is on w0).
The gain is larger on w0 (-1.21 vs -0.84), showing that **W4 and the spatial term are partly redundant** — W4 has already improved the local
evidence structure, so less headroom is left for the spatial term. This differs from the JS term (JS gains the same on w0/w4 and is orthogonal).

The offline proxy `tools/probe_score_terms.py` reproduces the end-to-end numbers exactly (predicts 23.80/94.62,
measured 23.80/94.62), so it can be used for candidate screening without a GPU.

## Closed directions (all measured with 3 seed)

| Direction | Result | Tool |
|---|---|---|
| `JS_SCALE` 50→20 | end-to-end **worse** +0.38~+0.54 FPR (the automatic ID calibration of coef cancels the scale) | `tools/probe_js_upgrades.py` |
| entropy-decoupled JS (`js_res`) | end-to-end 24.67/94.57 ≈ a tie, slightly worse; single feature 0.72 | `tools/probe_score_terms.py` |
| using only hbar / neffmix / dom as the second term | all worse (+0.96 / +2.82 / +1.82 FPR) | same as above |
| Tail-JS (`TopMean_k(d_p)`, Q95) | **none beats mean(=JS)**; the best, tail30, gives -0.077 FPR / -0.056 AUROC | `tools/probe_tail_js.py` |
| multi-temperature rank averaging, the `js/hbar` ratio | all worse than single-temperature js | `tools/probe_js_upgrades.py` |

Mechanism side: `partial_corr(R, js | hbar) = -0.923` (R is the pure direction alignment of patches away from uniform),
but after removing entropy the single feature is only 0.72 -> **the power of JS comes from the joint "entropy × coherence", each path alone is weaker**
(hbar 0.82, R 0.66, js 0.87). See `tools/dump_js_mechanism.py`,
`tools/report_patch_alignment.py`.

### Directions additionally closed in v7 (2026-11-14)

The spatial-aggregation and calibration lines have been exhausted; **the current `std(JS_c)` is saturated** and no alternative aggregation is stronger.

| Direction | Result |
|---|---|
| 27 cross-cell aggregations (std / min-k / max-k / tv / ctr × pooling avg·max × g=2,4,7,10) | none stably beats `std` (`tools/probe_spatial_sweep.py` / `probe_spatial_v2.py`) |
| grid g=10 / g=12 | per-seed sign flips, \|Δ\| ≤ 0.04, no material difference |
| max pooling within a cell (instead of avg) | **worse** +0.65~0.69 FPR (all g) |
| multi-scale stacking (g2 + g7) | +0.28 FPR |
| calibration variants (mad / winsor / rank) | mad and std are **bit-identical**; rank −0.09 |
| retuning $\gamma$ 0.2→0.5 | naively −0.90, but **LOOCV only −0.36**, and the per-OOD-set optima conflict with each other (iNat 0.6 / Texture 0.4 / SUN 0.2 / places 0.2) → that is tuning on the test set |
| the `PC` term (patch consistency) | slightly worse on Easy (6.37→6.49 / 7.22→7.47), a tie on Hard, a tie on 1k (23.33→23.38, 20.79→20.76) |
| `DIVMEAN_DIR=rev` (push pos away from the mean neg) | **+4.45 FPR on Hard** (per seed +4.99/+4.23/+4.12), while ID acc actually rises +0.9 → the training objective diverges from the detection objective |
| OOD-patch entropy maximization (w=1.0) | **+13.80 FPR** on im100, ID acc +1.15 |
| OOD-patch entropy maximization (w=0.3) | S-MCM 29.71 vs 26.03 baseline on Hard |
| ID-patch negative-entropy loss (correct selector, w=1.0) | **+2.15 FPR**, ID acc 73.4→69.0 |
| ID-patch negative-entropy loss (wrong selector, w=1.0) | +0.62 FPR, ID acc 72.7 |
| centre-border contrast (`ctr`) as a spatial term | 4-OOD AUROC only 56.65, places365 has a larger contrast than ID → **cannot** serve as evidence for "object-centred structure" |

Mechanism side: `partial_corr(R, js | hbar) = -0.923` (R is the pure direction alignment of patches away from uniform),
but after removing entropy the single feature is only 0.72 -> **the power of JS comes from the joint "entropy × coherence", each path alone is weaker**
(hbar 0.82, R 0.66, js 0.87). See `tools/dump_js_mechanism.py`,
`tools/report_patch_alignment.py`.
