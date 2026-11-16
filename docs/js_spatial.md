# Global-JS and Spatial-JS score post-processing

This document describes the two **test-time** score terms added on top of IIM:
a global patch-consensus term (`*-JS`) and a spatial-organisation term
(`*-JS-Spatial`).

> **The whole thing is post-processing.** No retraining, no weight change, no
> change to the training objective or the data pipeline. Only
> `IIM.test_ood` / `IIM.test_ood_imagenet` (i.e. the numbers written to
> `log.txt` during evaluation) are affected. Training with `train.py` is
> bit-for-bit identical to the plain IIM baseline.

## Scores

For every base score `b` in `{MCM, GL-MCM, R-MCM, S-MCM}` the evaluator reports
three numbers, 12 in total:

| name | definition |
|---|---|
| `b` | the original IIM score |
| `b-JS` | `b - coef * (JS - mu_js)` |
| `b-JS-Spatial` | `b - coef_g * (JS - mu_js) - coef_s * (JS_s - mu_s)` |

with patch logits `z_p` (`p = 1..196`, already divided by 100), `pi_p = softmax(tau * z_p)`
and `q = mean_p pi_p`:

$$
JS = H(q) - \tfrac{1}{M}\sum_p H(\pi_p)
\quad\text{(per image, larger = more ID-like)}
$$

$$
JS_s = \operatorname{std}_{g\times g}\Big(H(q_{\text{cell}}) - \overline{H(\pi)}_{\text{cell}}\Big)
$$

`JS` is computed on all patch tokens; `JS_s` first pools the patch distributions
into a `g x g` grid by adaptive average pooling and then takes the standard
deviation of the per-cell JS values.

## ID-only calibration

`coef`, `coef_g`, `coef_s`, `mu_*` are fitted **once, on the in-distribution
pass only** and reused for every OOD set:

$$
\text{coef} = \gamma\,\frac{\operatorname{std}_{ID}(b)}{\operatorname{std}_{ID}(JS)},
\qquad
\text{coef}_g = \alpha\,\gamma\,\frac{\operatorname{std}_{ID}(b)}{\operatorname{std}_{ID}(JS)},
\qquad
\text{coef}_s = (1-\alpha)\,\gamma\,\frac{\operatorname{std}_{ID}(b)}{\operatorname{std}_{ID}(JS_s)}
$$

so the auxiliary terms may move the ID score distribution by at most
`gamma * std_ID(b)` in total. No OOD sample ever enters the coefficients; the
OOD passes simply reuse `trainer._js_calib`.

## Defaults

| knob | default | meaning |
|---|---|---|
| `JS_GAMMA` (`gamma`) | **0.5** | budget of the auxiliary terms, in units of `std_ID(base)` |
| `JS_SPATIAL_A` (`alpha`) | **0.5** | share of the budget for the global `JS` term (`1 - alpha` for `JS_s`) |
| `JS_SPATIAL_GRID` (`g`) | 7 | grid size of the spatial term |
| `JS_SCALE` (`tau`) | 50 | softmax temperature on patch logits (= 0.5 on raw cos logits) |
| `JS_SPATIAL_W` | `""` | explicit split of the budget, e.g. `"1,0"` (global only), `"0,1"` (spatial only) |

They live in `configs/trainers/IIM/*.yaml` under `TRAINER.IIM`, can be
overridden on the command line, and still fall back to the environment variables
`JS_GAMMA / JS_SPATIAL_A / JS_SPATIAL_GRID / JS_SCALE / JS_SPATIAL_W` used by
earlier experiment scripts.

## How to run

Nothing changes in the pipeline -- evaluate as usual:

```bash
sh scripts/iim/eval.sh imagenet common 50 new \
   output/imagenet/IIM/common_4shots/.../seed1_new 300 1.0
```

Optional overrides:

```bash
python eval_ood_detection.py ... \
  --js_gamma 0.5 --js_spatial_a 0.5 --js_spatial_grid 7 \
  # or, for ablation,
  --js_spatial_w "1,0"   # global JS only
  --js_spatial_w "0,1"   # spatial JS only
```

The ID pass prints the calibration:

```
[IIM S-MCM-JS] ID-only calibration: mu_js=... std_js=... std_base=... coef=...
```

If an OOD pass is run without a preceding ID pass, `*-JS` and `*-JS-Spatial`
degrade to the base score and a warning is printed (they are never calibrated
on OOD data).

## Implementation

* `trainers/utils/js_spatial.py` -- `Patch_Agreement_Score`,
  `Patch_Spatial_Std_Score`, `calibrate_js_spatial`, `combine_js_spatial`,
  `configure`.
* `trainers/iim.py` -- `_base_and_aux_scores` + `_eval_ood_loader`, shared by
  `test_ood` and `test_ood_imagenet`; `Info()` returns the 12 score names.
* `trainers/zeroshot.py` -- the same definitions for the frozen zero-shot CLIP
  baseline (identical helpers, identical calibration).
* `eval_ood_detection.py` -- passes `calibrate=True` on the ID pass only.
