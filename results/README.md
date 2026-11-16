# Result reports and notes

Provenance note for the Markdown reports in this folder and for
[`../VERSIONS.md`](../VERSIONS.md):

* They are the **experiment log** of the research workspace, produced with the
  full research branch, which additionally contains the `S-MCM-PC` base score
  and the `eval_full9` / `eval_spatial` evaluation entry points.
* Those reports therefore quote `S-MCM-PC` as the base score and some of them
  use `gamma = 0.2`, which was the value used for the earlier ablation runs.
* **This repository's default configuration is `gamma = 0.5`, `alpha = 0.5`,
  `g = 7`** (see `configs/trainers/IIM/*.yaml` and
  [`../docs/js_spatial.md`](../docs/js_spatial.md)), applied as test-time
  post-processing on top of the four base scores
  `MCM / GL-MCM / R-MCM / S-MCM`.
* All numbers in these reports were obtained without any retraining: the JS and
  spatial-JS terms only re-score existing base scores at evaluation time.

Raw per-run logs are not committed; they live under `output/<dataset>/IIM/...`
and are excluded by `.gitignore`.
