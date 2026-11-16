# Intra-Image Mining and Symmetric Maximum Concept Matching for Few-Shot Out-of-Distribution Detection

Few-shot out-of-distribution (OOD) detection with a frozen CLIP backbone. The
detector learns class-specific local prompts plus a set of negative prompts, and
scores a test image by matching its global feature against frozen global prompts
and its local (patch) features against the learned local/negative prompts.

This release contains the training code, the evaluation code and the
configuration for the IIM baseline, the LocalPrompt baseline and the zero-shot
CLIP baseline, together with two optional **test-time score post-processing**
terms (global JS / spatial JS, see [below](#global-js--spatial-js-score-post-processing)).

## Quick Start

### Environment

The environment used for every number in this repo is pinned in
[`environment.yml`](environment.yml) (Python 3.13, torch 2.11 + CUDA 13.0):

```bash
conda env create -f environment.yml
conda activate iim-smcm
pip install -e Dassl
```

A curated alternative is [`requirements.txt`](requirements.txt):

```bash
conda create -n iim python=3.13
conda activate iim
pip install -r requirements.txt
```

### Install Dassl

`Dassl` is vendored in this repository (a lightly modified fork):

```bash
cd Dassl
pip install -r requirements.txt
python setup.py develop
```

### Data Preparation

Create a data root, for example `/data/ood`, and download the in-distribution
(ID) and out-of-distribution (OOD) datasets there.

#### In-distribution datasets

ImageNet-1K is the main ID dataset:

- create `imagenet/` under the data root and `images/` under it;
- download the dataset from the [official website](https://image-net.org/index.php)
  and extract the training and validation sets to `$DATA/imagenet/images`;
- put `classnames.txt` under `imagenet/data/`. The file used here can be obtained
  from [this link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view).

The smaller ID variants used in the paper (ImageNet100, ImageNet10, ImageNet20)
follow the same procedure; see [MCM](https://github.com/deeplearning-wisc/MCM)
for the detailed category names.

#### Out-of-distribution datasets

We use [iNaturalist](https://arxiv.org/abs/1707.06642),
[SUN](https://vision.princeton.edu/projects/2010/SUN/),
[Places](https://arxiv.org/abs/1610.02055) and
[Texture](https://arxiv.org/abs/1311.3618). The subsampled versions can be
downloaded following [this repository](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset).

Expected layout:

```
$DATA/ood
|-- imagenet
|   |-- classnames.txt
|   |-- train/            # 1,000 folders such as n01440764, n01443537, ...
|   `-- val/
|-- imagenet100
|   |-- classnames.txt
|   |-- train/
|   `-- val/
|-- imagenet10
|   |-- classnames.txt
|   |-- train/
|   `-- val/
|-- imagenet20
|   |-- classnames.txt
|   |-- train/
|   `-- val/
|-- iNaturalist
|-- SUN
|-- Places
`-- Texture
```

### Training

The training entry point is `scripts/iim/train.sh`; adjust the parameters and the
data path inside the script, then run

```bash
sh scripts/iim/train_all.sh
```

to launch every training task. The LocalPrompt and zero-shot baselines have
their own scripts under `scripts/localprompt/` and `scripts/zero-shot/`.

> Note: with ImageNet-1k as the ID dataset a GPU with more than 24 GB of memory
> is required (an H100 was used for the reported runs).

### Inference

The evaluation entry point is `scripts/iim/eval_all.sh`; adjust the parameters
and the data path inside `scripts/iim/eval.sh`, then run

```bash
sh scripts/iim/eval_all.sh
```

## Global-JS / Spatial-JS score post-processing

Evaluation additionally reports two **test-time** variants of every base score:
a global patch-consensus term (`*-JS`) and a spatial-organisation term
(`*-JS-Spatial`). `Info()` therefore returns 12 scores:

```
MCM / GL-MCM / R-MCM / S-MCM  x  {base, -JS, -JS-Spatial}
```

$$
S = b - \gamma\Big[\alpha\,\widehat{JS_g} + (1-\alpha)\,\widehat{JS_s}\Big],
\qquad \gamma = 0.5,\ \alpha = 0.5,\ g = 7
$$

where $\widehat{\cdot}$ standardises with **ID-set** mean/std only, so the
defaults are `JS_GAMMA: 0.5`, `JS_SPATIAL_A: 0.5`, `JS_SPATIAL_GRID: 7` in
`configs/trainers/IIM/*.yaml`.

> ⚠️ This is pure post-processing on the score: no retraining, no weight
> change, no change to the training loss or the data pipeline. Only the numbers
> written during evaluation change; `train.py` is identical to the plain IIM
> baseline.

See [`docs/js_spatial.md`](docs/js_spatial.md) for the definitions, the ID-only
calibration rule and the ablation switches (`--js_spatial_w "1,0"` / `"0,1"`).

## Repository layout

```
train.py                         IIM / LocalPrompt training
eval_ood_detection.py            OOD evaluation (12 scores with JS post-processing)
eval_ood_detection_zero_shot.py  zero-shot CLIP OOD evaluation
clip_w_local/                    CLIP with local (patch) feature output
trainers/                        IIM, LocalPrompt and ZeroShot trainers
trainers/utils/js_spatial.py     global-JS / spatial-JS score terms
configs/datasets/                dataset configs
configs/trainers/                IIM / LocalPrompt / ZeroShot configs
scripts/                         training and evaluation shell scripts
docs/                            notes (JS/spatial definitions, experiments)
results/                         experiment reports and result tables
figures/                         figures used in the reports
Dassl/                           vendored Dassl framework
```

## Acknowledgement

This repo is heavily borrowed from [CoOp](https://github.com/KaiyangZhou/CoOp),
[LoCoOp](https://github.com/AtsuMiyai/LoCoOp) and
[Local-Prompt](https://github.com/AuroraZengfh/Local-Prompt).
Thanks for these great works and for open-sourcing them.

## Contact

Questions can be raised through the issue tracker of this repository.
