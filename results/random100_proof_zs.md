# Random-sampling proof (zero-shot, ID = ImageNet-100)

Randomly sample 100 ID images + 100 OOD images (4 sets pooled, seed 0).

| metric | ID mean±std | ID median | OOD mean±std | OOD median | AUROC | p (Mann-Whitney) |
|---|---|---|---|---|---|---|
| JS | 0.399±0.062 | 0.396 | 0.302±0.049 | 0.301 | **89.65** | 3.4e-22 |
| spread | 1.205±0.583 | 1.022 | 0.557±0.120 | 0.522 | **92.18** | 6.7e-25 |
| sd_maxp | 0.106±0.049 | 0.094 | 0.048±0.015 | 0.045 | **91.60** | 2.9e-24 |

AUROC over 20 repeated independent samplings (100v100 each):

| metric | AUROC mean±std | min | max |
|---|---|---|---|
| JS | 89.75±1.38 | 87.21 | 92.62 |
| spread | 88.61±1.92 | 84.48 | 91.51 |
| sd_maxp | 88.26±1.84 | 85.40 | 91.22 |
