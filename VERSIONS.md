# 版本记录

> ## ⚠️ 重要：2026-11-11 01:00 起 few-shot split 已替换
>
> 发布用的 split（`data/ood/imagenet/split_fewshot/shot_*/seed_*.txt`）在
> **11-11 01:00** 被换成了与远端 `/root/autodl-tmp/IIM-SMCM/data/ood/imagenet/split_fewshot/`
> **100.00% 一致**的版本（shot 4/16 × seed 1/2/3 六个文件逐一核对过）。
>
> **11-11 之前的数字（本文件 w5 / v6 两节）用的是旧 split，与 v7 不可比。**
> 典型差异：ImageNet-500-Hard 4-shot 的 `S-MCM` 旧 split 为 **27.51**，
> 新 split 为 **29.29**（+1.78 FPR）。跨 split 比较会得出完全不同的结论，
> 论文里只能引用 v7（新 split）的数字。

数据集/协议固定：ImageNet-500-Hard，4-shot，ViT-B/16，50 epoch，评测分辨率 224。
所有指标为 3 个 seed × 4 个 OOD 集（iNaturalist / SUN / places365 / Texture）的均值，
配对统计为同 seed 配对（先每 seed 在 4 集上平均，再对 3 seed 求 mean ± sd）。

| 版本 | 训练 | 评测分数定义 |
|---|---|---|
| w0 | `DIVMEAN_W=0`（原始 IIM） | 双指数 |
| w4 | `DIVMEAN_W=4` | 双指数 |
| **w5** | **`DIVMEAN_W=4`** | **单指数** |
| **v7** | **同 w5** | **单指数 + 免训练空间分项** |

---

## v7 — 2026-11-14（当前版本，新 split）

4 个数据集 × {4,16}-shot × {base(w0), divmean(w4)} = **16 个配置**，每个 3 seed，
统一在发布 split 上重跑 `eval_full9` / `eval_spatial`（9 条分数一次输出）。

### 主结论：免训练评分增强在 16/16 个配置上一致改善

打分式（不训练、不改权重，只在评测时改排序）：

$$S = \underbrace{S_{\text{S-MCM-PC}}}_{\text{base}}
      -\gamma\Big[a\cdot\widehat{JS_g} + (1-a)\cdot\widehat{JS_s}\Big],
\quad \gamma=0.2,\ a=0.5,\ g=7$$

$\widehat{\cdot}$ 表示用 **ID 验证集**的均值/标准差标准化（注意：`eval_ood_detection.py:188`
用的是 `set_val_loader`，即 ID 的 val/test 划分，**不是**训练集——论文里必须写对，
而且要承认这是 transductive 的）。

| 数据集 | shots | 训练 | S-MCM | +JS | +Spatial | 合计 ΔFPR |
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

AUROC 在 16/16 个配置上也全部上升。**这是论文可以主张的核心结论**：
它不依赖训练改动，因此在任何训练口径下都成立。

### div_mean 训练改动的结论被削弱：只在 1k 上稳定为正

新 split 下 `S-MCM` 的 base→divmean 变化（同 seed 配对）：

| 数据集 | 4-shot | 16-shot |
|---|---|---|
| ImageNet-1k | **−0.55**（好） | **−0.74**（好） |
| ImageNet-500-Hard | +1.27（差） | −1.19（好） |
| ImageNet-500-Easy | +0.44（差） | +0.59（差） |
| ImageNet-100 | +0.90（差） | +0.73（差） |

7 个配置变差、4 个变好、1 个基本持平。**建议不要把 div_mean 作为主贡献**，
最多写成"在 ImageNet-1k 规模上有帮助"。旧 split 上看到的"w4 稳定改善"
（本文件 w5 节）在新 split 上不复现。

### 完整四表（FPR95 / AUROC）

见 `python tools/report_final_tables.py`（读 `/mnt/mydrive/iim/output{,_1k}` 的
`eval_full9` / `eval_spatial`），支持 `--only {1k,500,100,delta,all}`。

ImageNet-1k 4-shot base 的 9 条分数（新 split，最权威的一组）：

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

### Introduction 的动机统计（500-Hard，w4，3 seed）

`python tools/report_motivation.py --prefix w4`（纯 CPU，读 `/tmp/{sj,jt}_w4_s*.npz`）

| 量 | 值 | 用途 |
|---|---|---|
| AUROC S-MCM | 94.21 ± 0.24 | 基线 |
| AUROC $JS_g$ 单特征 | 85.23 ± 0.77 | 比基线**弱**，不能说"更强" |
| AUROC $JS_s$ 单特征 | 81.30 ± 0.66 | |
| AUROC $\bar H$ | 81.93 ± 1.18 | |
| AUROC R（方向对齐） | 65.58 ± 0.88 | 已剔除锐度污染的 `alignunit_*` |
| $\mathrm{Spearman}(S_{\text{MCM}}, JS_g)$ on ID | **−0.306 ± 0.003** | 互补的直接证据 |
| $\mathrm{Spearman}(JS_g, JS_s)$ on ID | **−0.015 ± 0.005** | 空间项与全局项**近乎正交** |
| $\text{partial\_corr}(R, JS_g \mid \bar H)$ | **−0.964 ± 0.001** | JS 不是锐度的替身 |
| $JS_g$ 均值 ID / OOD | 1.546 / 1.301 | 决定减号方向 |
| 4 个 OOD 集 × 3 seed 全部 ID > OOD | **3/3** ✓ | 符号方向不是挑出来的 |

救援率（95% TPR 下 S-MCM 漏掉的 OOD 中，被完整方法救回的比例）：
seed1 **8.2%**、seed2 **8.9%**、seed3 **9.7%**，即漏报率从 ~24% 降到 ~16%。

### 同上，在 ImageNet-1k 上复现（w0 base 4-shot，seed1）

`python tools/report_motivation.py --prefix 1k_w0 --seeds 1`
（dump：`tools/dump_js_mechanism.py` + `dump_spatial_js_v2.py`，各加 `--cfg common --part all`）

| 量 | 1k | 500-Hard | 结论是否一致 |
|---|---|---|---|
| S-MCM AUROC / FPR95 | 94.89 / 23.38 | 94.21 / 25.66 | — |
| $JS_g$ 单特征 AUROC | 80.64 | 85.23 | 都比基线弱 ✓ |
| $JS_s$ 单特征 AUROC | 82.18 | 81.30 | ✓ |
| $\bar H$ 单特征 AUROC | 82.97 | 81.93 | ✓ |
| S-MCM → +$JS_g$ → +$JS_s$ | 23.38 → 22.22 → **20.87** | 25.66 → 24.63 → **23.58** | 单调改善 ✓ |
| 合计 ΔFPR / ΔAUROC | **−2.51 / +0.50** | −2.08 / +0.55 | 端到端实测 −2.48 ✓ |
| $\mathrm{Spearman}(JS_g, JS_s)$ on ID | **−0.001** | −0.015 | 正交 ✓ |
| $\mathrm{Spearman}(S_{\text{MCM}}, JS_g)$ on ID | −0.073 | −0.306 | 都不冗余 ✓ |
| $\text{partial\_corr}(R, JS_g \mid \bar H)$ | −0.961 | −0.964 | ✓ |
| $JS_g$ 均值 ID / OOD | 1.470 / 1.275 | 1.546 / 1.301 | ID 更高 ✓ |
| 4 个 OOD 集全部 ID > OOD | ✓ | ✓ | ✓ |
| 救援率 | **11.3%** | 8.2–9.7% | 1k 上更明显 |

**1k 上三条 claim 全部成立，且救援率更高。建议正文动机图用 1k 这一组。**

定性图：`python tools/report_heatmap.py --npz /tmp/hm_w4_s1.npz --outdir figures`
→ `figures/figure_motivation_{dist,maps,heatmap}.png`。
单张样例的 7×7 图噪声大，**正文建议用 `dist`（定量）+ `maps`（集合平均）**，
逐图三联放附录。

**已排除的过度宣称**：不能说"ID 的 patch 一致性有物体中心结构"。
中心-边缘对比度 `center − border` 的 4-OOD 平均 AUROC 只有 **56.65**，
而且 places365 的对比度（0.070）比 ID（0.065）还大、Texture 为负（−0.037）。
$\sigma(JS_c)$ 的 81.47 才是有效统计量。

### 实现要点（论文方法节必须写清的四处）

1. **$\tau = 2$**。CLIP ViT-B/16 的 `logit_scale` raw = 4.6052 = $\ln 100$，
   `exp()` 后恰为 100.00000762939453。local logits 在评测里先除 100 再乘
   `JS_SCALE=50`，所以 $JS$ 里的 softmax 温度精确是 **2**。
2. **标定用 ID 验证集**，不是训练集（见上）。
3. **$g = 7$**：14×14 patch 自适应池化成 7×7 区域，每格 2×2 个 patch。
4. **`std` 是样本标准差（unbiased, $n-1$）**。`Patch_Spatial_Std_Score` 里
   `Tensor.std()` 默认 `unbiased=True`，同一张图上与 `unbiased=False` 会差
   $\sqrt{49/48}$，但因为后续按 ID 的 std 归一化，这个因子会被约掉。

另：`S_MCM_PC_Score` 本身要先在方法节定义（它是 $S_{\text{S-MCM}}$ 加一个
"patch 与全局共识的一致性"项），否则式子里出现未定义符号。

---

## w5 — 2026-11-10（**旧 split，已不可比**）

### 定义

在 w4 的基础上，把 `R-MCM` / `S-MCM-PC` 分母的**双重指数**改为**单次指数**：

$$p_{(p,c)}=\frac{e^{z_{pc}/T}}{\sum_{c'}e^{z_{pc'}/T}+\sum_{j}e^{n_{pj}/T}}$$

旧写法把 logit 先 `exp` 一次再整体 `exp` 一次（$e^{\,e^{z/T}/T}$），属于实现错误：
它把 logit 的相对差异压缩了两次，导致排序与标准 softmax 不同（实测两种写法的
排序相关系数仅 **0.833**）。

### 代码改动

- `trainers/utils/score_function.py`
  - 新增 `_pair_ratio(output_local, neg_output_local, T)`：`R_MCM_Score` 与
    `S_MCM_PC_Score` **共用**该函数，保证分母写法严格一致。
  - 新增开关 `R_MCM_EXP`：默认 `single`（w5）；设为 `double` 可 bit-exact 复现
    w4 的旧数字（已验证）。
- `DIVMEAN_W` 默认值已是 `4`，因此 w5 就是**默认配置下的训练 + 评测**。

### 主结果（FPR% ↓ / AUROC% ↑）

| score | w0（单指数） | w4（单指数） | w4 − w0（同 seed 配对） |
|---|---|---|---|
| MCM | 48.60 / 88.72 | 48.60 / 88.72 | 0.00 / 0.00（自检：应恒为 0） |
| GL-MCM | 39.23 / 89.71 | 38.51 / 90.15 | −0.72 / +0.44 |
| R-MCM | 35.66 / 91.28 | 32.38 / 92.41 | −3.28 / +1.13 |
| S-MCM | 27.51 / 93.61 | 26.03 / 94.04 | −1.48 / +0.43 |
| S-MCM-PC | 27.20 / 93.75 | 25.67 / 94.21 | −1.53 / +0.47 |
| **S-MCM-PC-JS** | **26.68 / 94.14** | **24.64 / 94.60** | **−2.04 ± 0.37 / +0.46 ± 0.06** |

`S-MCM-PC-JS` 的配对统计：ΔFPR $t=-9.68$，ΔAUROC $t=12.55$，3/3 seed 同号，
4/4 OOD 集全部改善（iNat −2.41/+0.36，SUN −0.72/+0.33，places −1.10/+0.21，
Texture −3.94/+0.93）。

ID 分类准确率：w0 = 69.87%，w4 = 69.97%（基本不变）。

### 与 w4（双指数）的差异

| score | w0 上（单−双） | w4 上（单−双） |
|---|---|---|
| R-MCM | +0.91 FPR / −0.14 AUROC | +0.39 FPR / −0.02 AUROC |
| S-MCM | −0.00 / +0.07 | −0.14 / +0.11 |
| S-MCM-PC | −0.07 / +0.06 | −0.19 / +0.09 |
| S-MCM-PC-JS | −0.07 / +0.07 | −0.27 / +0.09 |

结论：改单指数**不是**为了涨点。它对纯 R-MCM 略微不利（旧写法的"错误"恰好对
R-MCM 稍有利），但对最终分数几乎无影响 —— 因为 `S_MCM_Score` 的 glcm 项本来就是
单指数且量级占主导（$\lambda\cdot$glcm ≈ 0.14，而 R-MCM 子项仅 ≈ 0.01）。
净效果是 w4 − w0 的效应量更大、方差更小（ΔAUROC +0.43 ± 0.07 → **+0.46 ± 0.06**；
ΔFPR −1.84 ± 0.57 → **−2.04 ± 0.37**）。

### 复现

```bash
PY=/home/star/miniconda3/envs/dassl/bin/python
BASE=output/imagenet/IIM/ImageNet-500-Hard_4shots/numneg300_topk50_lamda5_div0.5_mm0.0_beta10.0

# 训练（w5 = 默认 DIVMEAN_W=4）
SEEDS=$S RUN_TAG=_divmean_w4 PYTHON=$PY \
  sh scripts/iim/train.sh imagenet ImageNet-500-Hard 4 5 0.5 50 new 300 0.0 10.0 max

# 评测（单指数 = 不设置 R_MCM_EXP）
sh scripts/iim/eval.sh imagenet ImageNet-500-Hard 50 new ${BASE}_divmean_w4/seed${S}_new 300 1.0 256

# 复现 w4 的旧（双指数）数字
R_MCM_EXP=double sh scripts/iim/eval.sh imagenet ImageNet-500-Hard 50 new <MODEL_DIR> 300 1.0 256
```

汇总脚本：`python tools/report_single_exp.py`（单/双指数对照，读 `/tmp/eval_single_*.log`）。

---

# v6（候选，**旧 split，数字不可比**）：S-MCM-PC-JS-Spatial

## 打分式

在 S-MCM-PC-JS 上再加一项**空间组织**统计量，两项共享同一 gamma 预算：

```
js       = H(mean_p q_p) - mean_p H(q_p)                  # 全局 patch 分歧度
js_std   = std_g( JS(cell_g) ),  g=7, 14x14 -> 7x7 自适应池化
S        = base - coef_js *(js    - mu_js )
                - coef_std*(js_std - mu_std)
coef_js  = a     * GAMMA * std_ID(base) / std_ID(js)        a = 0.5
coef_std = (1-a) * GAMMA * std_ID(base) / std_ID(js_std)    GAMMA = 0.2
```

直觉：ID 的 patch 共识集中成几个空间区域 -> 格子间差异大；OOD 弥散 -> 差异小。
实现：`Patch_Spatial_Std_Score` / `calibrate_js_spatial` / `combine_js_spatial`
（`trainers/utils/score_function.py`），评测自动输出
`S-MCM-PC-JS-Spatial` 与 `S-MCM-JS-Spatial`。可用 `JS_SPATIAL_A`、
`JS_SPATIAL_GRID` 覆盖。

## 结果（ImageNet-500-Hard 4-shot，4 OOD 平均，3 seed，同代码）

| score | w0 FPR95 / AUROC | w4 FPR95 / AUROC |
|---|---|---|
| MCM | 48.60 / 88.72 | 48.60 / 88.72 |
| S-MCM-PC | 27.20 / 93.75 | 25.67 / 94.21 |
| S-MCM-PC-JS | 26.68 / 94.14 | 24.64 / 94.60 |
| **S-MCM-PC-JS-Spatial** | **25.47 / 94.21** | **23.80 / 94.62** |
| S-MCM-JS-Spatial | 25.34 / 94.15 | 23.97 / 94.52 |

空间项增益（相对同基座 JS 版）：

| | dFPR | dAUROC | 逐 seed dFPR |
|---|---|---|---|
| w0 | **-1.209 ± 0.150** | **+0.069 ± 0.008** | -1.31 / -1.00 / -1.32 |
| w4 | -0.837 ± 0.045 | +0.020 ± 0.029 (ns) | -0.78 / -0.89 / -0.84 |

**结论**：3/3 seed 稳定降低 FPR95；但 w4 上 AUROC 无显著增益（w0 上有）。
增益在 w0 更大（-1.21 vs -0.84），说明 **W4 与空间项部分冗余**——W4 已改善局部
证据结构，留给空间项的余量变小。这与 JS 项不同（JS 在 w0/w4 上增益相同、正交）。

离线代理 `tools/probe_score_terms.py` 精确复现了端到端数字（预测 23.80/94.62，
实测 23.80/94.62），可用于无 GPU 的候选筛选。

## 已关闭的方向（均有 3 seed 实测）

| 方向 | 结果 | 工具 |
|---|---|---|
| `JS_SCALE` 50→20 | 端到端**变差** +0.38~+0.54 FPR（coef 的 ID 自动标定抵消尺度） | `tools/probe_js_upgrades.py` |
| 熵解耦 JS（`js_res`） | 端到端 24.67/94.57 ≈ 持平略差；单特征 0.72 | `tools/probe_score_terms.py` |
| 只用 hbar / neffmix / dom 作第二项 | 全部变差（+0.96 / +2.82 / +1.82 FPR） | 同上 |
| Tail-JS（`TopMean_k(d_p)`、Q95） | **无一项超过 mean(=JS)**，最好的 tail30 为 -0.077 FPR / -0.056 AUROC | `tools/probe_tail_js.py` |
| 多温度秩平均、`js/hbar` 比值 | 均劣于单温度 js | `tools/probe_js_upgrades.py` |

机制侧：`partial_corr(R, js | hbar) = -0.923`（R 为 patch 偏离均匀的纯方向对齐度），
但剥掉熵后单特征只剩 0.72 -> **JS 的力量来自"熵 × 相干性"的联合，单路更弱**
（hbar 0.82、R 0.66、js 0.87）。见 `tools/dump_js_mechanism.py`、
`tools/report_patch_alignment.py`。

### v7 追加关闭的方向（2026-11-14）

空间聚合与标定两条线已被穷举，**当前 `std(JS_c)` 已饱和**，没有任何替代聚合更强。

| 方向 | 结果 |
|---|---|
| 27 种跨 cell 聚合（std / min-k / max-k / tv / ctr × 池化 avg·max × g=2,4,7,10） | 没有一项稳定超过 `std`（`tools/probe_spatial_sweep.py` / `probe_spatial_v2.py`） |
| 网格 g=10 / g=12 | 逐 seed 符号翻转，\|Δ\| ≤ 0.04，无实质差异 |
| cell 内 max 池化（替代 avg） | **变差** +0.65~0.69 FPR（所有 g） |
| 多尺度堆叠（g2 + g7） | +0.28 FPR |
| 标定变体（mad / winsor / rank） | mad 与 std **bit-identical**；rank −0.09 |
| $\gamma$ 重调 0.2→0.5 | 朴素看 −0.90，但 **LOOCV 只有 −0.36**，且各 OOD 集最优值互相冲突（iNat 0.6 / Texture 0.4 / SUN 0.2 / places 0.2）→ 属于在测试集上调参 |
| `PC` 项（patch 一致性） | Easy 上略差（6.37→6.49 / 7.22→7.47），Hard 持平，1k 上 23.33→23.38、20.79→20.76 持平 |
| `DIVMEAN_DIR=rev`（让 pos 远离 mean neg） | **Hard 上 +4.45 FPR**（逐 seed +4.99/+4.23/+4.12），ID acc 反而 +0.9 → 训练目标与检测目标背离 |
| OOD-patch 熵最大化（w=1.0） | im100 上 **+13.80 FPR**，ID acc +1.15 |
| OOD-patch 熵最大化（w=0.3） | Hard 上 S-MCM 29.71 vs 26.03 基线 |
| ID-patch 负熵损失（正确 selector，w=1.0） | **+2.15 FPR**，ID acc 73.4→69.0 |
| ID-patch 负熵损失（错误 selector，w=1.0） | +0.62 FPR，ID acc 72.7 |
| 中心-边缘对比度（`ctr`）作为空间项 | 4-OOD AUROC 仅 56.65，places365 的对比度比 ID 还大 → **不能**作为"物体中心结构"的证据 |

机制侧：`partial_corr(R, js | hbar) = -0.923`（R 为 patch 偏离均匀的纯方向对齐度），
但剥掉熵后单特征只剩 0.72 -> **JS 的力量来自"熵 × 相干性"的联合，单路更弱**
（hbar 0.82、R 0.66、js 0.87）。见 `tools/dump_js_mechanism.py`、
`tools/report_patch_alignment.py`。
