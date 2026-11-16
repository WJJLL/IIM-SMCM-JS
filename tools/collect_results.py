"""Collect all IIM / zero-shot OOD evaluation results into CSV files.

Outputs
-------
output/results/all_results.csv       raw rows: id, cfg, shots, seed, source, score, ood, fpr95, auroc, aupr
output/results/summary_3seed.csv     seed-averaged rows (same columns without seed)
"""
import csv
import os
import re
from collections import defaultdict

import numpy as np

OUT_DIR = "output/results"
SD = "numneg300_topk50_lamda5_div0.5"
OODS = ["iNaturalist", "SUN", "places365", "Texture"]

FPR_RE = re.compile(r"^FPR:([0-9.]+), AUROC:([0-9.]+), AURPC:([0-9.]+)$")
AVG_RE = re.compile(r"^(.+?) score avg\. FPR:([0-9.]+), AUROC:([0-9.]+), AUPR:([0-9.]+)$")
OOD_RE = re.compile(r"^Evaluating OOD dataset (\w+)$")

# (id_data, cfg, shots, part, root, group_dir or "")
GROUPS = [
    ("ImageNet-500-Easy", "common", 4, "base", "/mnt/mydrive/iim/output", f"imagenet/common/base_4shots/{SD}"),
    ("ImageNet-500-Easy", "common", 16, "base", "/mnt/mydrive/iim/output", f"imagenet/common/base_16shots/{SD}"),
    ("ImageNet-500-Easy", "common_divmean", 4, "base", "/mnt/mydrive/iim/output", f"imagenet/common_divmean/base_4shots/{SD}"),
    ("ImageNet-500-Easy", "common_divmean", 16, "base", "/mnt/mydrive/iim/output", f"imagenet/common_divmean/base_16shots/{SD}"),
    ("ImageNet-500-Hard", "ImageNet-500-Hard", 4, "new", "/mnt/mydrive/iim/output", f"imagenet/ImageNet-500-Hard/new_4shots/{SD}"),
    ("ImageNet-500-Hard", "ImageNet-500-Hard", 16, "new", "/mnt/mydrive/iim/output", f"imagenet/ImageNet-500-Hard/new_16shots/{SD}"),
    ("ImageNet-500-Hard", "ImageNet-500-Hard-divmean2", 4, "new", "/mnt/mydrive/iim/output", f"imagenet/ImageNet-500-Hard-divmean2/new_4shots/{SD}"),
    ("ImageNet-500-Hard", "ImageNet-500-Hard-divmean2", 16, "new", "/mnt/mydrive/iim/output", f"imagenet/ImageNet-500-Hard-divmean2/new_16shots/{SD}"),
    ("ImageNet-100", "common", 4, "all", "/mnt/mydrive/iim/output", f"imagenet100/common/all_4shots/{SD}"),
    ("ImageNet-100", "common", 16, "all", "/mnt/mydrive/iim/output", f"imagenet100/common/all_16shots/{SD}"),
    ("ImageNet-100", "common_divmean", 4, "all", "/mnt/mydrive/iim/output", f"imagenet100/common_divmean/all_4shots/{SD}"),
    ("ImageNet-100", "common_divmean", 16, "all", "/mnt/mydrive/iim/output", f"imagenet100/common_divmean/all_16shots/{SD}"),
    ("ImageNet-1K", "common_4shots", 4, "all", "/mnt/mydrive/iim/output_1k/imagenet/IIM", f"common_4shots/{SD}"),
    ("ImageNet-1K", "common_4shots_dm2", 4, "all", "/mnt/mydrive/iim/output_1k/imagenet/IIM", f"common_4shots/{SD}_dm2"),
    ("ImageNet-1K", "common_16shots", 16, "all", "/mnt/mydrive/iim/output_1k/imagenet/IIM", f"common_16shots/{SD}"),
    ("ImageNet-1K", "common_16shots_dm2", 16, "all", "/mnt/mydrive/iim/output_1k/imagenet/IIM", f"common_16shots/{SD}_dm2"),
]

# score name mapping inside eval_smcm_spatial (pure spatial run)
SPATIAL_ALIAS = {
    "S-MCM-JS-Spatial score": "S-MCM-Spatial score",
    "S-MCM-PC-JS-Spatial score": "S-MCM-PC-Spatial score",
}


def parse_per_ood(path):
    """Return {ood: {score: (fpr, auroc, aupr)}}; also avg lines."""
    per = {o: {} for o in OODS}
    avg = {}
    cur, pending = None, None
    if not os.path.exists(path):
        return None, {}
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = OOD_RE.match(line)
            if m:
                cur, pending = m.group(1), None
                continue
            m = AVG_RE.match(line)
            if m:
                avg[m.group(1) + " score"] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
                continue
            if line.endswith(" score"):
                pending = line
                continue
            m = FPR_RE.match(line)
            if m and cur in per and pending is not None:
                per[cur][pending] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                pending = None
    return per, avg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for id_data, cfg, shots, part, root, rel in GROUPS:
        for seed in (1, 2, 3):
            ckpt = os.path.join(root, rel, f"seed{seed}_{part}")
            for source in ("eval_full9", "eval_spatial", "eval_smcm_spatial", "eval"):
                per, avg = parse_per_ood(os.path.join(ckpt, source, "log.txt"))
                if per is None:
                    continue
                for ood in OODS:
                    for score, vals in per[ood].items():
                        name = SPATIAL_ALIAS.get(score, score) if source == "eval_smcm_spatial" else score
                        rows.append((id_data, cfg, shots, seed, source, name, ood, *vals))
                for score, vals in avg.items():
                    name = SPATIAL_ALIAS.get(score, score) if source == "eval_smcm_spatial" else score
                    rows.append((id_data, cfg, shots, seed, source, name, "AVG", *vals))

    header = ["id_data", "cfg", "shots", "seed", "source", "score", "ood", "fpr95", "auroc", "aupr"]
    with open(os.path.join(OUT_DIR, "all_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # seed-averaged summary
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[:1] + r[1:3] + r[4:7]].append(r[7:])
    srows = []
    for key, vals in sorted(buckets.items()):
        arr = np.array(vals, dtype=float)
        srows.append(list(key) + list(arr.mean(axis=0)))
    with open(os.path.join(OUT_DIR, "summary_3seed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c for c in header if c != "seed"])
        w.writerows(srows)

    print(f"wrote {len(rows)} raw rows -> {OUT_DIR}/all_results.csv")
    print(f"wrote {len(srows)} mean rows -> {OUT_DIR}/summary_3seed.csv")


if __name__ == "__main__":
    main()
