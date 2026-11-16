"""Parse LocalPrompt eval logs and print per-seed / mean FPR95 & AUROC tables."""
import glob
import os
import re
import sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "output/localprompt_table/logs"

TAGS = ["imagenet100", "imagenet500easy", "imagenet500hard"]
SHOTS = [4, 16]
SEEDS = [1, 2, 3]
OOD_SETS = ["iNaturalist", "SUN", "places365", "Texture"]

line_fpr = re.compile(r"FPR:([0-9.]+), AUROC:([0-9.]+)")
avg_fpr = re.compile(r"avg\. FPR:([0-9.]+), AUROC:([0-9.]+)")


def parse(path):
    """Return {(ood_name, 'fpr'|'auroc'): value} for the Local-Prompt score, plus avg."""
    res = {}
    cur_set = None
    pending_score = None
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = re.match(r"Evaluating OOD dataset (\w+)", line)
            if m:
                cur_set = m.group(1)
                continue
            if line in ("MCM score", "Local-Prompt score"):
                pending_score = line
                continue
            m = avg_fpr.search(line)
            if m and line.startswith("Local-Prompt score avg."):
                res["avg"] = (float(m.group(1)), float(m.group(2)))
                continue
            m = line_fpr.search(line)
            if m and pending_score == "Local-Prompt score" and cur_set:
                res[cur_set] = (float(m.group(1)), float(m.group(2)))
                pending_score = None
    return res


def main():
    print(f"{'dataset':<16}{'shot':<6}{'seed':<6}{'avg FPR95':>10}{'avg AUROC':>11}")
    summary = {}
    for tag in TAGS:
        for shots in SHOTS:
            per_seed = []
            for seed in SEEDS:
                path = os.path.join(LOG_DIR, f"{tag}_{shots}shot_seed{seed}_eval.log")
                if not os.path.exists(path):
                    print(f"{tag:<16}{shots:<6}{seed:<6}{'MISSING':>10}")
                    continue
                r = parse(path)
                if "avg" not in r:
                    print(f"{tag:<16}{shots:<6}{seed:<6}{'PARSE-ERR':>10}")
                    continue
                fpr, auroc = r["avg"]
                per_seed.append((fpr, auroc))
                print(f"{tag:<16}{shots:<6}{seed:<6}{fpr:>10.2f}{auroc:>11.2f}")
            if per_seed:
                mf = sum(x[0] for x in per_seed) / len(per_seed)
                ma = sum(x[1] for x in per_seed) / len(per_seed)
                summary[(tag, shots)] = (mf, ma)
                print(f"{tag:<16}{shots:<6}{'mean':<6}{mf:>10.2f}{ma:>11.2f}")
                print("-" * 50)

    print("\n=== Seed-averaged (Average column of the paper) ===")
    print(f"{'dataset':<16}{'shot':<6}{'FPR95':>10}{'AUROC':>11}")
    for (tag, shots), (mf, ma) in summary.items():
        print(f"{tag:<16}{shots:<6}{mf:>10.2f}{ma:>11.2f}")


if __name__ == "__main__":
    main()
