import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_runs(path):
    with open(path, "r", encoding="utf-8") as f:
        runs = json.load(f)
    out = {}
    for r in runs:
        seed = int(r["seed"])
        y = np.array(r["y2022"], dtype=np.float32)
        p = np.array(r["p2022"], dtype=np.float32)
        th = float(r["eval2022"]["threshold"])
        out[seed] = {"y": y, "p": p, "th": th}
    return out


def main():
    parser = argparse.ArgumentParser(description="Build paired probability CSV from two run files.")
    parser.add_argument("--baseline-json", type=str, default="./paper_metrics/baseline_runs.json")
    parser.add_argument("--expert-json", type=str, default="./paper_metrics/expert_runs.json")
    parser.add_argument("--mode", type=str, choices=["seed", "mean"], default="seed",
                        help="seed: output one csv per common seed; mean: output one csv using mean probs across seeds")
    parser.add_argument("--seed", type=int, default=42, help="used when --mode seed")
    parser.add_argument("--out-csv", type=str, default="./paper_metrics/paired_probs_2022.csv")
    parser.add_argument("--out-thresholds", type=str, default="./paper_metrics/paired_thresholds.json")
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_thresholds = Path(args.out_thresholds)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    b = load_runs(args.baseline_json)
    e = load_runs(args.expert_json)

    common = sorted(set(b.keys()) & set(e.keys()))
    if not common:
        raise ValueError("No common seeds found between baseline and expert runs.")

    if args.mode == "seed":
        if args.seed not in common:
            raise ValueError(f"Requested seed={args.seed} not in common seeds: {common}")
        sb = b[args.seed]
        se = e[args.seed]

        if len(sb["y"]) != len(se["y"]):
            raise ValueError("Length mismatch between baseline and expert y arrays.")
        if not np.array_equal(sb["y"], se["y"]):
            raise ValueError("y labels are not row-wise equal between baseline and expert for selected seed.")

        df = pd.DataFrame({
            "y_true_expert": se["y"],
            "prob_expert": se["p"],
            "y_true_baseline": sb["y"],
            "prob_baseline": sb["p"],
        })
        df.to_csv(out_csv, index=False)

        thresholds = {
            "mode": "seed",
            "seed": args.seed,
            "threshold_expert": se["th"],
            "threshold_baseline": sb["th"],
            "common_seeds": common,
        }

    else:
        y_ref = b[common[0]]["y"]
        for s in common:
            if not np.array_equal(y_ref, b[s]["y"]) or not np.array_equal(y_ref, e[s]["y"]):
                raise ValueError("Label arrays are not aligned across seeds; cannot average probs safely.")

        p_b = np.mean(np.stack([b[s]["p"] for s in common], axis=0), axis=0)
        p_e = np.mean(np.stack([e[s]["p"] for s in common], axis=0), axis=0)

        df = pd.DataFrame({
            "y_true_expert": y_ref,
            "prob_expert": p_e,
            "y_true_baseline": y_ref,
            "prob_baseline": p_b,
        })
        df.to_csv(out_csv, index=False)

        thresholds = {
            "mode": "mean",
            "seed": None,
            "threshold_expert": float(np.mean([e[s]["th"] for s in common])),
            "threshold_baseline": float(np.mean([b[s]["th"] for s in common])),
            "common_seeds": common,
        }

    with open(out_thresholds, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    print(f"Saved paired csv: {out_csv}")
    print(f"Saved thresholds: {out_thresholds}")


if __name__ == "__main__":
    main()
