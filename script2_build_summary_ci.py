import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def bootstrap_ci_metric(y_true, y_prob, threshold, metric_name, n_boot=1000, seed=123):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if metric_name == "auc":
            vals.append(roc_auc_score(yt, yp))
        elif metric_name == "f1_pos":
            pred = (yp >= threshold).astype(int)
            vals.append(f1_score(yt, pred, zero_division=0))
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize_csv(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "auc", "pr_auc", "precision_pos", "recall_pos", "f1_pos",
        "bal_acc", "mcc", "brier", "ece", "threshold"
    ]
    rows = []
    for m, g in df.groupby("model"):
        row = {"model": m, "n_seeds": int(g["seed"].nunique())}
        for c in metric_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def load_probs_from_runs_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        runs = json.load(f)

    all_y = []
    all_p = []
    all_th = []
    for r in runs:
        y = np.array(r["y2022"], dtype=np.float32)
        p = np.array(r["p2022"], dtype=np.float32)
        th = float(r["eval2022"]["threshold"])
        all_y.append(y)
        all_p.append(p)
        all_th.append(th)
    return all_y, all_p, all_th


def aggregate_ci_from_seed_runs(y_list, p_list, th_list, n_boot=1000, seed=123):
    auc_l, auc_u = [], []
    f1_l, f1_u = [], []
    for i, (y, p, th) in enumerate(zip(y_list, p_list, th_list)):
        lo, hi = bootstrap_ci_metric(y, p, th, "auc", n_boot=n_boot, seed=seed + i)
        auc_l.append(lo)
        auc_u.append(hi)
        lo, hi = bootstrap_ci_metric(y, p, th, "f1_pos", n_boot=n_boot, seed=seed + 100 + i)
        f1_l.append(lo)
        f1_u.append(hi)
    return {
        "auc_ci_low_mean": float(np.mean(auc_l)),
        "auc_ci_high_mean": float(np.mean(auc_u)),
        "f1_ci_low_mean": float(np.mean(f1_l)),
        "f1_ci_high_mean": float(np.mean(f1_u)),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Build mean/std summary and bootstrap CI from per-seed results.")
    p.add_argument("--input-csv", type=str, default="./paper_metrics/metrics_per_seed_all.csv")
    p.add_argument("--baseline-json", type=str, default="./paper_metrics/baseline_runs.json")
    p.add_argument("--expert-json", type=str, default="./paper_metrics/expert_runs.json")
    p.add_argument("--out-csv", type=str, default="./paper_metrics/summary_mean_std_ci.csv")
    p.add_argument("--n-boot", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    df = pd.read_csv(args.input_csv)
    summary = summarize_csv(df)

    ci_rows = []
    if os.path.exists(args.baseline_json):
        yb, pb, thb = load_probs_from_runs_json(args.baseline_json)
        ci_b = aggregate_ci_from_seed_runs(yb, pb, thb, n_boot=args.n_boot, seed=123)
        ci_rows.append({"model": "baseline", **ci_b})

    if os.path.exists(args.expert_json):
        ye, pe, the = load_probs_from_runs_json(args.expert_json)
        ci_e = aggregate_ci_from_seed_runs(ye, pe, the, n_boot=args.n_boot, seed=456)
        ci_rows.append({"model": "expert", **ci_e})

    ci_df = pd.DataFrame(ci_rows)
    if len(ci_df) > 0:
        summary = summary.merge(ci_df, on="model", how="left")

    summary.to_csv(args.out_csv, index=False)
    print(f"Saved summary with CI to: {args.out_csv}")


if __name__ == "__main__":
    main()
