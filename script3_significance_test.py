import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def paired_bootstrap_pvalue(y_true, prob_a, prob_b, metric_fn, n_boot=2000, seed=123):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        pa = prob_a[idx]
        pb = prob_b[idx]
        diffs.append(metric_fn(yt, pa) - metric_fn(yt, pb))
    diffs = np.array(diffs)
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return float(p), float(np.mean(diffs)), float(ci_low), float(ci_high)


def parse_prob_columns(df: pd.DataFrame, prefix: str):
    y_col = f"y_true_{prefix}"
    p_col = f"prob_{prefix}"
    if y_col not in df.columns or p_col not in df.columns:
        raise ValueError(f"Missing columns: {y_col}, {p_col}")
    y = df[y_col].to_numpy(dtype=np.float32)
    p = df[p_col].to_numpy(dtype=np.float32)
    return y, p


def parse_args():
    p = argparse.ArgumentParser(description="Paired bootstrap significance test for two models.")
    p.add_argument("--input-csv", type=str, default="./paper_metrics/paired_probs_2022.csv")
    p.add_argument("--threshold-expert", type=float, default=0.5)
    p.add_argument("--threshold-baseline", type=float, default=0.5)
    p.add_argument("--out-txt", type=str, default="./paper_metrics/significance_report.txt")
    p.add_argument("--n-boot", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    y_e, p_e = parse_prob_columns(df, "expert")
    y_b, p_b = parse_prob_columns(df, "baseline")

    if not np.array_equal(y_e, y_b):
        raise ValueError("y_true_expert and y_true_baseline are not identical row-wise.")

    y = y_e

    # AUC significance
    p_auc, d_auc, lo_auc, hi_auc = paired_bootstrap_pvalue(
        y,
        p_e,
        p_b,
        metric_fn=lambda yt, yp: roc_auc_score(yt, yp),
        n_boot=args.n_boot,
        seed=123,
    )

    # F1 significance (threshold-dependent)
    p_f1, d_f1, lo_f1, hi_f1 = paired_bootstrap_pvalue(
        y,
        p_e,
        p_b,
        metric_fn=lambda yt, yp: f1_score(yt, (yp >= args.threshold_expert).astype(int), zero_division=0),
        n_boot=args.n_boot,
        seed=456,
    )

    # Baseline F1 deltas under its threshold for readability
    f1_e = f1_score(y, (p_e >= args.threshold_expert).astype(int), zero_division=0)
    f1_b = f1_score(y, (p_b >= args.threshold_baseline).astype(int), zero_division=0)
    auc_e = roc_auc_score(y, p_e)
    auc_b = roc_auc_score(y, p_b)

    lines = [
        "Paired Bootstrap Significance Report (2022)",
        "========================================",
        f"N samples: {len(y)}",
        f"Bootstrap rounds: {args.n_boot}",
        "",
        "Point Estimates:",
        f"  Expert   AUC: {auc_e:.6f}",
        f"  Baseline AUC: {auc_b:.6f}",
        f"  Delta AUC (Expert-Baseline): {auc_e - auc_b:.6f}",
        f"  Expert   F1(th={args.threshold_expert:.4f}): {f1_e:.6f}",
        f"  Baseline F1(th={args.threshold_baseline:.4f}): {f1_b:.6f}",
        f"  Delta F1 (Expert-Baseline): {f1_e - f1_b:.6f}",
        "",
        "Paired Bootstrap Tests:",
        f"  AUC  p-value: {p_auc:.6f}",
        f"  AUC  mean delta: {d_auc:.6f}",
        f"  AUC  95% CI of delta: [{lo_auc:.6f}, {hi_auc:.6f}]",
        f"  F1   p-value: {p_f1:.6f}",
        f"  F1   mean delta: {d_f1:.6f}",
        f"  F1   95% CI of delta: [{lo_f1:.6f}, {hi_f1:.6f}]",
        "",
        "Interpretation rule:",
        "  If p-value < 0.05, the improvement is statistically significant.",
    ]

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved significance report to: {args.out_txt}")


if __name__ == "__main__":
    main()
