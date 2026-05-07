import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


MODEL_NAME_MAP: Dict[str, str] = {
    "text_mlp": "MLP-Text",
    "fin_mlp": "MLP-Fin",
    "finp0_mlp": "MLP-Fin+P0",
    "logreg_fin": "LR-Fin",
    "rf_fin": "RF-Fin",
    "hgb_fin": "HGB-Fin",
    "baseline": "MLP-Concat",
    "expert": "EGA",
    "marketmirror": "MarketMirror",
}

MODEL_FAMILY_MAP: Dict[str, str] = {
    "text_mlp": "text-only",
    "fin_mlp": "financial-only",
    "finp0_mlp": "financial-only",
    "logreg_fin": "financial-only",
    "rf_fin": "financial-only",
    "hgb_fin": "financial-only",
    "baseline": "fusion",
    "expert": "expert-gated",
    "marketmirror": "dynamics",
}

METRIC_COLS: List[str] = [
    "auc",
    "pr_auc",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "bal_acc",
    "mcc",
    "brier",
    "ece",
    "threshold",
]


def parse_args():
    p = argparse.ArgumentParser(description="Merge per-seed results and build rankings.")
    p.add_argument("--metrics-dir", type=str, default="./paper_metrics")
    p.add_argument("--setting", type=str, default="submission")
    p.add_argument(
        "--inputs",
        type=str,
        nargs="*",
        default=[
            "metrics_per_seed_core_baselines.csv",
            "metrics_per_seed_baseline.csv",
            "metrics_per_seed_expert.csv",
        ],
        help="CSV files inside metrics-dir to merge.",
    )
    p.add_argument("--output-prefix", type=str, default="leaderboard")
    return p.parse_args()


def load_existing_csvs(metrics_dir: str, inputs: List[str]) -> List[pd.DataFrame]:
    dfs = []
    metrics_path = Path(metrics_dir)
    for name in inputs:
        path = metrics_path / name
        if path.exists():
            df = pd.read_csv(path)
            df["source_file"] = name
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No input CSV files were found. Please check --metrics-dir and --inputs.")
    return dfs


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["model", "seed"] + METRIC_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def add_metadata(df: pd.DataFrame, setting: str) -> pd.DataFrame:
    out = df.copy()
    out["pretty_model"] = out["model"].map(lambda x: MODEL_NAME_MAP.get(x, x))
    out["family"] = out["model"].map(lambda x: MODEL_FAMILY_MAP.get(x, "other"))
    out["setting"] = setting
    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {m: ["mean", "std"] for m in METRIC_COLS}
    summary = df.groupby(["pretty_model", "model", "family", "setting"], as_index=False).agg(agg_dict)
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns.values
    ]
    rename_map = {
        "pretty_model_": "pretty_model",
        "model_": "model",
        "family_": "family",
        "setting_": "setting",
    }
    summary = summary.rename(columns=rename_map)

    if "seed" in df.columns:
        seed_counts = df.groupby("model")["seed"].nunique().reset_index().rename(columns={"seed": "n_seeds"})
        summary = summary.merge(seed_counts, on="model", how="left")

    preferred_order = [
        "MLP-Text",
        "MLP-Fin",
        "MLP-Fin+P0",
        "LR-Fin",
        "RF-Fin",
        "HGB-Fin",
        "MLP-Concat",
        "EGA",
        "MarketMirror",
    ]
    summary["sort_key"] = summary["pretty_model"].apply(
        lambda x: preferred_order.index(x) if x in preferred_order else 999
    )
    summary = summary.sort_values(["sort_key", "pretty_model"]).drop(columns=["sort_key"])
    return summary


def build_rankings(summary: pd.DataFrame):
    rank_auc = summary.sort_values(["auc_mean", "f1_pos_mean"], ascending=[False, False]).reset_index(drop=True)
    rank_auc.insert(0, "rank", range(1, len(rank_auc) + 1))

    rank_f1 = summary.sort_values(["f1_pos_mean", "auc_mean"], ascending=[False, False]).reset_index(drop=True)
    rank_f1.insert(0, "rank", range(1, len(rank_f1) + 1))

    rank_mcc = summary.sort_values(["mcc_mean", "f1_pos_mean"], ascending=[False, False]).reset_index(drop=True)
    rank_mcc.insert(0, "rank", range(1, len(rank_mcc) + 1))

    return rank_auc, rank_f1, rank_mcc


def to_pretty_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "pretty_model",
        "family",
        "n_seeds",
        "auc_mean",
        "auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_pos_mean",
        "f1_pos_std",
        "mcc_mean",
        "mcc_std",
        "brier_mean",
        "brier_std",
        "ece_mean",
        "ece_std",
    ]
    existing = [c for c in cols if c in summary.columns]
    pretty = summary[existing].copy()

    def pm(a, b):
        if pd.isna(a):
            return "--"
        if pd.isna(b):
            return f"{a:.4f}"
        return f"{a:.4f} ± {b:.4f}"

    out = pd.DataFrame()
    out["Model"] = pretty["pretty_model"]
    out["Family"] = pretty["family"]
    if "n_seeds" in pretty.columns:
        out["Seeds"] = pretty["n_seeds"]
    if "auc_mean" in pretty.columns:
        out["ROC-AUC"] = [pm(a, b) for a, b in zip(pretty["auc_mean"], pretty.get("auc_std", pd.Series([None] * len(pretty))))]
    if "pr_auc_mean" in pretty.columns:
        out["PR-AUC"] = [pm(a, b) for a, b in zip(pretty["pr_auc_mean"], pretty.get("pr_auc_std", pd.Series([None] * len(pretty))))]
    if "f1_pos_mean" in pretty.columns:
        out["F1"] = [pm(a, b) for a, b in zip(pretty["f1_pos_mean"], pretty.get("f1_pos_std", pd.Series([None] * len(pretty))))]
    if "mcc_mean" in pretty.columns:
        out["MCC"] = [pm(a, b) for a, b in zip(pretty["mcc_mean"], pretty.get("mcc_std", pd.Series([None] * len(pretty))))]
    if "brier_mean" in pretty.columns:
        out["Brier"] = [pm(a, b) for a, b in zip(pretty["brier_mean"], pretty.get("brier_std", pd.Series([None] * len(pretty))))]
    if "ece_mean" in pretty.columns:
        out["ECE"] = [pm(a, b) for a, b in zip(pretty["ece_mean"], pretty.get("ece_std", pd.Series([None] * len(pretty))))]
    return out


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for _, row in df.iterrows():
        vals = [str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    metrics_path = Path(args.metrics_dir)
    metrics_path.mkdir(parents=True, exist_ok=True)

    dfs = load_existing_csvs(args.metrics_dir, args.inputs)
    merged = pd.concat(dfs, ignore_index=True)
    merged = ensure_columns(merged)
    merged = add_metadata(merged, args.setting)

    merged_path = metrics_path / "metrics_per_seed_all.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[saved] {merged_path}")

    summary = build_summary(merged)
    summary_path = metrics_path / f"{args.output_prefix}_mean_std.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    pretty = to_pretty_table(summary)
    pretty_path = metrics_path / f"{args.output_prefix}_pretty.csv"
    pretty.to_csv(pretty_path, index=False)
    print(f"[saved] {pretty_path}")

    rank_auc, rank_f1, rank_mcc = build_rankings(summary)
    rank_auc_path = metrics_path / "ranking_auc.csv"
    rank_f1_path = metrics_path / "ranking_f1.csv"
    rank_mcc_path = metrics_path / "ranking_mcc.csv"
    rank_auc.to_csv(rank_auc_path, index=False)
    rank_f1.to_csv(rank_f1_path, index=False)
    rank_mcc.to_csv(rank_mcc_path, index=False)
    print(f"[saved] {rank_auc_path}")
    print(f"[saved] {rank_f1_path}")
    print(f"[saved] {rank_mcc_path}")

    md = markdown_table(pretty)
    md_path = metrics_path / f"{args.output_prefix}_markdown.txt"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[saved] {md_path}")

    print("\nTop by AUC:")
    print(rank_auc[["rank", "pretty_model", "family", "auc_mean", "f1_pos_mean", "mcc_mean"]].head(10).to_string(index=False))

    print("\nTop by F1:")
    print(rank_f1[["rank", "pretty_model", "family", "f1_pos_mean", "auc_mean", "mcc_mean"]].head(10).to_string(index=False))

    print("\nTop by MCC:")
    print(rank_mcc[["rank", "pretty_model", "family", "mcc_mean", "f1_pos_mean", "auc_mean"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
