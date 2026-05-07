import argparse
import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset


os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", "1")


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_pt_dataset(path: str):
    data = torch.load(path, map_location="cpu", weights_only=True)
    x = data["x"].cpu().numpy().astype(np.float32)
    y = data["y"].cpu().numpy().astype(np.float32)
    return x, y


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def choose_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1_vals = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    idx = int(np.argmax(f1_vals))
    return float(thresholds[idx])


@dataclass
class EvalStats:
    auc: float
    pr_auc: float
    precision_pos: float
    recall_pos: float
    f1_pos: float
    bal_acc: float
    mcc: float
    brier: float
    ece: float
    threshold: float


def eval_with_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> EvalStats:
    y_pred = (y_prob >= threshold).astype(np.int32)
    return EvalStats(
        auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        precision_pos=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_pos=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_pos=float(f1_score(y_true, y_pred, zero_division=0)),
        bal_acc=float(balanced_accuracy_score(y_true, y_pred)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        brier=float(np.mean((y_prob - y_true) ** 2)),
        ece=expected_calibration_error(y_true, y_prob, n_bins=10),
        threshold=float(threshold),
    )


class TextMLP(nn.Module):
    def __init__(self, input_dim: int = 768, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertGatedNet(nn.Module):
    def __init__(self, semantic_dim: int = 768, financial_dim: int = 8):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.financial_dim = financial_dim
        self.s_tower = nn.Sequential(
            nn.Linear(semantic_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 16),
        )
        self.f_tower = nn.Sequential(
            nn.Linear(financial_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.gate = nn.Sequential(
            nn.Linear(financial_dim + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 + 16 + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.norm_s = nn.LayerNorm(semantic_dim)
        self.norm_f = nn.LayerNorm(financial_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s_feat = self.norm_s(x[:, : self.semantic_dim])
        p_feat = x[:, self.semantic_dim : self.semantic_dim + 1]
        f_feat = self.norm_f(x[:, self.semantic_dim + 1 :])
        h_s = self.s_tower(s_feat)
        h_f = self.f_tower(f_feat)
        g = self.gate(torch.cat([f_feat, p_feat], dim=1))
        combined = torch.cat([h_s * (1.0 - g), h_f * g, p_feat], dim=1)
        return self.classifier(combined)

    def predict_gate(self, x: torch.Tensor) -> torch.Tensor:
        s_dim = self.semantic_dim
        p_feat = x[:, s_dim : s_dim + 1]
        f_feat = self.norm_f(x[:, s_dim + 1 :])
        return self.gate(torch.cat([f_feat, p_feat], dim=1))


def infer_probs(model, loader, device, output_is_prob: bool):
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            out = model(bx).squeeze(1)
            if not output_is_prob:
                out = torch.sigmoid(out)
            probs.append(out.cpu().numpy())
            labels.append(by.numpy())
    return np.concatenate(labels).astype(np.int32), np.concatenate(probs).astype(np.float64)


def infer_gates(model: ExpertGatedNet, loader, device):
    gates = []
    model.eval()
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            g = model.predict_gate(bx).squeeze(1).cpu().numpy()
            gates.append(g)
    return np.concatenate(gates).astype(np.float64)


def train_model(model_name, model, x_train, y_train, x_valid, y_valid, args, device, output_is_prob):
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    valid_loader = make_loader(x_valid, y_valid, args.eval_batch_size, shuffle=False)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    if output_is_prob:
        criterion = nn.BCELoss(reduction="none")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = -1.0
    best_state = None
    best_epoch = 0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(bx)
            if output_is_prob:
                w = torch.where(by > 0.5, pos_weight, torch.tensor(1.0, device=device))
                loss = (w * criterion(out, by)).mean()
            else:
                loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        yv, pv = infer_probs(model, valid_loader, device, output_is_prob=output_is_prob)
        auc_valid = roc_auc_score(yv, pv)
        if auc_valid > best_auc:
            best_auc = float(auc_valid)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                break

    model.load_state_dict(best_state)
    return best_epoch


def population_stability_index(a: np.ndarray, b: np.ndarray, n_bins: int = 10) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(a, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    a_hist, _ = np.histogram(a, bins=edges)
    b_hist, _ = np.histogram(b, bins=edges)
    eps = 1e-8
    a_pct = a_hist / max(a_hist.sum(), 1) + eps
    b_pct = b_hist / max(b_hist.sum(), 1) + eps
    return float(np.sum((b_pct - a_pct) * np.log(b_pct / a_pct)))


def feature_drift_summary(x_ref: np.ndarray, x_new: np.ndarray, prefix: str, max_dims: int = 768):
    dim = min(x_ref.shape[1], max_dims)
    rows = []
    for j in range(dim):
        a = x_ref[:, j]
        b = x_new[:, j]
        ks = ks_2samp(a, b)
        rows.append({
            "group": prefix,
            "feature_idx": j,
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "psi": population_stability_index(a, b, n_bins=10),
            "ref_mean": float(np.mean(a)),
            "new_mean": float(np.mean(b)),
            "abs_mean_shift": float(abs(np.mean(b) - np.mean(a))),
        })
    df = pd.DataFrame(rows)
    return {
        "group": prefix,
        "n_features": int(dim),
        "ks_mean": float(df["ks_stat"].mean()),
        "ks_median": float(df["ks_stat"].median()),
        "ks_max": float(df["ks_stat"].max()),
        "psi_mean": float(df["psi"].mean()),
        "psi_median": float(df["psi"].median()),
        "psi_max": float(df["psi"].max()),
    }, df


def run_once(args, seed: int):
    set_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_all, y_all = load_pt_dataset(args.train_pt)
    x_2022_all, y_2022 = load_pt_dataset(args.val2022_pt)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_all,
        y_all,
        test_size=args.valid_ratio,
        random_state=args.split_seed,
        stratify=y_all,
    )

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_valid = scaler.transform(x_valid).astype(np.float32)
    x_2022 = scaler.transform(x_2022_all).astype(np.float32)

    x_train_text = x_train[:, : args.sem_dim]
    x_valid_text = x_valid[:, : args.sem_dim]
    x_2022_text = x_2022[:, : args.sem_dim]

    valid_full_loader = make_loader(x_valid, y_valid, args.eval_batch_size, shuffle=False)
    test_full_loader = make_loader(x_2022, y_2022, args.eval_batch_size, shuffle=False)
    valid_text_loader = make_loader(x_valid_text, y_valid, args.eval_batch_size, shuffle=False)
    test_text_loader = make_loader(x_2022_text, y_2022, args.eval_batch_size, shuffle=False)

    text_model = TextMLP(input_dim=args.sem_dim, hidden=args.hidden_dim, dropout=args.dropout).to(device)
    text_epoch = train_model(
        "text_mlp", text_model, x_train_text, y_train, x_valid_text, y_valid, args, device, output_is_prob=False
    )

    yv_text, pv_text = infer_probs(text_model, valid_text_loader, device, output_is_prob=False)
    th_text = choose_best_threshold(yv_text, pv_text)
    valid_text_stats = eval_with_threshold(yv_text, pv_text, th_text)
    yt_text, pt_text = infer_probs(text_model, test_text_loader, device, output_is_prob=False)
    test_text_stats = eval_with_threshold(yt_text, pt_text, th_text)

    fin_dim = x_train.shape[1] - args.sem_dim - 1
    if fin_dim <= 0:
        raise ValueError(f"Expected full feature layout [sem_dim, p0, financial]. Got total dim={x_train.shape[1]}.")
    expert_model = ExpertGatedNet(semantic_dim=args.sem_dim, financial_dim=fin_dim).to(device)
    expert_epoch = train_model(
        "ega", expert_model, x_train, y_train, x_valid, y_valid, args, device, output_is_prob=True
    )

    yv_ega, pv_ega = infer_probs(expert_model, valid_full_loader, device, output_is_prob=True)
    th_ega = choose_best_threshold(yv_ega, pv_ega)
    valid_ega_stats = eval_with_threshold(yv_ega, pv_ega, th_ega)
    yt_ega, pt_ega = infer_probs(expert_model, test_full_loader, device, output_is_prob=True)
    test_ega_stats = eval_with_threshold(yt_ega, pt_ega, th_ega)

    gv = infer_gates(expert_model, valid_full_loader, device)
    gt = infer_gates(expert_model, test_full_loader, device)

    def drop_row(model_name, valid_stats, test_stats, best_epoch):
        v = asdict(valid_stats)
        t = asdict(test_stats)
        out = {"model": model_name, "seed": seed, "best_epoch": best_epoch}
        for k in v:
            out[f"valid2020_{k}"] = v[k]
            out[f"test2022_{k}"] = t[k]
            if k != "threshold":
                out[f"drop_{k}"] = v[k] - t[k]
                out[f"relative_drop_{k}"] = (v[k] - t[k]) / (abs(v[k]) + 1e-8)
        out["prob_psi_valid_to_2022"] = population_stability_index(
            pv_text if model_name == "text_mlp" else pv_ega,
            pt_text if model_name == "text_mlp" else pt_ega,
            n_bins=10,
        )
        ks = ks_2samp(
            pv_text if model_name == "text_mlp" else pv_ega,
            pt_text if model_name == "text_mlp" else pt_ega,
        )
        out["prob_ks_stat_valid_to_2022"] = float(ks.statistic)
        out["prob_ks_pvalue_valid_to_2022"] = float(ks.pvalue)
        return out

    rows = [
        drop_row("text_mlp", valid_text_stats, test_text_stats, text_epoch),
        drop_row("ega", valid_ega_stats, test_ega_stats, expert_epoch),
    ]

    gate_summary = {
        "seed": seed,
        "valid_gate_mean": float(np.mean(gv)),
        "valid_gate_std": float(np.std(gv)),
        "test_gate_mean": float(np.mean(gt)),
        "test_gate_std": float(np.std(gt)),
        "gate_mean_shift": float(np.mean(gt) - np.mean(gv)),
        "gate_psi_valid_to_2022": population_stability_index(gv, gt, n_bins=10),
        "gate_ks_stat_valid_to_2022": float(ks_2samp(gv, gt).statistic),
        "gate_ks_pvalue_valid_to_2022": float(ks_2samp(gv, gt).pvalue),
    }

    return rows, gate_summary, {
        "seed": seed,
        "y_valid": y_valid.tolist(),
        "y_2022": y_2022.tolist(),
        "p_valid_text": pv_text.tolist(),
        "p_2022_text": pt_text.tolist(),
        "p_valid_ega": pv_ega.tolist(),
        "p_2022_ega": pt_ega.tolist(),
        "gate_valid_ega": gv.tolist(),
        "gate_2022_ega": gt.tolist(),
    }


def summarize_drift(df: pd.DataFrame):
    metrics = [
        "valid2020_auc", "test2022_auc", "drop_auc", "relative_drop_auc",
        "valid2020_pr_auc", "test2022_pr_auc", "drop_pr_auc", "relative_drop_pr_auc",
        "valid2020_f1_pos", "test2022_f1_pos", "drop_f1_pos", "relative_drop_f1_pos",
        "valid2020_mcc", "test2022_mcc", "drop_mcc", "relative_drop_mcc",
        "valid2020_brier", "test2022_brier", "drop_brier",
        "valid2020_ece", "test2022_ece", "drop_ece",
        "prob_psi_valid_to_2022", "prob_ks_stat_valid_to_2022",
    ]
    rows = []
    for model, g in df.groupby("model"):
        row = {"model": model, "n_seeds": int(g["seed"].nunique())}
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_std"] = float(g[m].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Concept drift analysis: text-only vs expert-gated model.")
    p.add_argument("--train-pt", type=str, default="./train2020_phase2_expert.pt")
    p.add_argument("--val2022-pt", type=str, default="./val2022_phase2_expert.pt")
    p.add_argument("--out-dir", type=str, default="./paper_metrics/drift_analysis")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    p.add_argument("--split-seed", type=int, default=999)
    p.add_argument("--valid-ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--sem-dim", type=int, default=768)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.2)
    p.add_argument("--feature-drift-max-text-dims", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    x_all, y_all = load_pt_dataset(args.train_pt)
    x_2022_all, _ = load_pt_dataset(args.val2022_pt)
    x_train_raw, x_valid_raw, _, _ = train_test_split(
        x_all,
        y_all,
        test_size=args.valid_ratio,
        random_state=args.split_seed,
        stratify=y_all,
    )
    scaler = RobustScaler()
    scaler.fit(x_train_raw)
    x_valid_scaled = scaler.transform(x_valid_raw).astype(np.float32)
    x_2022_scaled = scaler.transform(x_2022_all).astype(np.float32)

    text_summary, text_feat_df = feature_drift_summary(
        x_valid_scaled[:, : args.sem_dim],
        x_2022_scaled[:, : args.sem_dim],
        prefix="text_embedding",
        max_dims=args.feature_drift_max_text_dims,
    )
    p0_summary, p0_feat_df = feature_drift_summary(
        x_valid_scaled[:, args.sem_dim : args.sem_dim + 1],
        x_2022_scaled[:, args.sem_dim : args.sem_dim + 1],
        prefix="p0",
        max_dims=1,
    )
    fin_summary, fin_feat_df = feature_drift_summary(
        x_valid_scaled[:, args.sem_dim + 1 :],
        x_2022_scaled[:, args.sem_dim + 1 :],
        prefix="financial",
        max_dims=10_000,
    )
    pd.concat([text_feat_df, p0_feat_df, fin_feat_df], ignore_index=True).to_csv(
        os.path.join(args.out_dir, "feature_drift_by_dim.csv"), index=False
    )
    with open(os.path.join(args.out_dir, "feature_drift_summary.json"), "w", encoding="utf-8") as f:
        json.dump([text_summary, p0_summary, fin_summary], f, indent=2, ensure_ascii=False)

    all_rows = []
    gate_rows = []
    prob_records = []
    for seed in args.seeds:
        print(f"\n=== drift seed={seed} ===")
        rows, gate_summary, probs = run_once(args, seed)
        all_rows.extend(rows)
        gate_rows.append(gate_summary)
        prob_records.append(probs)
        for r in rows:
            print(
                f"{r['model']} | "
                f"valid AUC={r['valid2020_auc']:.4f}, 2022 AUC={r['test2022_auc']:.4f}, drop={r['drop_auc']:.4f} | "
                f"valid F1={r['valid2020_f1_pos']:.4f}, 2022 F1={r['test2022_f1_pos']:.4f}, drop={r['drop_f1_pos']:.4f} | "
                f"prob PSI={r['prob_psi_valid_to_2022']:.4f}"
            )

    drift_df = pd.DataFrame(all_rows)
    gate_df = pd.DataFrame(gate_rows)
    drift_df.to_csv(os.path.join(args.out_dir, "drift_metrics_per_seed.csv"), index=False)
    gate_df.to_csv(os.path.join(args.out_dir, "gate_drift_per_seed.csv"), index=False)

    summary_df = summarize_drift(drift_df)
    summary_df.to_csv(os.path.join(args.out_dir, "drift_summary_mean_std.csv"), index=False)

    with open(os.path.join(args.out_dir, "prediction_records.json"), "w", encoding="utf-8") as f:
        json.dump(prob_records, f, indent=2, ensure_ascii=False)

    print("\n=== Drift Summary ===")
    show_cols = [
        "model",
        "test2022_auc_mean",
        "drop_auc_mean",
        "test2022_f1_pos_mean",
        "drop_f1_pos_mean",
        "test2022_brier_mean",
        "drop_brier_mean",
        "test2022_ece_mean",
        "drop_ece_mean",
        "prob_psi_valid_to_2022_mean",
    ]
    print(summary_df[[c for c in show_cols if c in summary_df.columns]].to_string(index=False))
    print(f"\nSaved drift analysis to: {args.out_dir}")


if __name__ == "__main__":
    main()
