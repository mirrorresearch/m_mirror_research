import argparse
import json
import os
import random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


class ExpertGatedNet(nn.Module):
    def __init__(self, input_dim: int, sem_dim: int = 768, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        if input_dim <= sem_dim + 1:
            raise ValueError("input_dim is too small for the configured semantic and risk dimensions.")
        self.sem_dim = sem_dim
        self.risk_dim = 1
        self.fin_dim = input_dim - sem_dim - self.risk_dim

        self.sem_proj = nn.Sequential(
            nn.Linear(self.sem_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fin_proj = nn.Sequential(
            nn.Linear(self.fin_dim + self.risk_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sem = x[:, : self.sem_dim]
        risk = x[:, self.sem_dim : self.sem_dim + 1]
        fin = x[:, self.sem_dim + 1 :]

        h_sem = self.sem_proj(sem)
        h_fin = self.fin_proj(torch.cat([risk, fin], dim=1))

        g = self.gate(torch.cat([h_sem, h_fin], dim=1))
        h = g * h_fin + (1.0 - g) * h_sem
        return self.out(h)


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


def load_pt_dataset(path: str):
    data = torch.load(path, map_location="cpu", weights_only=True)
    x = data["x"].cpu().numpy().astype(np.float32)
    y = data["y"].cpu().numpy().astype(np.float32)
    return x, y


def choose_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_vals = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    idx = int(np.argmax(f1_vals))
    return float(thresholds[idx]) if len(thresholds) else 0.5


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


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def run_once(args, seed: int):
    set_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_all, y_train_all = load_pt_dataset(args.train_pt)
    x_2022, y_2022 = load_pt_dataset(args.val2022_pt)

    x_train, x_valid2020, y_train, y_valid2020 = train_test_split(
        x_train_all,
        y_train_all,
        test_size=args.valid_ratio,
        random_state=args.split_seed,
        stratify=y_train_all,
    )

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_valid2020 = scaler.transform(x_valid2020).astype(np.float32)
    x_2022 = scaler.transform(x_2022).astype(np.float32)

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    valid_loader = make_loader(x_valid2020, y_valid2020, args.eval_batch_size, shuffle=False)
    val2022_loader = make_loader(x_2022, y_2022, args.eval_batch_size, shuffle=False)

    model = ExpertGatedNet(
        input_dim=x_train.shape[1],
        sem_dim=args.sem_dim,
        hidden=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
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
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        all_prob = []
        all_true = []
        with torch.no_grad():
            for bx, by in valid_loader:
                logits = model(bx.to(device))
                prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                all_prob.append(prob)
                all_true.append(by.numpy())
        prob_valid = np.concatenate(all_prob)
        true_valid = np.concatenate(all_true)
        auc_valid = roc_auc_score(true_valid, prob_valid)

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
    model.eval()

    def infer(loader):
        probs, labels = [], []
        with torch.no_grad():
            for bx, by in loader:
                logit = model(bx.to(device))
                probs.append(torch.sigmoid(logit).squeeze(1).cpu().numpy())
                labels.append(by.numpy())
        return np.concatenate(labels), np.concatenate(probs)

    y_valid, p_valid = infer(valid_loader)
    threshold = choose_best_threshold(y_valid, p_valid)
    valid_stats = eval_with_threshold(y_valid, p_valid, threshold)

    y_22, p_22 = infer(val2022_loader)
    eval2022_stats = eval_with_threshold(y_22, p_22, threshold)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "valid2020": asdict(valid_stats),
        "eval2022": asdict(eval2022_stats),
        "y2022": y_22.tolist(),
        "p2022": p_22.tolist(),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run expert-gated model with multiple seeds and export per-seed metrics.")
    p.add_argument("--train-pt", type=str, default="./train2020_phase2_expert.pt")
    p.add_argument("--val2022-pt", type=str, default="./val2022_phase2_expert.pt")
    p.add_argument("--out-dir", type=str, default="./paper_metrics")
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
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    rows = []

    for seed in args.seeds:
        print(f"=== expert seed={seed} ===")
        r = run_once(args, seed)
        all_results.append(r)
        s = r["eval2022"]
        rows.append({"model": "expert", "seed": seed, **s})
        print(f"AUC={s['auc']:.4f}, F1={s['f1_pos']:.4f}, th={s['threshold']:.4f}")

    with open(os.path.join(args.out_dir, "expert_runs.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metrics_per_seed_expert.csv"), index=False)
    print("Saved expert results.")


if __name__ == "__main__":
    main()
