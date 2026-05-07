import argparse
import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.3):
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


def load_pt_dataset(path: str):
    data = torch.load(path, map_location="cpu", weights_only=True)
    x = data["x"].cpu().numpy().astype(np.float32)
    y = data["y"].cpu().numpy().astype(np.float32)
    return x, y


def split_features_safe(x: np.ndarray, sem_dim: int):
    total_dim = x.shape[1]
    if total_dim < sem_dim:
        raise ValueError(f"Input dim {total_dim} < sem_dim={sem_dim}.")

    x_text = x[:, :sem_dim]

    remainder = total_dim - sem_dim
    if remainder <= 0:
        x_finp0 = None
        x_fin = None
    elif remainder == 1:
        x_finp0 = x[:, sem_dim:]
        x_fin = None
    else:
        x_finp0 = x[:, sem_dim:]
        x_fin = x[:, sem_dim + 1 :]

    return x_text, x_finp0, x_fin


def choose_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1_vals = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    idx = int(np.argmax(f1_vals))
    return float(thresholds[idx])


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


def run_mlp_once(model_name, x_train_all, y_train_all, x_2022, y_2022, args, seed):
    set_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    x_2022_scaled = scaler.transform(x_2022).astype(np.float32)

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    valid_loader = make_loader(x_valid2020, y_valid2020, args.eval_batch_size, shuffle=False)
    test_loader = make_loader(x_2022_scaled, y_2022, args.eval_batch_size, shuffle=False)

    model = SimpleMLP(input_dim=x_train.shape[1], hidden=args.hidden_dim, dropout=args.dropout).to(device)
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
        all_prob, all_true = [], []
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

    y_22, p_22 = infer(test_loader)
    eval2022_stats = eval_with_threshold(y_22, p_22, threshold)

    return {
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "valid2020": asdict(valid_stats),
        "eval2022": asdict(eval2022_stats),
        "y2022": y_22.tolist(),
        "p2022": p_22.tolist(),
    }


def run_sklearn_once(model_name, build_model_fn, x_train_all, y_train_all, x_2022, y_2022, args, seed):
    set_seed(seed, deterministic=True)

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
    x_2022_scaled = scaler.transform(x_2022).astype(np.float32)

    model = build_model_fn(seed)
    model.fit(x_train, y_train)

    p_valid = model.predict_proba(x_valid2020)[:, 1]
    threshold = choose_best_threshold(y_valid2020, p_valid)
    valid_stats = eval_with_threshold(y_valid2020, p_valid, threshold)

    p_22 = model.predict_proba(x_2022_scaled)[:, 1]
    eval2022_stats = eval_with_threshold(y_2022, p_22, threshold)

    return {
        "model": model_name,
        "seed": seed,
        "best_epoch": -1,
        "valid2020": asdict(valid_stats),
        "eval2022": asdict(eval2022_stats),
        "y2022": y_2022.tolist(),
        "p2022": p_22.tolist(),
    }


def build_logreg(seed):
    return LogisticRegression(
        random_state=seed,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
    )


def build_rf(seed):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def build_hgb(seed):
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=seed,
    )


def save_results(out_dir, model_name, results):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{model_name}_runs.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    rows = []
    for r in results:
        rows.append({"model": model_name, "seed": r["seed"], **r["eval2022"]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"metrics_per_seed_{model_name}.csv"), index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Run core baselines safely.")
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

    x_train_all, y_train_all = load_pt_dataset(args.train_pt)
    x_2022, y_2022 = load_pt_dataset(args.val2022_pt)

    x_text_train, x_finp0_train, x_fin_train = split_features_safe(x_train_all, args.sem_dim)
    x_text_2022, x_finp0_2022, x_fin_2022 = split_features_safe(x_2022, args.sem_dim)

    print(f"train shape: {x_train_all.shape}")
    print(f"test  shape: {x_2022.shape}")
    print(f"text dim: {x_text_train.shape[1]}")
    print(f"finp0 dim: {0 if x_finp0_train is None else x_finp0_train.shape[1]}")
    print(f"fin   dim: {0 if x_fin_train is None else x_fin_train.shape[1]}")

    all_rows = []
    experiment_plan = []

    # 总是跑 text
    experiment_plan.append(
        ("text_mlp", lambda seed: run_mlp_once("text_mlp", x_text_train, y_train_all, x_text_2022, y_2022, args, seed))
    )

    # 有 finp0 就跑
    if x_finp0_train is not None and x_finp0_train.shape[1] > 0:
        experiment_plan.append(
            ("finp0_mlp", lambda seed: run_mlp_once("finp0_mlp", x_finp0_train, y_train_all, x_finp0_2022, y_2022, args, seed))
        )

    # 有 pure financial 才跑这些
    if x_fin_train is not None and x_fin_train.shape[1] > 0:
        experiment_plan.extend([
            ("fin_mlp", lambda seed: run_mlp_once("fin_mlp", x_fin_train, y_train_all, x_fin_2022, y_2022, args, seed)),
            ("logreg_fin", lambda seed: run_sklearn_once("logreg_fin", build_logreg, x_fin_train, y_train_all, x_fin_2022, y_2022, args, seed)),
            ("rf_fin", lambda seed: run_sklearn_once("rf_fin", build_rf, x_fin_train, y_train_all, x_fin_2022, y_2022, args, seed)),
            ("hgb_fin", lambda seed: run_sklearn_once("hgb_fin", build_hgb, x_fin_train, y_train_all, x_fin_2022, y_2022, args, seed)),
        ])
    else:
        print("[warn] pure financial features not found; skipping fin_mlp/logreg_fin/rf_fin/hgb_fin")

    for model_name, runner in experiment_plan:
        print(f"\n===== running {model_name} =====")
        results = []
        for seed in args.seeds:
            r = runner(seed)
            results.append(r)
            s = r["eval2022"]
            all_rows.append({"model": model_name, "seed": seed, **s})
            print(f"seed={seed} AUC={s['auc']:.4f} F1={s['f1_pos']:.4f} th={s['threshold']:.4f}")
        save_results(args.out_dir, model_name, results)

    pd.DataFrame(all_rows).to_csv(os.path.join(args.out_dir, "metrics_per_seed_core_baselines.csv"), index=False)
    print("\nSaved core baseline metrics to metrics_per_seed_core_baselines.csv")


if __name__ == "__main__":
    main()