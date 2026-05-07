import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    f1_score,
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


def choose_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1_vals = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    idx = int(np.argmax(f1_vals))
    return float(thresholds[idx])


def eval_with_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(np.int32)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }

class ExpertGatedNet(nn.Module):
    def __init__(self, semantic_dim=768, financial_dim=8):
        super().__init__()
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

    def forward(self, x):
        s_feat = self.norm_s(x[:, :768])
        p_feat = x[:, 768:769]
        f_feat = self.norm_f(x[:, 769:])

        h_s = self.s_tower(s_feat)
        h_f = self.f_tower(f_feat)
        g = self.gate(torch.cat([f_feat, p_feat], dim=1))
        combined = torch.cat([h_s * (1 - g), h_f * g, p_feat], dim=1)
        return self.classifier(combined)

def run_once(args, seed: int):
    set_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_all, y_train_all = load_pt_dataset(args.train_pt)
    x_2022_all, y_2022 = load_pt_dataset(args.val2022_pt)

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
    x_2022 = scaler.transform(x_2022_all).astype(np.float32)

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    valid_loader = make_loader(x_valid2020, y_valid2020, args.eval_batch_size, shuffle=False)
    val2022_loader = make_loader(x_2022, y_2022, args.eval_batch_size, shuffle=False)

    model = ExpertGatedNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction="none")

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    best_auc = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device).unsqueeze(1)  # [B,1]

            optimizer.zero_grad()
            p = model(bx)  # already sigmoid, probability in [0,1]

            w = torch.where(by > 0.5, pos_weight, torch.tensor(1.0, device=device))
            loss = (w * criterion(p, by)).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for bx, by in valid_loader:
                bx = bx.to(device)
                p = model(bx).squeeze(1).cpu().numpy()
                y_prob.append(p)
                y_true.append(by.numpy())
        y_true = np.concatenate(y_true).astype(np.int32)
        y_prob = np.concatenate(y_prob).astype(np.float64)
        auc_valid = float(roc_auc_score(y_true, y_prob))

        if auc_valid > best_auc:
            best_auc = auc_valid
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    def predict_probs(loader):
        probs, labels = [], []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                p = model(bx).squeeze(1).cpu().numpy()
                probs.append(p)
                labels.append(by.numpy())
        return np.concatenate(labels).astype(np.int32), np.concatenate(probs).astype(np.float64)

    y_valid, p_valid = predict_probs(valid_loader)
    threshold = choose_best_threshold(y_valid, p_valid)
    valid_stats = eval_with_threshold(y_valid, p_valid, threshold)

    y_22, p_22 = predict_probs(val2022_loader)
    eval2022_stats = eval_with_threshold(y_22, p_22, threshold)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "valid2020": valid_stats,
        "eval2022": eval2022_stats,
    }


def summarize(results):
    keys = ["auc", "pr_auc", "precision", "recall", "f1"]
    summary = {}
    for block in ["valid2020", "eval2022"]:
        summary[block] = {}
        for k in keys:
            vals = np.array([r[block][k] for r in results], dtype=np.float64)
            summary[block][k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        th = np.array([r[block]["threshold"] for r in results], dtype=np.float64)
        summary[block]["threshold"] = {
            "mean": float(th.mean()),
            "std": float(th.std()),
            "min": float(th.min()),
            "max": float(th.max()),
        }
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Reproducible protocol for mm_phase2_expert_trainer.")
    p.add_argument("--train-pt", type=str, default="./train2020_phase2_expert.pt")
    p.add_argument("--val2022-pt", type=str, default="./val2022_phase2_expert.pt")
    p.add_argument("--out-dir", type=str, default="./expert_repro_runs")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    p.add_argument("--split-seed", type=int, default=999)
    p.add_argument("--valid-ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        print(f"\n=== Expert run seed={seed} ===")
        result = run_once(args, seed)
        all_results.append(result)
        print(
            f"valid2020 AUC={result['valid2020']['auc']:.4f} | "
            f"2022 AUC={result['eval2022']['auc']:.4f} | "
            f"2022 F1={result['eval2022']['f1']:.4f} | "
            f"th={result['eval2022']['threshold']:.4f}"
        )

    summary = summarize(all_results)
    out_json = {
        "config": vars(args),
        "runs": all_results,
        "summary": summary,
    }
    output_path = os.path.join(args.out_dir, "results_2020.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    print("\n=== Expert Reproducibility Summary (2022) ===")
    print(
        f"AUC mean±std: "
        f"{summary['eval2022']['auc']['mean']:.4f} ± {summary['eval2022']['auc']['std']:.4f}"
    )
    print(
        f"F1  mean±std: "
        f"{summary['eval2022']['f1']['mean']:.4f} ± {summary['eval2022']['f1']['std']:.4f}"
    )
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()