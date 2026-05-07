from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def merge_multimodal_features(excel_path, mu_path, prob_path, output_name):
    excel_path = Path(excel_path)
    mu_path = Path(mu_path)
    prob_path = Path(prob_path)
    output_path = Path(f"./{output_name}_phase2.pt")

    df = pd.read_excel(excel_path)
    mu_features = np.load(mu_path)
    prob_features = np.load(prob_path).reshape(-1, 1)

    fin_cols = ["net_profit", "revenue", "operating_cash_flow", "pe_ratio"]
    existing_cols = [c for c in fin_cols if c in df.columns]
    fin_data = df[existing_cols].fillna(0).values if existing_cols else np.zeros((len(df), 0), dtype=np.float32)
    fin_scaled = StandardScaler().fit_transform(fin_data) if fin_data.shape[1] > 0 else fin_data

    combined = np.hstack([mu_features, prob_features, fin_scaled])
    labels = df["Isviolated"].fillna(0).values
    final_payload = {
        "x": torch.tensor(combined, dtype=torch.float32),
        "y": torch.tensor(labels, dtype=torch.float32),
    }
    torch.save(final_payload, output_path)


if __name__ == "__main__":
    merge_multimodal_features("./data2020.xlsx", "./f2020_mu.npy", "./f2020_prob.npy", "train2020")
    merge_multimodal_features("./data2022.xlsx", "./f2022_mu.npy", "./f2022_prob.npy", "val2022")