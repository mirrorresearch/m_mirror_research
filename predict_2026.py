from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel
from tqdm import tqdm


class MarketMirrorPhase1(nn.Module):
    def __init__(self, model_path="./roberta_local"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden_size = self.encoder.config.hidden_size
        self.deception_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.deception_head(pooled)


def run_batch_detection():
    weights_path = Path("./mm_phase1_final.pth")
    processed_data_path = Path("./processed_test_data.pt")
    original_excel = Path("./data_test2026.xlsx")
    output_excel = Path("./results_2026_detected.xlsx")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MarketMirrorPhase1().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    data = torch.load(processed_data_path, map_location="cpu", weights_only=True)
    dataset = TensorDataset(data["input_ids"], data["attention_mask"])
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for ids, mask in tqdm(loader):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids.to(device), mask.to(device))
                probs = torch.sigmoid(logits).cpu().float().numpy().flatten()
                all_probs.extend(probs)

    df = pd.read_excel(original_excel)
    if len(all_probs) != len(df):
        all_probs = all_probs[: len(df)] if len(all_probs) > len(df) else all_probs + [0] * (len(df) - len(all_probs))

    df["Violation_Probability"] = all_probs
    df["Violation_Prediction"] = [1 if p > 0.5 else 0 for p in all_probs]
    df = df.sort_values(by="Violation_Probability", ascending=False)
    df.to_excel(output_excel, index=False)


if __name__ == "__main__":
    run_batch_detection()