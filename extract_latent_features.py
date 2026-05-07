from pathlib import Path

import numpy as np
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
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        self.deception_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        mu = self.fc_mu(pooled)
        logits = self.deception_head(pooled)
        return mu, torch.sigmoid(logits)


def extract_and_save(data_pt_path, save_prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarketMirrorPhase1().to(device)
    model_path = Path("./mm_phase1_final.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data = torch.load(data_pt_path, map_location="cpu", weights_only=True)
    dataset = TensorDataset(data["input_ids"], data["attention_mask"])
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_mus = []
    all_probs = []
    with torch.no_grad():
        for ids, mask in tqdm(loader):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                mu, prob = model(ids.to(device), mask.to(device))
            all_mus.append(mu.cpu().float().numpy())
            all_probs.append(prob.cpu().float().numpy())

    mu_final = np.concatenate(all_mus)
    prob_final = np.concatenate(all_probs)
    np.save(f"./{save_prefix}_mu.npy", mu_final)
    np.save(f"./{save_prefix}_prob.npy", prob_final)


if __name__ == "__main__":
    extract_and_save("./processed_data.pt", "f2020")
    if Path("./processed_data2022.pt").exists():
        extract_and_save("./processed_data2022.pt", "f2022")