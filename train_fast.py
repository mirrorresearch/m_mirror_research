import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, get_cosine_schedule_with_warmup
from tqdm import tqdm

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./roberta_local"))
DATA_PATH = Path(os.getenv("DATA_PATH", "./processed_data.pt"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "./mm_phase1_final.pth"))


class MarketMirrorPhase1(nn.Module):
    def __init__(self, model_path: str | Path = MODEL_DIR):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(model_path), local_files_only=True)
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
        s_logits = self.deception_head(pooled)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return s_logits, kl_div


def train():
    batch_size = 128
    device = torch.device("cuda")

    data = torch.load(DATA_PATH, map_location="cpu", weights_only=True)
    dataset = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    model = MarketMirrorPhase1().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler("cuda")

    total_steps = len(loader) * 3
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(3):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for ids, mask, lbls in pbar:
            ids, mask, lbls = ids.to(device), mask.to(device), lbls.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                s_logits, kl_div = model(ids, mask)
                loss_bce = criterion(s_logits, lbls)
                loss_kl = 0.01 * kl_div.mean()
                total_loss = loss_bce + loss_kl
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_PATH)


if __name__ == "__main__":
    train()