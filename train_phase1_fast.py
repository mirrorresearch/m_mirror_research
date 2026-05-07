from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class MarketMirrorPhase1(nn.Module):
    def __init__(self, model_name="./roberta_local"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden_size = self.encoder.config.hidden_size
        self.deception_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        s_hat = self.deception_head(pooled)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return s_hat, kl_div


def start_training():
    batch_size = 128
    epochs = 3
    lr = 2e-5
    data_path = Path("./processed_data.pt")
    save_path = Path("./mm_phase1_final.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_data = torch.load(data_path, weights_only=True)
    dataset = TensorDataset(full_data["input_ids"], full_data["attention_mask"], full_data["labels"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = MarketMirrorPhase1().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for ids, mask, lbls in pbar:
            ids, mask, lbls = ids.to(device), mask.to(device), lbls.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                s_hat, kl_div = model(ids, mask)
                loss_bce = nn.BCELoss()(s_hat, lbls)
                loss_kl = 0.01 * kl_div.mean()
                loss = loss_bce + loss_kl
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar.set_postfix(bce=f"{loss_bce.item():.4f}", kl=f"{loss_kl.item():.4f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    start_training()