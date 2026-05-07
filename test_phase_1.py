from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
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


def run_evaluation():
    data_path = Path("./processed_data.pt")
    weights_path = Path("./mm_phase1_final.pth")
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(data_path, map_location="cpu", weights_only=True)
    full_dataset = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])
    test_size = 10000
    indices = list(range(len(full_dataset) - test_size, len(full_dataset)))
    test_dataset = torch.utils.data.Subset(full_dataset, indices)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MarketMirrorPhase1().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for ids, mask, lbls in tqdm(loader):
            ids, mask = ids.to(device), mask.to(device)
            logits = model(ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(lbls.numpy())

    all_preds = np.array(all_preds).flatten()
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    print(classification_report(all_labels, binary_preds, target_names=["class_0", "class_1"]))
    print(f"AUC: {roc_auc_score(all_labels, all_preds):.4f}")


if __name__ == "__main__":
    run_evaluation()