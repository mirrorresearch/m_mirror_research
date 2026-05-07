import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


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
        return self.classifier(combined), g


class MarketMirrorSimulator:
    def __init__(self, model_path, data_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = ExpertGatedNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        data = torch.load(data_path, map_location="cpu", weights_only=True)
        self.agents_state = data["x"].to(self.device)

    def run_simulation(self, steps=50, lr=0.01, sigma=0.005):
        energy_trajectory = []
        for _ in range(steps):
            self.agents_state.requires_grad_(True)
            probs, gates = self.model(self.agents_state)
            potential = -torch.log(1.0 - probs + 1e-7)
            self.model.zero_grad()
            potential.sum().backward()
            forces = -self.agents_state.grad
            with torch.no_grad():
                f_forces = forces[:, 769:]
                kinetic_energy = torch.sum(f_forces**2, dim=1, keepdim=True) * (1 - gates)
                energy_trajectory.append(kinetic_energy.mean().item())
                noise = torch.randn_like(self.agents_state[:, 769:]) * sigma
                self.agents_state[:, 769:] += lr * f_forces + noise
                self.agents_state.grad.zero_()
        return energy_trajectory


if __name__ == "__main__":
    model_file = Path(os.getenv("MODEL_FILE", "./mm_phase2_expert_best.pth"))
    data_file = Path(os.getenv("DATA_FILE", "./val2022_phase2_expert.pt"))
    output_file = Path(os.getenv("OUTPUT_FILE", "./risk_trajectory_2022.csv"))

    sim = MarketMirrorSimulator(model_file, data_file)
    trajectory = sim.run_simulation(steps=50, lr=0.02, sigma=0.01)
    pd.DataFrame({"step": range(len(trajectory)), "G_star": trajectory}).to_csv(output_file, index=False)