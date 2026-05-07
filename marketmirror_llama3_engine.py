from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn


class MarketMirrorLlama(nn.Module):
    def __init__(self, semantic_dim=4096, financial_dim=8):
        super().__init__()
        self.s_tower = nn.Sequential(
            nn.Linear(semantic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
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
        s_feat = self.norm_s(x[:, :4096])
        p_feat = x[:, 4096:4097]
        f_feat = self.norm_f(x[:, 4097:])
        h_s = self.s_tower(s_feat)
        h_f = self.f_tower(f_feat)
        g = self.gate(torch.cat([f_feat, p_feat], dim=1))
        combined = torch.cat([h_s * (1 - g), h_f * g, p_feat], dim=1)
        return self.classifier(combined), g


def run_simulation(model, data_x, steps=50, device="cpu"):
    model.to(device)
    model.eval()
    state = data_x.clone().detach().to(device)
    trajectory = []
    for _ in range(steps):
        state.requires_grad_(True)
        probs, gates = model(state)
        potential = -torch.log(1.0 - probs + 1e-7)
        model.zero_grad()
        potential.sum().backward()
        forces = -state.grad
        with torch.no_grad():
            force_multiplier = 50.0
            f_forces = forces[:, 4097:] * force_multiplier
            kinetic_energy = (torch.sum(f_forces**2, dim=1, keepdim=True) * (1 - gates)).mean().item()
            trajectory.append(kinetic_energy)
            noise_sigma = 0.001
            state[:, 4097:] += 0.5 * f_forces + torch.randn_like(f_forces) * noise_sigma
            state.grad.zero_()
    return trajectory


def visualize_risk_evolution(traj, filename="./marketmirror_final_plot.png"):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(traj, label="Risk evolution", color="#1f77b4", linewidth=3)
    plt.title("Risk evolution over simulation steps", fontsize=14)
    plt.xlabel("Simulation steps", fontsize=12)
    plt.ylabel("Energy proxy $G^*(t)$", fontsize=12)
    plt.legend()
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_data = torch.randn(1000, 4096 + 1 + 8)
    dummy_data[:, 4096] = 0.8
    model = MarketMirrorLlama()
    risk_trajectory = run_simulation(model, dummy_data, device=device)
    visualize_risk_evolution(risk_trajectory)