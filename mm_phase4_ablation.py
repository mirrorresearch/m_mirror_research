import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

class ExpertGatedNet(nn.Module):
    def __init__(self, semantic_dim=768, financial_dim=8):
        super().__init__()
        self.s_tower = nn.Sequential(nn.Linear(semantic_dim, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 16))
        self.f_tower = nn.Sequential(nn.Linear(financial_dim, 32), nn.ReLU(), nn.Linear(32, 16))
        self.gate = nn.Sequential(nn.Linear(financial_dim + 1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(16 + 16 + 1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
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

def run_ablation():
    device = torch.device('cuda')
    model = ExpertGatedNet().to(device)
    model.load_state_dict(torch.load("./mm_phase2_expert_best.pth"))
    model.eval()

    data = torch.load("./val2022_phase2_sector.pt", weights_only=True)
    initial_state = data['x'].to(device)

    traj_full = []
    traj_ablated = []
    
    steps = 50
    lr = 0.02
    sigma = 0.01

    state_full = initial_state.clone()
    state_ablated = initial_state.clone()

    for t in range(steps):

        state_full.requires_grad_(True)
        p1, g1 = model(state_full)
        pot1 = -torch.log(1 - p1 + 1e-7)
        model.zero_grad()
        pot1.sum().backward()
        f1 = -state_full.grad
        with torch.no_grad():
            ke1 = (torch.sum(f1[:, 769:]**2, dim=1, keepdim=True) * (1 - g1)).mean().item()
            state_full[:, 769:] += lr * f1[:, 769:] + torch.randn_like(f1[:, 769:]) * sigma
            state_full.grad.zero_()
        traj_full.append(ke1)


        state_ablated.requires_grad_(True)
        p2, g2 = model(state_ablated)
        pot2 = -torch.log(1 - p2 + 1e-7)
        model.zero_grad()
        pot2.sum().backward()
        f2 = -state_ablated.grad
        with torch.no_grad():

            ke2 = torch.sum(f2[:, 769:]**2, dim=1).mean().item()
            state_ablated[:, 769:] += lr * f2[:, 769:] + torch.randn_like(f2[:, 769:]) * sigma
            state_ablated.grad.zero_()
        traj_ablated.append(ke2)

    df = pd.DataFrame({"step": range(steps), "Full_Model": traj_full, "Ablated_No_Damping": traj_ablated})
    df.to_csv("./ablation_trajectory.csv", index=False)


if __name__ == "__main__":
    run_ablation()