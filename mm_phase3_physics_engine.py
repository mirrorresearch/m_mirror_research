import os
from pathlib import Path

import numpy as np
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


class MarketMirrorPhysicsEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = ExpertGatedNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def compute_causal_dynamics(self, x_batch):
        x = x_batch.clone().detach().to(self.device)
        x.requires_grad_(True)
        prob, gate_w = self.model(x)
        eps = 1e-7
        potential = -torch.log(1.0 - prob + eps)
        self.model.zero_grad()
        grad_outputs = torch.ones_like(potential)
        gradients = torch.autograd.grad(
            outputs=potential,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        forces = -gradients
        financial_forces = forces[:, 769:]
        return {
            "potential": potential.detach().cpu().numpy(),
            "full_force": forces.detach().cpu().numpy(),
            "financial_force": financial_forces.detach().cpu().numpy(),
            "gate_weight": gate_w.detach().cpu().numpy(),
            "prob": prob.detach().cpu().numpy(),
        }


if __name__ == "__main__":
    test_input = torch.randn(10, 777)
    model_path = Path(os.getenv("MODEL_PATH", "./mm_phase2_expert_best.pth"))

    if model_path.exists():
        engine = MarketMirrorPhysicsEngine(model_path)
        dynamics = engine.compute_causal_dynamics(test_input)
        force_magnitude = np.linalg.norm(dynamics["financial_force"], axis=1)
        print(f"Potential mean: {dynamics['potential'].mean():.4f}")
        print(f"Financial force shape: {dynamics['financial_force'].shape}")
        print(f"Gate weight sample: {dynamics['gate_weight'][:5].flatten()}")
        print(f"Force magnitude mean: {force_magnitude.mean():.4f}")
    else:
        print(f"Model file not found: {model_path}")