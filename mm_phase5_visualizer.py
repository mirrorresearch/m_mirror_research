from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_risk_trajectory(csv_path, output_path):
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", palette="muted")
    plt.plot(
        df["step"],
        df["G_star"],
        color="#1f77b4",
        linewidth=3,
        label="Risk trajectory",
        zorder=3,
    )
    plt.scatter(df["step"][::5], df["G_star"][::5], color="#1f77b4", s=50, edgecolors="white", zorder=4)
    initial_energy = df["G_star"].iloc[0]
    final_energy = df["G_star"].iloc[-1]
    plt.annotate(
        f"Initial state\n$G^*_0 = {initial_energy:.4f}$",
        xy=(0, initial_energy),
        xytext=(5, initial_energy + 0.001),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        fontsize=10,
        fontweight="bold",
    )
    plt.annotate(
        rf"Final state\n$G^*_\infty = {final_energy:.4f}$",
        xy=(len(df) - 1, final_energy),
        xytext=(35, final_energy + 0.003),
        arrowprops=dict(facecolor="green", shrink=0.05, width=1, headwidth=5),
        fontsize=10,
        color="green",
        fontweight="bold",
    )
    plt.axvspan(0, 20, color="red", alpha=0.05, label="Early phase")
    plt.axvspan(30, 50, color="green", alpha=0.05, label="Late phase")
    plt.title("Risk trajectory over simulation steps", fontsize=16, pad=15)
    plt.xlabel("Simulation step ($t$)", fontsize=13)
    plt.ylabel("Energy proxy $G^*(t)$", fontsize=13)
    plt.ylim(0, max(df["G_star"]) * 1.2)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    csv_input = "./risk_trajectory_2022.csv"
    png_output = "./marketmirror_risk_evolution.png"
    plot_risk_trajectory(csv_input, png_output)