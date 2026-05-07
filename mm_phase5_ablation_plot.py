import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ablation():
    df = pd.read_csv("./ablation_trajectory.csv")
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    plt.plot(df['step'], df['Ablated_No_Damping'], color='red', linestyle='--', linewidth=2, label='Ablated (No Deception Damping)')
    plt.plot(df['step'], df['Full_Model'], color='blue', linewidth=3, label='MarketMirror (Full Expert Gated)')

    plt.fill_between(df['step'], df['Full_Model'], df['Ablated_No_Damping'], color='gray', alpha=0.1, label='Energy Dissipation Gap')

    plt.title('Ablation Study: Impact of Expert Gating on Systemic Risk', fontsize=15)
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Systemic Kinetic Energy $G^*(t)$', fontsize=12)
    plt.legend()
    
    plt.savefig("./ablation_comparison_plot.png", dpi=300)

if __name__ == "__main__":
    plot_ablation()