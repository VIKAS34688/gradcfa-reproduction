"""
Phase 2 Figure Generator: Faithfulness Tests and Feature Importance.
Generates:
  - phase2/results/fetal/fig2_phase2_faithfulness.png
  - phase2/results/fetal/fig3_feature_importance.png
"""
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dynamic project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while PROJECT_ROOT.parent != PROJECT_ROOT and not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

P1_RESULTS_DIR = PROJECT_ROOT / "phase1" / "results" / "fetal"
P2_RESULTS_DIR = PROJECT_ROOT / "phase2" / "results" / "fetal"
os.makedirs(P2_RESULTS_DIR, exist_ok=True)

def generate_phase2_figures():
    # 1. Feature Importance Plot (fig3_feature_importance.png)
    fa_csv = P1_RESULTS_DIR / "fa_C.csv"
    if not fa_csv.exists():
        fa_csv = PROJECT_ROOT / "results_fetal" / "fa_C.csv"
        
    if fa_csv.exists():
        fa_df = pd.read_csv(fa_csv)
        top_feats = fa_df.groupby("feature")["score"].mean().sort_values(ascending=True).tail(15)

        plt.figure(figsize=(7, 5), dpi=150)
        plt.barh(top_feats.index, top_feats.values, color="#2b5c8f", edgecolor="black", height=0.65)
        plt.title("Top 15 Attributed Features (GradCFA Config C, 71 queries)", fontsize=11, fontweight="bold")
        plt.xlabel("Mean Attribution Score", fontsize=10)
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_fig3 = P2_RESULTS_DIR / "fig3_feature_importance.png"
        plt.savefig(out_fig3)
        plt.close()
        print("Generated Phase 2 Figure:", out_fig3)

    # 2. Faithfulness T1 & T2 Curves (fig2_phase2_faithfulness.png)
    k_vals = np.arange(1, 11)
    t1_grad = [0.0116, 0.0247, 0.0544, 0.0811, 0.1089, 0.1303, 0.1520, 0.1874, 0.2005, 0.2250]
    t1_shap = [0.1704, 0.3231, 0.4141, 0.4643, 0.4878, 0.5069, 0.5181, 0.5218, 0.5243, 0.5272]
    t1_pig  = [0.1665, 0.3138, 0.3873, 0.4422, 0.4770, 0.4954, 0.5099, 0.5193, 0.5277, 0.5309]
    t1_rand = [0.0176, 0.0359, 0.0556, 0.0779, 0.0991, 0.1226, 0.1445, 0.1640, 0.1879, 0.2093]

    t2_grad = [0.0028, 0.0028, 0.0141, 0.0197, 0.0225, 0.0338, 0.0366, 0.0592, 0.0732, 0.0901]
    t2_shap = [0.0169, 0.1042, 0.2028, 0.3268, 0.4479, 0.6085, 0.7324, 0.7972, 0.8394, 0.8451]
    t2_pig  = [0.0169, 0.1014, 0.2310, 0.3972, 0.5296, 0.6873, 0.8141, 0.8873, 0.8930, 0.9014]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    ax1.plot(k_vals, t1_shap, "o-", color="#2ca02c", label="Shapley (Permutation)", linewidth=1.8, markersize=5)
    ax1.plot(k_vals, t1_pig, "s-", color="#1f77b4", label="Path-IG (Ours)", linewidth=1.8, markersize=5)
    ax1.plot(k_vals, t1_rand, "^--", color="#7f7f7f", label="Random", linewidth=1.2, markersize=4)
    ax1.plot(k_vals, t1_grad, "d-", color="#d62728", label="GradCFA (Paper)", linewidth=1.8, markersize=5)
    ax1.set_title("Test T1: Query Class Probability Drop", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Number of Features Ablated (k)", fontsize=9)
    ax1.set_ylabel("Drop in Predicted Class Probability", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8)

    ax2.plot(k_vals, t2_shap, "o-", color="#2ca02c", label="Shapley (Permutation)", linewidth=1.8, markersize=5)
    ax2.plot(k_vals, t2_pig, "s-", color="#1f77b4", label="Path-IG (Ours)", linewidth=1.8, markersize=5)
    ax2.plot(k_vals, t2_grad, "d-", color="#d62728", label="GradCFA (Paper)", linewidth=1.8, markersize=5)
    ax2.set_title("Test T2: CF Invalidity Fraction", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Number of Features Reset to Query (k)", fontsize=9)
    ax2.set_ylabel("Fraction of CFs Rendered Invalid", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_fig2 = P2_RESULTS_DIR / "fig2_phase2_faithfulness.png"
    plt.savefig(out_fig2)
    plt.close()
    print("Generated Phase 2 Figure:", out_fig2)

if __name__ == "__main__":
    generate_phase2_figures()

