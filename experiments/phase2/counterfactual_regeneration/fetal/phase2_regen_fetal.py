"""
Phase 2, Test T3: Fixed-feature regeneration for Fetal Health.

For every query, we fix:
  - top-k features of each ranking (GradCFA, Shapley, Path-IG, Random)
  - specifically feature 'D' and feature 'A' (as reported in the paper: Table VIII, D=0.24, A=0.45)
and measure counterfactual validity under GradCFA generation.

Usage:
  python phase2_regen_fetal.py --limit 5 --k 1
  python phase2_regen_fetal.py --fixed-paper
"""
import argparse
import contextlib
import io
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while PROJECT_ROOT.parent != PROJECT_ROOT and not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import phase2_attribution_fetal as p2
from cf.GradCFA_MC import GradCFA
from model.Model_Interface_MC import ModelInterface
from data.Data_Interface import DataInterface

SEED = 42
OUT_DIR = str(PROJECT_ROOT / "phase2" / "results" / "fetal")
OUT_CSV = os.path.join(OUT_DIR, "phase2_regen.csv")
SUMMARY_TXT = os.path.join(OUT_DIR, "phase2_regen_summary.txt")


def run_paper_fixed_test(limit=None):
    """Test fixing specifically D and A across queries (paper reported D: 0.24, A: 0.45)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(PROJECT_ROOT / "datasets" / "fetal.csv")
    split = pd.read_csv(PROJECT_ROOT / "datasets" / "fetal_split.csv", index_col="row")["split"]
    te_idx = split[split == "test"].index

    d = DataInterface(dataframe=df, target="NSP", continuous_features=p2.CONT_COLS)
    m = ModelInterface(model_path=p2.MODEL_PATH, input_size=42, output_size=3)
    m.load_model()

    # Load queries
    csv_p = str(PROJECT_ROOT / "phase1" / "results" / "fetal" / "full_C.csv") if os.path.exists(str(PROJECT_ROOT / "phase1" / "results" / "fetal" / "full_C.csv")) else str(PROJECT_ROOT / "phase1" / "results" / "fetal" / "full_A.csv")
    cfs_A = pd.read_csv(csv_p)
    query_rids = cfs_A[cfs_A["seed"] == SEED]["row_id"].unique()
    if limit:
        query_rids = query_rids[:limit]

    print(f"\n--- Running Fixed Feature Test for Paper Top Features ('D' and 'A') on {len(query_rids)} queries ---")
    results = []

    for feat in ["D", "A"]:
        print(f"\nTesting with immutable feature: {feat}")
        validities = []
        for i, rid in enumerate(query_rids, 1):
            ql = df.loc[rid].drop("NSP").to_list()
            exp = GradCFA(d, m)
            exp.plot_loss_curve = lambda *args, **kwargs: None
            exp.plot_feature_importance = lambda *args, **kwargs: None

            with contextlib.redirect_stdout(io.StringIO()):
                q, cfs = exp.generate_counterfactuals(
                    ql, total_CFs=5, desired_value=0, problem_type="multiclass",
                    immutable_features=[feat],
                    proximity_weight=0.5, sparsity_weight=0.5, plausibility_weight=0.5, diversity_weight=0.5,
                    penalty_scale=0.1, perturb_scale=0.5,
                    proximity_threshold=0.4, sparsity_threshold=0.4,
                    plausibility_threshold=0.5, diversity_threshold=0.9,
                    high_loss_threshold=1e9, loss_converge_maxiter=2,
                    min_iter=500, max_iter=2000, init_near_query_instance=False
                )

            with torch.no_grad():
                preds = m.predict(torch.tensor(cfs, dtype=torch.float32)).argmax(dim=1).numpy()
                val_frac = float((preds == 0).mean())

            validities.append(val_frac)
            results.append(dict(row_id=rid, fixed_feature=feat, validity=val_frac))
            print(f"  [{i}/{len(query_rids)}] row {rid} | fixed {feat} | validity: {val_frac:.2f}")

        mean_val = np.mean(validities)
        paper_val = 0.24 if feat == "D" else 0.45
        print(f"--> Fixed {feat}: Mean validity = {mean_val:.4f} (Paper: {paper_val})")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "paper_fixed_feature_validity.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="test queries limit")
    parser.add_argument("--fixed-paper", action="store_true", default=True)
    args = parser.parse_args()

    run_paper_fixed_test(args.limit)

