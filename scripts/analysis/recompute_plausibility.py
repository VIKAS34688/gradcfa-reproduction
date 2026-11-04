"""
Plausibility sensitivity analysis. No counterfactual generation is re-run: it reads the saved
counterfactuals (results_di/cfs_*/s<seed>_r<row>.npz) and recomputes plausibility with several
possible definitions, because the paper does not state k and its Eq. 5 is ambiguous.

Run from the repository root:
    python recompute_plausibility.py                      # uses results_di/cfs_C_all and cfs_A_all
    python recompute_plausibility.py results_di/cfs_C_all # one folder

Definitions (all use L1 distance in the DataInterface encoding space, over the 690 observed rows):
  eq5_k      : the paper's Eq. 5 as written: distances to the k nearest observed rows, scaled by the
               min and max among those k distances (cannot go below 1/k). Reported for k = 2, 3, 5, 10.
  raw_k      : plain mean distance to the k nearest rows (what the repository computes).
  rel_k      : raw_k divided by the typical k-NN distance between real rows (1.0 = as plausible as real data).
  g01_k      : raw_k scaled to 0-1 by the smallest and largest counterfactual-to-data distance in the query.
Lower is better for every definition.
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Locate the repository root regardless of this script's subdirectory.
from pathlib import Path

def _find_project_root(start):
    for candidate in (start, *start.parents):
        if (candidate / "README.md").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate the GradCFs repository root.")

PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)

# Preserve the repository's existing relative data/results paths.
import os
import sys
os.chdir(PROJECT_ROOT)

for _path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "src",
):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from data.Data_Interface import DataInterface

KS = [2, 3, 5, 10]
PAPER_PLAUS = 0.37


def knn_dists(dist, k):
    """dist: (n_cf, n_obs). Returns the k smallest distances per counterfactual, ascending."""
    return np.sort(dist, axis=1)[:, :k]


def plaus_definitions(dist, typical):
    out = {}
    for k in KS:
        vals = knn_dists(dist, k)
        dmin, dmax = vals.min(axis=1, keepdims=True), vals.max(axis=1, keepdims=True)
        out[f"eq5_k{k}"] = float(((vals - dmin) / (dmax - dmin + 1e-8)).mean())
        raw = float(vals.mean())
        out[f"raw_k{k}"] = raw
        out[f"rel_k{k}"] = raw / typical[k]
        out[f"g01_k{k}"] = (raw - dist.min()) / (dist.max() - dist.min() + 1e-8)
    return out


def main(folders):
    df = pd.read_csv("data/raw/credit.csv").drop(columns="ZipCode")
    for col in ["Industry", "Ethnicity", "Citizen"]:
        df[col] = LabelEncoder().fit_transform(df[col])
    d = DataInterface(dataframe=df, target="Approved",
                      continuous_features=["Age", "Debt", "YearsEmployed", "Income"])
    obs = d.norm_encoded_features.to_numpy(dtype=np.float64)

    # typical k-NN distance between real rows (excluding the row itself)
    pair = cdist(obs, obs, "cityblock")
    np.fill_diagonal(pair, np.inf)
    typical = {k: float(np.sort(pair, axis=1)[:, :k].mean()) for k in KS}

    summary = []
    for folder in folders:
        rows = []
        for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
            cfs = np.load(path)["cfs"].astype(np.float64)
            dist = cdist(cfs, obs, "cityblock")
            r = plaus_definitions(dist, typical)
            r["file"] = os.path.basename(path)
            rows.append(r)
        res = pd.DataFrame(rows)
        mean = res.drop(columns="file").mean()
        mean.name = os.path.basename(folder.rstrip("/\\"))
        summary.append(mean)
        print(f"{mean.name}: {len(res)} counterfactual sets")
    table = pd.DataFrame(summary).T.round(3)
    table.insert(0, "paper", [PAPER_PLAUS if n == "eq5_k3" else np.nan for n in
                              [i if i.startswith("eq5_k3") else "" for i in table.index]])
    print("\n" + table.to_string())
    os.makedirs("results_di", exist_ok=True)
    table.to_csv("results_di/plausibility_sensitivity.csv")
    print("\nSaved results_di/plausibility_sensitivity.csv")
    print("Typical real-data k-NN distances:", {k: round(v, 3) for k, v in typical.items()})


if __name__ == "__main__":
    folders = sys.argv[1:] or ["results_di/cfs_C_all", "results_di/cfs_A_all"]
    main(folders)
