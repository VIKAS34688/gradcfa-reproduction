"""
Why does the numpy copy of the model disagree with the stored validity numbers?

Run from the repository root (next to phase2_attribution.py):
    python diagnose_model.py

For the first 5 saved counterfactual sets it prints, side by side, the validity stored in the results
CSV, the validity recomputed with the real torch model, and with the numpy copy, plus the raw probabilities.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("src"))
from arc.Binary_Architecture import Model
import phase2_attribution as p2

MODEL = "pytorch_model_di.pth"
sd = torch.load(MODEL, map_location="cpu")
width = sd["fc1.weight"].shape[1]
tm = Model(width)
tm.load_state_dict(sd)
tm.eval()
nm = p2.MLP(p2.load_weights(MODEL))

sets = p2.load_sets("results_di/cfs_C_all", 42, limit=5)
full = pd.read_csv("results_di/full_C_all.csv")
full = full[full.seed == 42].set_index("row_id")
print("model input width:", width)

for r, (q, cfs) in sets.items():
    with torch.no_grad():
        pq_t = tm(torch.tensor(q, dtype=torch.float32)[None, :]).item()
        pc_t = tm(torch.tensor(cfs, dtype=torch.float32)).view(-1).numpy()
    pc_n = nm.prob(cfs)
    target = 1 - int(pq_t >= 0.5)
    v_t = float(((pc_t >= 0.5).astype(int) == target).mean())
    v_n = float(((pc_n >= 0.5).astype(int) == target).mean())
    print(f"\nrow {r}: stored valid_frac = {full.loc[r, 'valid_frac']:.2f} | torch now = {v_t:.2f} | numpy = {v_n:.2f} | target class = {target}")
    print("   query prob   torch / numpy :", round(pq_t, 6), "/", round(float(nm.prob(q)[0]), 6))
    print("   cf probs     torch :", np.round(pc_t, 6))
    print("   cf probs     numpy :", np.round(pc_n, 6))
    print("   max |torch - numpy| =", float(np.abs(pc_t - pc_n).max()),
          "| closest cf to the 0.5 boundary:", float(np.abs(pc_t - 0.5).min()))
