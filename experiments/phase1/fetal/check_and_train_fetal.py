"""
Train the black-box 64-32-3 MLP in DataInterface space for Fetal Health.

Paper settings:
  - 60/20/20 train/validation/test split
  - 64 -> 32 MLP, 3 outputs, CrossEntropyLoss / softmax
  - Target class: 0 (Normal)
  - Paper accuracy: train 0.92 / val 0.93 / test 0.91
"""
import os
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Dynamic project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while PROJECT_ROOT.parent != PROJECT_ROOT and not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arc.Multiclass_Architecture import Model
from data.Data_Interface import DataInterface

SEED = 42
MODEL_PATH = PROJECT_ROOT / "pytorch_model_fetal.pth"
P1_RESULTS_DIR = PROJECT_ROOT / "phase1" / "results" / "fetal"
META_PATH = P1_RESULTS_DIR / "model_fetal_meta.txt"

CONT_COLS = ["LBE","LB","AC","FM","UC","ASTV","MSTV","ALTV","MLTV","DL","DS","DP",
             "Width","Min","Max","Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"]
CAT_COLS = ["A","B","C","D","E","AD","DE","LD","FS","SUSP"]


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main():
    seed_all(SEED)
    df = pd.read_csv(PROJECT_ROOT / "datasets" / "fetal.csv")
    split = pd.read_csv(PROJECT_ROOT / "datasets" / "fetal_split.csv", index_col="row")["split"]
    
    d = DataInterface(dataframe=df, target="NSP", continuous_features=CONT_COLS)
    Z = d.norm_encoded_features
    y = df["NSP"]

    tr_idx = split[split == "train"].index
    va_idx = split[split == "val"].index
    te_idx = split[split == "test"].index

    Xtr = torch.tensor(Z.loc[tr_idx].values, dtype=torch.float32)
    ytr = torch.tensor(y.loc[tr_idx].values, dtype=torch.long)
    Xva = torch.tensor(Z.loc[va_idx].values, dtype=torch.float32)
    yva = torch.tensor(y.loc[va_idx].values, dtype=torch.long)
    Xte = torch.tensor(Z.loc[te_idx].values, dtype=torch.float32)
    yte = torch.tensor(y.loc[te_idx].values, dtype=torch.long)

    model = Model(Xtr.shape[1], 3)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    epochs = 120
    batch_size = 64
    n_batches = int(np.ceil(len(Xtr) / batch_size))

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(Xtr))
        model.train()
        for b in range(n_batches):
            b_idx = perm[b * batch_size : (b + 1) * batch_size]
            opt.zero_grad()
            out = model(Xtr[b_idx])
            loss = crit(out, ytr[b_idx])
            loss.backward()
            opt.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        acc_tr = (model(Xtr).argmax(dim=1) == ytr).float().mean().item()
        acc_va = (model(Xva).argmax(dim=1) == yva).float().mean().item()
        acc_te = (model(Xte).argmax(dim=1) == yte).float().mean().item()

    print("=" * 60)
    print(f"Model Accuracies:")
    print(f"  Train:      {acc_tr:.4f}  (paper: 0.92)")
    print(f"  Validation: {acc_va:.4f}  (paper: 0.93)")
    print(f"  Test:       {acc_te:.4f}  (paper: 0.91)")
    print("=" * 60)

    torch.save(model.state_dict(), str(MODEL_PATH))
    os.makedirs(P1_RESULTS_DIR, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        f.write(f"train_acc={acc_tr:.6f}\nval_acc={acc_va:.6f}\ntest_acc={acc_te:.6f}\n")
    print(f"Saved {MODEL_PATH} and {META_PATH}")

    # Legacy mirror if results_fetal exists
    legacy_dir = PROJECT_ROOT / "results_fetal"
    if legacy_dir.exists():
        with open(legacy_dir / "model_fetal_meta.txt", "w", encoding="utf-8") as f:
            f.write(f"train_acc={acc_tr:.6f}\nval_acc={acc_va:.6f}\ntest_acc={acc_te:.6f}\n")


if __name__ == "__main__":
    main()
