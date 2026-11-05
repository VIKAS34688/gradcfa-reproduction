"""
Trains the 64-32 multi-class model on the Obesity data IN THE DataInterface ENCODING SPACE (60/20/20 split, seed 42).
Run from the repository root:   python obesity_train.py            (optional: --epochs 200)
Writes pytorch_model_obesity.pth and results_obesity/model_obesity_meta.txt.
Choose --epochs on the VALIDATION accuracy only, never on the test accuracy.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from obesity_common import *
from arc.Multiclass_Architecture import Model

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=200)   # the authors' notebook uses 200; the paper does not state it
args = ap.parse_args()

raw = pd.read_csv("datasets/obesity.csv")
classes = sorted(raw["NObeyesdad"].unique())
print("Label mapping:", dict(zip(classes, range(len(classes)))))
assert classes[1] == "Normal_Weight" and len(classes) == N_CLASSES, "label mapping is not what the plan expects"

df, d, Z, y = load_data()
print("Rows:", len(df), "| encoded input size:", Z.shape[1])
Z_train, Z_val, Z_test, y_train, y_val, y_test = split(Z, y)
print("Split sizes train/val/test:", len(Z_train), len(Z_val), len(Z_test))


def T(z):
    return torch.tensor(z.to_numpy(dtype=np.float32))


Xtr, Xva, Xte = T(Z_train), T(Z_val), T(Z_test)
ytr = torch.tensor(y_train.values, dtype=torch.long)

seed_all(SEED)
model = Model(Z.shape[1], N_CLASSES)
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
for ep in range(args.epochs):
    opt.zero_grad()
    loss = crit(model(Xtr), ytr)
    loss.backward()
    opt.step()
    if (ep + 1) % 20 == 0:
        print(f"Epoch [{ep + 1}/{args.epochs}], Loss: {loss.item():.5f}")


def acc(X, yy):
    with torch.no_grad():
        model.eval()
        return (model(X).argmax(1).numpy() == yy.values).mean()


a_tr, a_va, a_te = acc(Xtr, y_train), acc(Xva, y_val), acc(Xte, y_test)
print(f"\nAccuracy train / val / test: {a_tr:.4f} / {a_va:.4f} / {a_te:.4f}   (paper: 0.90 / 0.84 / 0.85)")
torch.save(model.state_dict(), MODEL_PATH)
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(META_PATH, "w", encoding="utf-8") as f:
    f.write(f"{float(a_te)!r}\n")
    f.write(f"epochs={args.epochs} train={a_tr:.6f} val={a_va:.6f} test={a_te:.6f}\n")
print("Saved", MODEL_PATH, "and", META_PATH)
