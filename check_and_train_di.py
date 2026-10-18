"""
Step 1: measure a problem in the original notebook pipeline.
Step 2: retrain the black-box model in the encoding space that GradCFA really feeds it.

Problem: binary_notebook trains the model on pd.get_dummies(...) columns in the order
['Gender','Married','BankCustomer','Industry','Ethnicity','PriorDefault','CreditScore','Employed',...]
and z-scores EVERY column (including the one-hot columns). But DataInterface (used to generate
counterfactuals and to evaluate them) orders the groups ...,'Employed','CreditScore',... and only
normalises the 4 continuous features. So at counterfactual time the model receives columns in a
different order (25 of 61 positions differ) and with different scaling than it was trained on.

Run from the repository root (needs the old pytorch_model.pth for the comparison in Step 1):

    python check_and_train_di.py

It writes pytorch_model_di.pth and results_di/model_di_meta.txt.
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.abspath("src"))
from arc.Binary_Architecture import Model
from data.Data_Interface import DataInterface

SEED = 42
OLD_MODEL = "pytorch_model.pth"
NEW_MODEL = "pytorch_model_di.pth"
CAT_NB = ['Gender', 'Married', 'BankCustomer', 'Industry', 'Ethnicity', 'PriorDefault',
          'CreditScore', 'Employed', 'DriversLicense', 'Citizen']
CONT = ['Age', 'Debt', 'YearsEmployed', 'Income']


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def predict(model, arr):
    with torch.no_grad():
        return (model(torch.tensor(np.asarray(arr, dtype=np.float32))) >= 0.5).long().view(-1).numpy()


# ---------- data, built exactly like the notebook ----------
df = pd.read_csv("datasets/credit.csv").drop(columns="ZipCode")
for col in ["Industry", "Ethnicity", "Citizen"]:
    df[col] = LabelEncoder().fit_transform(df[col])
y = df["Approved"]
X_nb = pd.get_dummies(df.drop(columns=["Approved"]), columns=CAT_NB)          # notebook column order
X_train_nb, X_test_nb, y_train, y_test = train_test_split(X_nb, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train_nb)

d = DataInterface(dataframe=df, target="Approved", continuous_features=CONT)
Z = d.norm_encoded_features                                                   # DataInterface space
n_diff = sum(a != b for a, b in zip(X_nb.columns, Z.columns))

# ---------- Step 1: how bad is the mismatch for the OLD model? ----------
print("=" * 70)
print(f"Columns: {Z.shape[1]} | positions where the two column orders differ: {n_diff}")
if os.path.exists(OLD_MODEL):
    old = Model(X_nb.shape[1])
    old.load_state_dict(torch.load(OLD_MODEL, map_location="cpu"))
    old.eval()
    te = X_test_nb.index
    acc_nb = (predict(old, scaler.transform(X_test_nb)) == y_test.values).mean()
    acc_di = (predict(old, Z.loc[te].to_numpy(dtype=float)) == y_test.values).mean()
    p_nb = predict(old, scaler.transform(X_nb))
    p_di = predict(old, Z.to_numpy(dtype=float))
    print(f"OLD model, test accuracy in the notebook space      : {acc_nb:.4f}")
    print(f"OLD model, test accuracy in the DataInterface space : {acc_di:.4f}")
    print(f"OLD model, same prediction in both spaces (all rows): {(p_nb == p_di).mean():.4f}")
else:
    print("Old model file not found, skipping the comparison.")

# ---------- Step 2: retrain in the DataInterface space ----------
print("=" * 70)
print("Training a new model in the DataInterface space (same architecture and settings)...")
seed_all(SEED)
tr, te = X_train_nb.index, X_test_nb.index
Xtr = torch.tensor(Z.loc[tr].to_numpy(dtype=np.float32))
Xte = torch.tensor(Z.loc[te].to_numpy(dtype=np.float32))
ytr = torch.tensor(y.loc[tr].values, dtype=torch.float32).view(-1, 1)
yte = y.loc[te].values

model = Model(Z.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(model(Xtr), ytr)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

model.eval()
acc_new = float((predict(model, Xte.numpy()) == yte).mean())
print(f"NEW model test accuracy (DataInterface space): {acc_new}")
torch.save(model.state_dict(), NEW_MODEL)
os.makedirs("results_di", exist_ok=True)
with open("results_di/model_di_meta.txt", "w", encoding="utf-8") as f:
    f.write(f"{acc_new!r}\n")
print(f"Saved {NEW_MODEL} and results_di/model_di_meta.txt")
