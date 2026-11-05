"""
Step 1: measure a problem in the original notebook pipeline.
Step 2: retrain the black-box model in the encoding space that GradCFA really feeds it.

Problem: binary_notebook trains the model on pd.get_dummies(...) columns in the order
['Gender','Married','BankCustomer','Industry','Ethnicity','PriorDefault','CreditScore','Employed',...]
and z-scores EVERY column (including the one-hot columns). But DataInterface (used to generate
counterfactuals and to evaluate them) orders the groups ...,'Employed','CreditScore',... and only
normalises the 4 continuous features. So at counterfactual time the model receives columns in a
different order (25 of 61 positions differ) and with different scaling than it was trained on.

Run from repository root or phase1/code/:
    python check_and_train_di.py

It writes pytorch_model_di.pth and phase1/results/credit/model_di_meta.txt.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Dynamic project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while PROJECT_ROOT.parent != PROJECT_ROOT and not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arc.Binary_Architecture import Model
from data.Data_Interface import DataInterface

SEED = 42
OLD_MODEL = PROJECT_ROOT / "pytorch_model.pth"
NEW_MODEL = PROJECT_ROOT / "pytorch_model_di.pth"
P1_CREDIT_DIR = PROJECT_ROOT / "phase1" / "results" / "credit"
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


seed_all(SEED)
df = pd.read_csv(PROJECT_ROOT / "datasets" / "credit.csv")

# ----------------- Step 1: Reproduce notebook's mismatched pipeline -----------------
print("=== Step 1: check old model ===")
df_nb = df.copy()
for col in CAT_NB:
    le = LabelEncoder()
    df_nb[col] = le.fit_transform(df_nb[col])
X_nb = pd.get_dummies(df_nb.drop(columns=['Approved']), columns=CAT_NB)
X_nb = pd.DataFrame(StandardScaler().fit_transform(X_nb), columns=X_nb.columns)
y_nb = df_nb['Approved']

tr, te = train_test_split(df.index, test_size=0.2, random_state=SEED)
if OLD_MODEL.exists():
    old = Model(X_nb.shape[1])
    old.load_state_dict(torch.load(OLD_MODEL, map_location="cpu"))
    old.eval()
    acc_nb = float((predict(old, X_nb.loc[te].values) == y_nb.loc[te].values).mean())
    print(f"Old model test acc in its OWN training space: {acc_nb}")
else:
    print(f"(No {OLD_MODEL.name} to check)")

# ----------------- Step 2: Train properly in DataInterface space -----------------
print("\n=== Step 2: retrain model in DataInterface space ===")
d = DataInterface(dataframe=df, target="Approved", continuous_features=CONT)
Z = d.norm_encoded_features
y = df['Approved']

Xtr = torch.tensor(Z.loc[tr].values, dtype=torch.float32)
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
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
acc_new = float((predict(model, Z.loc[te].values) == yte).mean())
print(f"NEW model test accuracy (DataInterface space): {acc_new:.4f}")

torch.save(model.state_dict(), str(NEW_MODEL))
os.makedirs(P1_CREDIT_DIR, exist_ok=True)
with open(P1_CREDIT_DIR / "model_di_meta.txt", "w", encoding="utf-8") as f:
    f.write(f"{acc_new!r}\n")
print(f"Saved {NEW_MODEL} and {P1_CREDIT_DIR / 'model_di_meta.txt'}")

legacy_dir = PROJECT_ROOT / "results_di"
if legacy_dir.exists():
    with open(legacy_dir / "model_di_meta.txt", "w", encoding="utf-8") as f:
        f.write(f"{acc_new!r}\n")
