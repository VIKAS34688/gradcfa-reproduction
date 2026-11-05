"""Build datasets/fetal.csv and the 60/20/20 split from the UCI CTG file."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while PROJECT_ROOT.parent != PROJECT_ROOT and not (PROJECT_ROOT / "datasets").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SEED = 42
raw_path = PROJECT_ROOT / "datasets" / "CTG.xls"
raw = pd.read_excel(raw_path, sheet_name="Raw Data")

CONT = ["LBE","LB","AC","FM","UC","ASTV","MSTV","ALTV","MLTV","DL","DS","DP",
        "Width","Min","Max","Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"]  # 22
CAT  = ["A","B","C","D","E","AD","DE","LD","FS","SUSP"]                                   # 10
missing = [c for c in CONT + CAT + ["NSP"] if c not in raw.columns]
assert not missing, f"columns not found: {missing}\nAvailable: {list(raw.columns)}"

df = raw[CONT + CAT + ["NSP"]].dropna().copy()      # drop footer/empty rows, as the paper drops missing values
df["NSP"] = df["NSP"].astype(int) - 1                # 1,2,3 -> 0,1,2
for c in CAT:
    df[c] = df[c].astype(int)
df = df.reset_index(drop=True)

print("rows:", len(df), "(paper: 2126)")
print("features:", len(CONT) + len(CAT), "(paper: 32)")
print("class counts:", df["NSP"].value_counts().sort_index().to_dict(), "(expected about 1655 / 295 / 176)")

idx = np.arange(len(df))
tr, tmp = train_test_split(idx, test_size=0.4, random_state=SEED, stratify=df["NSP"])
va, te  = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=df.loc[tmp, "NSP"])
split = pd.Series("train", index=idx); split[va] = "val"; split[te] = "test"
df.to_csv(PROJECT_ROOT / "datasets" / "fetal.csv", index=False)
split.rename("split").to_csv(PROJECT_ROOT / "datasets" / "fetal_split.csv", index_label="row")
print("split sizes:", split.value_counts().to_dict())
