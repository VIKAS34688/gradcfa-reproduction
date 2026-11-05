"""Shared code for the Obesity pipeline (data, encoding space, 60/20/20 split). Used by obesity_train.py and obesity_eval.py."""
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath("src"))
from data.Data_Interface import DataInterface

SEED = 42
# Same column choice as the authors' mc_notebook.ipynb (8 numeric + 8 categorical).
# NOTE: the paper text says 6 continuous + 10 categorical; we follow the authors' code and document the difference.
CONT = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
CAT = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'FHWO', 'CAEC', 'MTRANS']
MODEL_PATH = "pytorch_model_obesity.pth"
META_PATH = "results_obesity/model_obesity_meta.txt"
RESULTS_DIR = "results_obesity"
TARGET_CLASS = 1          # normal weight (alphabetical label encoding: 0 Insufficient, 1 Normal, 2 Ob.I, 3 Ob.II, 4 Ob.III, 5 Over.I, 6 Over.II)
N_CLASSES = 7


def seed_all(s):
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)


def load_data():
    """Returns df (label encoded, target last), DataInterface d, encoded space Z (what the model is trained on) and y."""
    df = pd.read_csv("datasets/obesity.csv")
    df = df.rename(columns={'NObeyesdad': 'target', 'family_history_with_overweight': 'FHWO'})
    df = df.dropna().reset_index(drop=True)          # paper: drop rows with missing values (none expected)
    for c in CAT + ['target']:
        df[c] = LabelEncoder().fit_transform(df[c])
    df = df[[c for c in df.columns if c != 'target'] + ['target']]   # DataInterface expects the target LAST
    d = DataInterface(dataframe=df, target='target', continuous_features=CONT)
    Z = d.norm_encoded_features                      # DataInterface space: the one GradCFA really feeds the model
    return df, d, Z, df['target']


def split(Z, y):
    """60/20/20 train/validation/test, random_state=42 (no stratification)."""
    Z_tmp, Z_test, y_tmp, y_test = train_test_split(Z, y, test_size=0.2, random_state=SEED)
    Z_train, Z_val, y_train, y_val = train_test_split(Z_tmp, y_tmp, test_size=0.25, random_state=SEED)
    return Z_train, Z_val, Z_test, y_train, y_val, y_test
