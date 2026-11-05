"""
Phase 2: Is GradCFA's feature attribution trustworthy? (Fetal Health dataset)

Computes offline from existing results:
  - Exact/Permutation Shapley values over the 32 original features
  - Path-IG: integrated gradients from query to counterfactuals
  - Agreement between rankings: Spearman, Kendall, top-1, top-3
  - Faithfulness test T1: replace top-k features by background, measure drop in query predicted class prob
  - Faithfulness test T2: reset top-k features in CF to query, measure drop in target class 0 prob

Usage:
  python phase2_attribution_fetal.py --limit 5
  python phase2_attribution_fetal.py
"""
import argparse
import glob
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

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
from data.Data_Interface import DataInterface

MODEL_PATH = str(PROJECT_ROOT / "pytorch_model_fetal.pth")
CONT_COLS = ["LBE","LB","AC","FM","UC","ASTV","MSTV","ALTV","MLTV","DL","DS","DP",
             "Width","Min","Max","Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"]
CAT_COLS = ["A","B","C","D","E","AD","DE","LD","FS","SUSP"]

N_BG = 50
N_RANDOM = 30
IG_STEPS = 64
K_MAX = 10
BOOT = 2000
TARGET_CLASS = 0
DEFAULT_OUT = str(PROJECT_ROOT / "phase2" / "results" / "fetal")


def load_weights(path):
    sd = torch.load(path, map_location="cpu")
    keys = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
    return [sd[k].detach().numpy().astype(np.float64) for k in keys]


class MLP:
    def __init__(self, W):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = W

    def predict_class(self, X):
        X = np.atleast_2d(X)
        h1 = np.maximum(X @ self.W1.T + self.b1, 0)
        h2 = np.maximum(h1 @ self.W2.T + self.b2, 0)
        logits = h2 @ self.W3.T + self.b3
        return np.argmax(logits, axis=1)

    def prob(self, X, cls=TARGET_CLASS):
        """Softmax probability for class cls."""
        X = np.atleast_2d(X)
        h1 = np.maximum(X @ self.W1.T + self.b1, 0)
        h2 = np.maximum(h1 @ self.W2.T + self.b2, 0)
        logits = h2 @ self.W3.T + self.b3
        exp_z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return probs[:, cls]

    def grad(self, X, cls=TARGET_CLASS):
        """Gradient of class cls softmax probability wrt input X."""
        X = np.atleast_2d(X)
        z1 = X @ self.W1.T + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W2.T + self.b2
        h2 = np.maximum(z2, 0)
        logits = h2 @ self.W3.T + self.b3
        exp_z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        dp_dz = -probs * probs[:, [cls]]
        dp_dz[:, cls] += probs[:, cls]

        d2 = (dp_dz @ self.W3) * (z2 > 0)
        d1 = (d2 @ self.W2) * (z1 > 0)
        return d1 @ self.W1


def build_space():
    df = pd.read_csv(PROJECT_ROOT / "datasets" / "fetal.csv")
    split = pd.read_csv(PROJECT_ROOT / "datasets" / "fetal_split.csv", index_col="row")["split"]
    d = DataInterface(dataframe=df, target="NSP", continuous_features=CONT_COLS)
    Z = d.norm_encoded_features
    tr_idx = split[split == "train"].index
    Z_train = Z.loc[tr_idx]

    groups = {}
    for i, c in enumerate(Z.columns):
        if c in CONT_COLS:
            groups.setdefault(c, []).append(i)
        else:
            feat_name = c.rsplit("_", 1)[0]
            groups.setdefault(feat_name, []).append(i)

    return Z_train.to_numpy(dtype=np.float64), groups


def perm_shapley(model, q, bg, groups_idx, cls=TARGET_CLASS, n_perms=300, seed=42):
    """Permutation sampling Shapley over feature groups for class cls."""
    n_feats = len(groups_idx)
    n_bg = len(bg)
    phi = np.zeros(n_feats, dtype=float)
    rng = np.random.default_rng(seed)

    for _ in range(n_perms):
        p = rng.permutation(n_feats)
        b_idx = rng.integers(0, n_bg)
        cur = bg[b_idx].copy()
        cur_val = model.prob(cur, cls=cls)[0]
        for f in p:
            nxt = cur.copy()
            nxt[groups_idx[f]] = q[groups_idx[f]]
            nxt_val = model.prob(nxt, cls=cls)[0]
            phi[f] += (nxt_val - cur_val)
            cur = nxt
            cur_val = nxt_val

    return phi / n_perms


def path_ig(model, q, cfs, groups_idx, steps=IG_STEPS, cls=TARGET_CLASS):
    alphas = (np.arange(steps) + 0.5) / steps
    out = []
    for c in cfs:
        pts = q[None, :] + alphas[:, None] * (c - q)[None, :]
        ig = model.grad(pts, cls=cls).mean(0) * (c - q)
        out.append([ig[cols].sum() for cols in groups_idx])
    return np.abs(np.array(out)).mean(0)


def largest_edit(q, cfs, groups_idx):
    return np.mean([[np.abs(c - q)[cols].sum() for cols in groups_idx] for c in cfs], axis=0)


def order_of(scores):
    return list(np.argsort(-np.asarray(scores, dtype=float), kind="stable"))


def t1_curve(model, q, bg, groups_idx, order, ycls, K=K_MAX):
    p0 = model.prob(q, cls=ycls)[0]
    mask = np.zeros(q.size, dtype=bool)
    out = np.zeros(K)
    for k in range(K):
        mask[groups_idx[order[k]]] = True
        replaced = np.where(mask[None, :], bg, q[None, :])
        pk = model.prob(replaced, cls=ycls).mean()
        out[k] = p0 - pk
    return out


def t2_curve(model, q, cfs, groups_idx, order, target=TARGET_CLASS, K=K_MAX):
    drops = np.zeros((len(cfs), K))
    flips = np.zeros((len(cfs), K))
    for ci, c in enumerate(cfs):
        pt0 = model.prob(c, cls=target)[0]
        x = c.copy()
        for k in range(K):
            cols = groups_idx[order[k]]
            x[cols] = q[cols]
            ptk = model.prob(x, cls=target)[0]
            drops[ci, k] = pt0 - ptk
            flips[ci, k] = float(model.predict_class(x)[0] != target)
    return drops.mean(0), flips.mean(0)


def agreement(a, b):
    ta, tb = set(order_of(a)[:3]), set(order_of(b)[:3])
    sp = spearmanr(a, b).correlation
    kt = kendalltau(a, b).correlation
    return dict(
        spearman=float(sp) if not np.isnan(sp) else 0.0,
        kendall=float(kt) if not np.isnan(kt) else 0.0,
        top1=float(np.argmax(a) == np.argmax(b)),
        top3=len(ta & tb) / 3.0,
    )


def load_sets(cf_dir, seed=42):
    out = {}
    pattern = os.path.join(cf_dir, f"s{seed}_r*.npz")
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        row = int(base.split("_r")[1].split(".npz")[0])
        data = np.load(path)
        out[row] = (data["q"].squeeze(), data["cfs"])
    return out


def load_fa(fa_path, seed=42, features=None):
    if not os.path.exists(fa_path):
        return {}
    df = pd.read_csv(fa_path)
    df = df[df["seed"] == seed]
    out = {}
    for row, g in df.groupby("row_id"):
        s = g.set_index("feature")["score"].reindex(features).fillna(0.0)
        out[int(row)] = s.to_numpy(dtype=float)
    return out


def run_analysis(cf_dir, fa_path, out_dir=DEFAULT_OUT, limit=None, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    model = MLP(load_weights(MODEL_PATH))
    Ztr, groups = build_space()
    features = list(groups.keys())
    groups_idx = [groups[f] for f in features]

    rng_bg = np.random.default_rng(42)
    bg = Ztr[rng_bg.choice(len(Ztr), N_BG, replace=False)]

    sets = load_sets(cf_dir, seed)
    fa_scores = load_fa(fa_path, seed, features)

    rows = sorted(r for r in sets if r in fa_scores)
    if limit:
        rows = rows[:limit]

    print(f"Running Phase 2 attribution analysis on {len(rows)} queries...")
    METHODS = ["GradCFA", "Shapley", "Path-IG", "Largest edit"]
    per = {m: dict(t1=[], t2d=[], t2f=[]) for m in METHODS + ["Random"]}
    agree_rows = []
    shap_cache_rows = []
    used = []
    rng = np.random.default_rng(42)

    for i, r in enumerate(rows, 1):
        q, cfs = sets[r]
        ycls = int(model.predict_class(q)[0])
        phi = perm_shapley(model, q, bg, groups_idx, cls=TARGET_CLASS, seed=42 + r)

        for feat, score in zip(features, phi):
            shap_cache_rows.append(dict(row_id=r, feature=feat, phi=float(score)))

        scores = {
            "GradCFA": np.asarray(fa_scores[r], dtype=float),
            "Shapley": np.abs(phi),
            "Path-IG": path_ig(model, q, cfs, groups_idx, cls=TARGET_CLASS),
            "Largest edit": largest_edit(q, cfs, groups_idx),
        }

        for a, b in [("GradCFA", "Shapley"), ("Path-IG", "Shapley"), ("GradCFA", "Path-IG")]:
            agree_rows.append(dict(row=r, pair=f"{a} vs {b}", **agreement(scores[a], scores[b])))

        for m in METHODS:
            o = order_of(scores[m])
            per[m]["t1"].append(t1_curve(model, q, bg, groups_idx, o, ycls))
            d, f = t2_curve(model, q, cfs, groups_idx, o, TARGET_CLASS)
            per[m]["t2d"].append(d)
            per[m]["t2f"].append(f)

        t1r, t2dr, t2fr = [], [], []
        for _ in range(N_RANDOM):
            o = list(rng.permutation(len(features)))
            t1r.append(t1_curve(model, q, bg, groups_idx, o, ycls))
            d, f = t2_curve(model, q, cfs, groups_idx, o, TARGET_CLASS)
            t2dr.append(d)
            t2fr.append(f)
        per["Random"]["t1"].append(np.mean(t1r, 0))
        per["Random"]["t2d"].append(np.mean(t2dr, 0))
        per["Random"]["t2f"].append(np.mean(t2fr, 0))
        used.append(r)

        if i % 10 == 0 or i == len(rows):
            print(f"  Processed [{i}/{len(rows)}] queries")

    for m in per:
        for k in per[m]:
            per[m][k] = np.array(per[m][k])

    # Save cache
    pd.DataFrame(shap_cache_rows).to_csv(os.path.join(out_dir, "phase2_shapley_cache.csv"), index=False)
    agree_df = pd.DataFrame(agree_rows)
    agree_df.to_csv(os.path.join(out_dir, "phase2_agreement_per_query.csv"), index=False)

    # Summary table
    lines = []
    lines.append("=" * 70)
    lines.append(f"Phase 2 Attribution Analysis (Fetal Health, {len(used)} queries)")
    lines.append("=" * 70)
    lines.append("\nAGREEMENT BETWEEN RANKINGS (Mean across queries):")
    agree_mean = agree_df.groupby("pair")[["spearman", "kendall", "top1", "top3"]].mean()
    lines.append(agree_mean.round(4).to_string())

    lines.append("\n\nFAITHFULNESS TEST T1: DROP IN QUERY PREDICTED-CLASS PROBABILITY (Mean across queries):")
    t1_table = pd.DataFrame({m: per[m]["t1"].mean(0) for m in METHODS + ["Random"]}, index=[f"k={k+1}" for k in range(K_MAX)])
    lines.append(t1_table.round(4).to_string())

    lines.append("\n\nFAITHFULNESS TEST T2: DROP IN CF TARGET-CLASS PROBABILITY (Mean across queries):")
    t2d_table = pd.DataFrame({m: per[m]["t2d"].mean(0) for m in METHODS + ["Random"]}, index=[f"k={k+1}" for k in range(K_MAX)])
    lines.append(t2d_table.round(4).to_string())

    lines.append("\n\nFAITHFULNESS TEST T2: CF INVALIDITY FRACTION (Mean across queries):")
    t2f_table = pd.DataFrame({m: per[m]["t2f"].mean(0) for m in METHODS + ["Random"]}, index=[f"k={k+1}" for k in range(K_MAX)])
    lines.append(t2f_table.round(4).to_string())

    summary_text = "\n".join(lines) + "\n"
    print("\n" + summary_text)
    out_file = os.path.join(out_dir, "phase2_summary.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf-dir", default=str(PROJECT_ROOT / "phase1" / "results" / "fetal" / "cfs_A"))
    parser.add_argument("--fa", default=str(PROJECT_ROOT / "phase1" / "results" / "fetal" / "fa_A.csv"))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out
    if args.limit and args.out == DEFAULT_OUT:
        out_dir = os.path.join(DEFAULT_OUT, f"limit_{args.limit}")
    run_analysis(args.cf_dir, args.fa, out_dir, args.limit, args.seed)

