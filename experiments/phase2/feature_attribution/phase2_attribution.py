"""
Phase 2: Is GradCFA's feature attribution trustworthy?

Everything here is computed offline from results you already have (no counterfactual generation):
  results_di/cfs_C_all/s42_r<row>.npz   saved query + 5 counterfactuals per test row (DataInterface space)
  results_di/fa_C_all.csv               GradCFA's feature scores (14 original features) per query
  results_di/full_C_all.csv             used only to check that the numpy model matches the torch model
  pytorch_model_di.pth                  the black-box model

What it computes (per test query, then averaged over all queries with 95% bootstrap intervals):
  1. Exact Shapley values (the quantity SHAP's KernelExplainer estimates) over the 14 original features,
     interventional, with a 50-row background from the training data.
  2. Path-IG: a displacement-aware attribution for the counterfactuals (our proposed alternative):
     integrated gradients from the query to each counterfactual, summed per feature group, |.| averaged over counterfactuals.
  3. Agreement between rankings: Spearman, Kendall, top-1 match, top-3 overlap.
  4. Faithfulness test T1 (query level): replace the top-k ranked features by background values and
     measure the drop in the model's predicted-class probability. Bigger drop = better ranking.
  5. Faithfulness test T2 (counterfactual level, the paper's own claim): reset the top-k ranked features of a
     valid counterfactual back to the query's values and measure the drop in target-class probability
     (and how often the counterfactual stops being valid). Bigger drop = better ranking.
  Rankings compared: GradCFA, Shapley, Path-IG, "Largest edit" (reference: features the counterfactuals changed most), Random.

Run from the repository root:
    python phase2_attribution.py --limit 5        # quick test on 5 queries
    python phase2_attribution.py                  # all queries (a few minutes)
    python phase2_attribution.py --check-shap     # also verify the Shapley values against the shap library (3 queries)
    python phase2_attribution.py --cf-dir results_di/cfs_A_all --fa results_di/fa_A_all.csv --full-csv results_di/full_A_all.csv
"""
import argparse
import glob
import os
import sys
import warnings
from math import factorial

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("src"))

MODEL_PATH = "pytorch_model_di.pth"
CONT = ["Age", "Debt", "YearsEmployed", "Income"]
N_BG = 50          # background rows for Shapley and for the ablation test
N_RANDOM = 30      # random rankings averaged per query
IG_STEPS = 64
K_MAX = 13         # features removed / reset: 1..13 (14 would be identical for every ranking)
BOOT = 2000


# ---------------------------------------------------------------- model (numpy copy of the torch MLP)
def load_weights(path):
    import torch
    sd = torch.load(path, map_location="cpu")
    keys = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
    return [sd[k].detach().numpy().astype(np.float64) for k in keys]


class MLP:
    def __init__(self, W):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = W

    def prob(self, X):
        X = np.atleast_2d(X)
        h1 = np.maximum(X @ self.W1.T + self.b1, 0)
        h2 = np.maximum(h1 @ self.W2.T + self.b2, 0)
        z = (h2 @ self.W3.T + self.b3).ravel()
        return 1.0 / (1.0 + np.exp(-z))

    def grad(self, X):
        """d probability / d input, shape (n, D)."""
        X = np.atleast_2d(X)
        z1 = X @ self.W1.T + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W2.T + self.b2
        h2 = np.maximum(z2, 0)
        z = (h2 @ self.W3.T + self.b3).ravel()
        p = 1.0 / (1.0 + np.exp(-z))
        d2 = self.W3 * (z2 > 0)              # (n, 32)
        d1 = (d2 @ self.W2) * (z1 > 0)       # (n, 64)
        return (d1 @ self.W1) * (p * (1 - p))[:, None]


def py(p, cls):
    """Probability of class cls given the probability of class 1."""
    return p if cls == 1 else 1.0 - p


# ---------------------------------------------------------------- data space
def build_space():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from data.Data_Interface import DataInterface
    df = pd.read_csv("datasets/credit.csv").drop(columns="ZipCode")
    for col in ["Industry", "Ethnicity", "Citizen"]:
        df[col] = LabelEncoder().fit_transform(df[col])
    d = DataInterface(dataframe=df, target="Approved", continuous_features=CONT)
    Z = d.norm_encoded_features
    Z_train, _, _, _ = train_test_split(Z, df["Approved"], test_size=0.2, random_state=42)
    groups = {}
    for i, c in enumerate(Z.columns):
        groups.setdefault(c if c in CONT else c.rsplit("_", 1)[0], []).append(i)
    return Z_train.to_numpy(dtype=np.float64), groups


# ---------------------------------------------------------------- attributions
def exact_shapley(model, q, bg, groups_idx):
    """Exact interventional Shapley values over feature groups (2^n coalitions)."""
    n, D = len(groups_idx), q.size
    masks = np.arange(2 ** n)
    bits = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)      # True = group keeps the query's values
    colmask = np.zeros((2 ** n, D), dtype=bool)
    for gi, cols in enumerate(groups_idx):
        colmask[:, cols] = bits[:, [gi]]
    v = np.zeros(2 ** n)
    for b in bg:
        v += model.prob(np.where(colmask, q[None, :], b[None, :]))
    v /= len(bg)
    pop = bits.sum(1)
    w = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    phi = np.zeros(n)
    for i in range(n):
        idx0 = masks[~bits[:, i]]
        phi[i] = np.sum(w[pop[idx0]] * (v[idx0 | (1 << i)] - v[idx0]))
    return phi


def path_ig(model, q, cfs, groups_idx, steps=IG_STEPS):
    """Integrated gradients from q to each counterfactual, summed per group, |.| averaged over counterfactuals."""
    alphas = (np.arange(steps) + 0.5) / steps
    out = []
    for c in cfs:
        pts = q[None, :] + alphas[:, None] * (c - q)[None, :]
        ig = model.grad(pts).mean(0) * (c - q)
        out.append([ig[cols].sum() for cols in groups_idx])
    return np.abs(np.array(out)).mean(0)


def largest_edit(q, cfs, groups_idx):
    return np.mean([[np.abs(c - q)[cols].sum() for cols in groups_idx] for c in cfs], axis=0)


def order_of(scores):
    return list(np.argsort(-np.asarray(scores, dtype=float), kind="stable"))


# ---------------------------------------------------------------- faithfulness tests
def t1_curve(model, q, bg, groups_idx, order, ycls, K=K_MAX):
    D = q.size
    p0 = py(model.prob(q)[0], ycls)
    mask = np.zeros(D, dtype=bool)
    out = np.zeros(K)
    for k in range(K):
        mask[groups_idx[order[k]]] = True
        out[k] = p0 - py(model.prob(np.where(mask[None, :], bg, q[None, :])).mean(), ycls)
    return out


def t2_curve(model, q, cfs_valid, groups_idx, order, target, K=K_MAX):
    drops = np.zeros((len(cfs_valid), K))
    flips = np.zeros((len(cfs_valid), K))
    for ci, c in enumerate(cfs_valid):
        pt0 = py(model.prob(c)[0], target)
        x = c.copy()
        for k in range(K):
            cols = groups_idx[order[k]]
            x[cols] = q[cols]
            ptk = py(model.prob(x)[0], target)
            drops[ci, k] = pt0 - ptk
            flips[ci, k] = float(ptk < 0.5)
    return drops.mean(0), flips.mean(0)


def agreement(a, b):
    ta, tb = set(order_of(a)[:3]), set(order_of(b)[:3])
    return dict(
        spearman=spearmanr(a, b).correlation,
        kendall=kendalltau(a, b).correlation,
        top1=float(np.argmax(a) == np.argmax(b)),
        top3=len(ta & tb) / 3.0,
    )


# ---------------------------------------------------------------- statistics
def boot_idx(n, seed=0):
    return np.random.default_rng(seed).integers(0, n, (BOOT, n))


def mean_ci(arr, idx):
    """arr: (Q,) or (Q, K). Returns mean, lo, hi (95% bootstrap over queries)."""
    arr = np.asarray(arr, dtype=float)
    boots = arr[idx].mean(axis=1)
    return arr.mean(0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


# ---------------------------------------------------------------- main analysis
def run_analysis(model, groups, bg, sets, fa_scores, features, shap_cache=None, rng_seed=42):
    """sets: {row: (q, cfs)}; fa_scores: {row: scores aligned to `features`}."""
    groups_idx = [groups[f] for f in features]
    rows = sorted(r for r in sets if r in fa_scores)
    rng = np.random.default_rng(rng_seed)
    METHODS = ["GradCFA", "Shapley", "Path-IG", "Largest edit"]

    per = {m: dict(t1=[], t2d=[], t2f=[]) for m in METHODS + ["Random"]}
    agree_rows = []
    used = []
    shap_out = {}
    for r in rows:
        q, cfs = sets[r]
        pq = model.prob(q)[0]
        ycls = int(pq >= 0.5)
        target = 1 - ycls
        valid = cfs[(model.prob(cfs) >= 0.5).astype(int) == target]
        if len(valid) == 0:
            continue
        if shap_cache is not None and r in shap_cache:
            phi = shap_cache[r]
        else:
            phi = exact_shapley(model, q, bg, groups_idx)
        shap_out[r] = phi
        scores = {
            "GradCFA": np.asarray(fa_scores[r], dtype=float),
            "Shapley": np.abs(phi),
            "Path-IG": path_ig(model, q, cfs, groups_idx),
            "Largest edit": largest_edit(q, cfs, groups_idx),
        }
        for a, b in [("GradCFA", "Shapley"), ("Path-IG", "Shapley"), ("GradCFA", "Path-IG")]:
            agree_rows.append(dict(row=r, pair=f"{a} vs {b}", **agreement(scores[a], scores[b])))
        for m in METHODS:
            o = order_of(scores[m])
            per[m]["t1"].append(t1_curve(model, q, bg, groups_idx, o, ycls))
            d, f = t2_curve(model, q, valid, groups_idx, o, target)
            per[m]["t2d"].append(d)
            per[m]["t2f"].append(f)
        t1r, t2dr, t2fr = [], [], []
        for _ in range(N_RANDOM):
            o = list(rng.permutation(len(features)))
            t1r.append(t1_curve(model, q, bg, groups_idx, o, ycls))
            d, f = t2_curve(model, q, valid, groups_idx, o, target)
            t2dr.append(d)
            t2fr.append(f)
        per["Random"]["t1"].append(np.mean(t1r, 0))
        per["Random"]["t2d"].append(np.mean(t2dr, 0))
        per["Random"]["t2f"].append(np.mean(t2fr, 0))
        used.append(r)

    for m in per:
        for key in per[m]:
            per[m][key] = np.array(per[m][key])
    return dict(per=per, agree=pd.DataFrame(agree_rows), rows=used, shap=shap_out, methods=METHODS + ["Random"])


def summarize(res, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    per, agree, methods = res["per"], res["agree"], res["methods"]
    Q = len(res["rows"])
    idx = boot_idx(Q)
    lines = [f"Phase 2 summary ({tag}): {Q} queries with at least one valid counterfactual",
             "95% intervals are bootstrap over queries.", ""]

    # agreement
    lines.append("Agreement between rankings (random: Spearman 0, top-1 0.071, top-3 overlap 0.214)")
    arows = []
    for pair, g in agree.groupby("pair"):
        for metric in ["spearman", "kendall", "top1", "top3"]:
            vals = g[metric].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            m, lo, hi = mean_ci(vals, boot_idx(len(vals), 1))
            arows.append(dict(pair=pair, metric=metric, mean=m, lo=lo, hi=hi))
    A = pd.DataFrame(arows)
    A.to_csv(os.path.join(out_dir, "agreement_summary.csv"), index=False)
    lines.append(A.pivot(index="pair", columns="metric", values="mean").round(3).to_string())
    lines.append("")

    # faithfulness curves and AUC
    tests = [("T1_ablation_drop", "t1"), ("T2_reversal_drop", "t2d"), ("T2_reversal_flip_rate", "t2f")]
    curve_rows, auc = [], {}
    for name, key in tests:
        for m in methods:
            mean, lo, hi = mean_ci(per[m][key], idx)
            for k in range(len(mean)):
                curve_rows.append(dict(test=name, method=m, k=k + 1, mean=mean[k], lo=lo[k], hi=hi[k]))
            a_mean, a_lo, a_hi = mean_ci(per[m][key].mean(1), idx)
            auc[(name, m)] = (float(a_mean), float(a_lo), float(a_hi))
    pd.DataFrame(curve_rows).to_csv(os.path.join(out_dir, "faithfulness_curves.csv"), index=False)

    for name, key in tests:
        lines.append(f"{name}: mean over k=1..{K_MAX} (higher = more faithful ranking)")
        for m in methods:
            a, lo, hi = auc[(name, m)]
            k1 = mean_ci(per[m][key][:, 0], idx)[0]
            k3 = mean_ci(per[m][key][:, 2], idx)[0]
            lines.append(f"  {m:13s} avg {a:.4f} [{lo:.4f}, {hi:.4f}]   k=1 {float(k1):.4f}   k=3 {float(k3):.4f}")
        # paired differences against GradCFA
        for m in ["Shapley", "Path-IG", "Largest edit"]:
            diff = per[m][key].mean(1) - per["GradCFA"][key].mean(1)
            a, lo, hi = mean_ci(diff, idx)
            lines.append(f"  paired {m} - GradCFA: {float(a):+.4f} [{float(lo):+.4f}, {float(hi):+.4f}]")
        diff = per["Path-IG"][key].mean(1) - per["Shapley"][key].mean(1)
        a, lo, hi = mean_ci(diff, idx)
        lines.append(f"  paired Path-IG - Shapley: {float(a):+.4f} [{float(lo):+.4f}, {float(hi):+.4f}]")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "phase2_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cdf = pd.DataFrame(curve_rows)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, _) in zip(axes, tests):
            for m in methods:
                g = cdf[(cdf.test == name) & (cdf.method == m)]
                ax.plot(g.k, g["mean"], marker="o", ms=3, label=m)
                ax.fill_between(g.k, g.lo, g.hi, alpha=0.15)
            ax.set_title(name)
            ax.set_xlabel("number of top-ranked features changed (k)")
        axes[0].set_ylabel("effect on the model")
        axes[0].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_faithfulness.png"), dpi=150)
        plt.close()
        fig, ax = plt.subplots(figsize=(6, 4))
        piv = A[A.metric.isin(["top1", "top3"])].pivot(index="pair", columns="metric", values="mean")
        piv.plot(kind="bar", ax=ax)
        ax.axhline(1 / 14, ls="--", c="gray")
        ax.set_ylabel("agreement (dashed: random top-1)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_agreement.png"), dpi=150)
        plt.close()
    except Exception as e:  # figures are optional
        print("Could not draw figures:", e)


# ---------------------------------------------------------------- IO helpers
def load_sets(cf_dir, seed, limit=None):
    sets = {}
    for path in sorted(glob.glob(os.path.join(cf_dir, f"s{seed}_r*.npz"))):
        row = int(os.path.basename(path).split("_r")[1].split(".")[0])
        z = np.load(path)
        q = z["q"].reshape(-1).astype(np.float64)
        sets[row] = (q, z["cfs"].reshape(-1, q.size).astype(np.float64))
    if limit:
        sets = {r: sets[r] for r in list(sets)[:limit]}
    return sets


def load_fa(path, seed, features):
    fa = pd.read_csv(path)
    fa = fa[fa.seed == seed]
    out = {}
    for row, g in fa.groupby("row_id"):
        s = g.set_index("feature")["score"].reindex(features)
        if not s.isna().any():
            out[int(row)] = s.to_numpy(dtype=float)
    return out


def check_model(model, sets, full_csv, seed):
    """The numpy model must reproduce the valid_frac that the torch run stored in the results CSV."""
    full = pd.read_csv(full_csv)
    full = full[full.seed == seed].set_index("row_id")
    bad, n = 0, 0
    for r, (q, cfs) in sets.items():
        if r not in full.index:
            continue
        target = 1 - int(model.prob(q)[0] >= 0.5)
        vf = float(((model.prob(cfs) >= 0.5).astype(int) == target).mean())
        n += 1
        bad += abs(vf - float(full.loc[r, "valid_frac"])) > 1e-6   # the stored value is 32-bit (0.8 is saved as 0.80000001)
    print(f"Model check: {n - bad}/{n} queries reproduce the stored valid_frac exactly")
    return bad == 0


def check_shap_library(model, q, bg, groups_idx):
    import shap
    n, D = len(groups_idx), q.size

    def f(M):
        M = np.atleast_2d(M) > 0.5
        colmask = np.zeros((M.shape[0], D), dtype=bool)
        for gi, cols in enumerate(groups_idx):
            colmask[:, cols] = M[:, [gi]]
        return np.mean([model.prob(np.where(colmask, q[None, :], b[None, :])) for b in bg], axis=0)

    ex = shap.KernelExplainer(f, np.zeros((1, n)))
    sv = np.asarray(ex.shap_values(np.ones((1, n)), nsamples=2 ** n - 2, l1_reg=False, silent=True)).reshape(-1)
    mine = exact_shapley(model, q, bg, groups_idx)
    return float(np.max(np.abs(sv - mine)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-dir", default="results_di/cfs_C_all")
    ap.add_argument("--fa", default="results_di/fa_C_all.csv")
    ap.add_argument("--full-csv", default="results_di/full_C_all.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check-shap", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = os.path.basename(args.cf_dir.rstrip("/\\")).replace("cfs_", "")
    out_dir = args.out or os.path.join("results_di", f"phase2_{tag}")
    if args.limit and not args.out:
        out_dir += f"_limit{args.limit}"   # quick tests can never overwrite the full results

    model = MLP(load_weights(MODEL_PATH))
    Ztr, groups = build_space()
    features = list(groups.keys())
    bg = Ztr[np.random.default_rng(42).choice(len(Ztr), N_BG, replace=False)]
    sets = load_sets(args.cf_dir, args.seed, args.limit)
    fa_scores = load_fa(args.fa, args.seed, features)
    print(f"{len(sets)} counterfactual sets, {len(fa_scores)} GradCFA score vectors, {len(features)} features")
    if not check_model(model, sets, args.full_csv, args.seed):
        print("WARNING: the numpy model does not match the stored results; stop and tell your assistant.")
        return

    if args.check_shap:
        for r in list(sets)[:3]:
            diff = check_shap_library(model, sets[r][0], bg, [groups[f] for f in features])
            print(f"Row {r}: max |exact - shap library| = {diff:.2e}")

    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join("results_di", "phase2_shapley_cache.csv")
    cache = {}
    if os.path.exists(cache_path):
        c = pd.read_csv(cache_path)
        for r, g in c.groupby("row_id"):
            s = g.set_index("feature")["phi"].reindex(features)
            if not s.isna().any():
                cache[int(r)] = s.to_numpy(dtype=float)
    res = run_analysis(model, groups, bg, sets, fa_scores, features, shap_cache=cache)
    allphi = {**cache, **res["shap"]}
    pd.DataFrame([dict(row_id=r, feature=f, phi=v) for r, ph in allphi.items()
                  for f, v in zip(features, ph)]).to_csv(cache_path, index=False)
    summarize(res, out_dir, tag)
    print("\nSaved to", out_dir)


if __name__ == "__main__":
    main()