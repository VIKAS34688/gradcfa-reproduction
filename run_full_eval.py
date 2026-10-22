"""
Full test-set evaluation of GradCFA (DiPACE_FA) on the Credit Approval dataset.

Run from the repository root (the folder containing `src`, `datasets` and `pytorch_model.pth`):

    python run_full_eval.py --config A --seeds 42 --limit 3     # quick test (3 queries)
    python run_full_eval.py --config A --seeds 42 43 44         # full run, 3 seeds
    python run_full_eval.py --config A --summarize              # print mean +/- std table
    python run_full_eval.py --config A --which all              # all 138 test rows (target = opposite class)
    python run_full_eval.py --config C --which all --summarize --subset rejected   # summary over the 69 rejected rows only

Design choices (document these in your report):
  * The black-box model is NOT retrained here. It is loaded from `pytorch_model_di.pth`, created by
    check_and_train_di.py in the SAME encoding space that DataInterface (and so GradCFA) feeds the
    model. Only the counterfactual generation uses the other seeds.
  * Results go to `results_di/` (the older `results/` folder came from the notebook's mismatched pipeline).
  * Queries (default, --which neg) = every test-set row (20% split, random_state=42) that the model
    predicts as class 0 (rejected); the target class is 1 (approved). With --which all, every test
    row is used and the target is the opposite of the model's prediction (files get an `_all` suffix).
  * Each query gets its own seed (seed * 1000 + row_id), so a resumed run gives the same results.
  * Plausibility is reported two ways: the repo's raw k-NN distance, and the paper's Eq. 5
    (distances scaled by the min and max among the k nearest points). k = 3 (repo default;
    the paper does not state k).
  * Proximity, sparsity, diversity and confidence use the repo's own Evaluate class.
"""
import argparse
import contextlib
import io
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # no pop-up windows, the GradCFA code calls plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.append(os.path.abspath("src"))
from arc.Binary_Architecture import Model
from data.Data_Interface import DataInterface
from model.Model_Interface import ModelInterface
from cf.GradCFA import DiPACE_FA
from eval.Evaluation import Evaluate

MODEL_PATH = "pytorch_model_di.pth"
N_CF = 5
K = 3
RESULTS_DIR = "results_di"

CAT_COLS = ['Gender', 'Married', 'BankCustomer', 'Industry', 'Ethnicity', 'PriorDefault',
            'CreditScore', 'Employed', 'DriversLicense', 'Citizen']
CONT_COLS = ['Age', 'Debt', 'YearsEmployed', 'Income']

PAPER_HYPERPARAMS = dict(
    proximity_weight=0.5, sparsity_weight=0.5, plausibility_weight=0.5, diversity_weight=0.5,
    penalty_scale=0.1, perturb_scale=0.5,
    proximity_threshold=0.2, sparsity_threshold=0.2,
    plausibility_threshold=0.4, diversity_threshold=0.9,
    high_loss_threshold=0.5, loss_converge_maxiter=2,
)

CONFIGS = {
    "A": dict(PAPER_HYPERPARAMS),                              # the paper's Credit settings
    "B": dict(high_loss_threshold=0.7),                        # repo defaults, as in the authors' notebook
    # approximations of the paper's ablations (use later, Days 8-9)
    "A_nopert": dict(PAPER_HYPERPARAMS, high_loss_threshold=1e9),   # perturbation never triggers
    "A_nopen": dict(PAPER_HYPERPARAMS, penalty_scale=0.0),          # no additional penalty
    # Same as A but loss_converge_maxiter=1, so that the perturb-and-retry step actually runs
    "C": dict(PAPER_HYPERPARAMS, loss_converge_maxiter=1),
    "C_nopert": dict(PAPER_HYPERPARAMS, loss_converge_maxiter=1, high_loss_threshold=1e9),
    "C_nopen": dict(PAPER_HYPERPARAMS, loss_converge_maxiter=1, penalty_scale=0.0),
    # Both perturbation and penalty off (cheap): lets us separate the effect of each one
    "C_nopert_nopen": dict(PAPER_HYPERPARAMS, loss_converge_maxiter=1, high_loss_threshold=1e9, penalty_scale=0.0),
}

PAPER_ROW = {"prox": 0.15, "spars": 0.18, "plaus_eq5_k3": 0.37, "div": 0.96, "conf": 0.78, "avg5": 0.81}


def seed_all(s):
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)


def prepare(which="neg"):
    """Build the DataInterface encoding space (the one GradCFA really feeds the model) and load the model."""
    df = pd.read_csv("datasets/credit.csv").drop(columns="ZipCode")
    for col in ["Industry", "Ethnicity", "Citizen"]:
        df[col] = LabelEncoder().fit_transform(df[col])
    y = df["Approved"]

    d = DataInterface(dataframe=df, target="Approved", continuous_features=CONT_COLS)
    Z = d.norm_encoded_features
    input_size = Z.shape[1]
    Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.2, random_state=42)

    model = Model(input_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        pred_test = (model(torch.tensor(Z_test.to_numpy(dtype=np.float32))) >= 0.5).float().view(-1)
        acc = (pred_test == torch.tensor(y_test.values, dtype=torch.float32)).float().mean().item()
        pred_all = (model(torch.tensor(Z.to_numpy(dtype=np.float32))) >= 0.5).long().view(-1).numpy()

    m = ModelInterface(model_path=MODEL_PATH, input_size=input_size)
    obs = torch.tensor(Z.to_numpy(dtype=np.float32), dtype=torch.float)

    positions = Z.index.get_indexer(Z_test.index)
    queries = []
    for row_id, pos in zip(Z_test.index, positions):
        pred = int(pred_all[pos])
        if which == "neg" and pred != 0:
            continue
        # target class = the opposite of the model's prediction (1 for rejected rows, 0 for approved rows)
        queries.append((int(row_id), df.loc[row_id].drop("Approved").to_list(), 1 - pred))

    print(f"Test rows: {len(Z_test)} | queries used ({which}): {len(queries)} | model test accuracy: {acc:.6f}")
    meta = os.path.join(RESULTS_DIR, "model_di_meta.txt")
    if os.path.exists(meta):
        expected = float(open(meta, encoding="utf-8").read().strip())
        if abs(acc - expected) > 1e-6:
            print(f"WARNING: accuracy differs from the one saved at training time ({expected:.6f}). "
                  "Re-run check_and_train_di.py.")
    return dict(d=d, m=m, model=model, obs=obs, queries=queries)


def plausibility(dist, k=K):
    """dist: (n_cf, n_observed) L1 distances. Returns (raw k-NN distance, paper Eq. 5 value)."""
    vals, _ = torch.topk(dist, k, largest=False)             # k nearest, ascending
    raw = vals.mean().item()
    dmin = vals.min(dim=1, keepdim=True).values
    dmax = vals.max(dim=1, keepdim=True).values
    eq5 = ((vals - dmin) / (dmax - dmin + 1e-8)).mean().item()
    return raw, eq5


def metrics_for(ctx, ev, q, cfs, target=1):
    q = torch.Tensor(q).float()
    c = torch.Tensor(cfs).float()
    dist = torch.cdist(c, ctx["obs"], p=1.0)
    raw, eq5 = plausibility(dist)
    with torch.no_grad():
        valid_frac = ((ctx["model"](c).view(-1) >= 0.5).long() == target).float().mean().item()
    return dict(
        prox=ev.evaluate_proximity(q, c).item(),
        spars=ev.evaluate_sparsity(q, c).item(),
        div=ev.evaluate_diversity(c).item(),
        plaus_raw_k3=raw,
        plaus_eq5_k3=eq5,
        conf=ev.evaluate_confidence(c, desired_class=target).item(),
        valid_frac=valid_frac,
    )


def run(config, seeds, limit, which="neg"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ctx = prepare(which)
    tag = config if which == "neg" else f"{config}_all"
    ev = Evaluate(ctx["d"], ctx["m"])
    kwargs = CONFIGS[config]
    out_csv = os.path.join(RESULTS_DIR, f"full_{tag}.csv")
    fa_csv = os.path.join(RESULTS_DIR, f"fa_{tag}.csv")
    err_txt = os.path.join(RESULTS_DIR, f"errors_{tag}.txt")
    cf_dir = os.path.join(RESULTS_DIR, f"cfs_{tag}")   # raw counterfactuals, so metrics can be recomputed later
    os.makedirs(cf_dir, exist_ok=True)

    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["seed"].astype(int), prev["row_id"].astype(int)))

    queries = ctx["queries"][:limit] if limit else ctx["queries"]

    for seed in seeds:
        todo = [(rid, ql, tgt) for rid, ql, tgt in queries if (seed, rid) not in done]
        print(f"\n[config {tag}] seed {seed}: {len(todo)} queries to run ({len(queries) - len(todo)} already done)")
        t_start = time.time()
        for i, (rid, ql, tgt) in enumerate(todo, 1):
            t0 = time.time()
            try:
                seed_all(seed * 1000 + rid)
                exp = DiPACE_FA(ctx["d"], ctx["m"])
                with contextlib.redirect_stdout(io.StringIO()):
                    q, cfs = exp.generate_counterfactuals(ql, total_CFs=N_CF, **kwargs)
                plt.close("all")
                row = metrics_for(ctx, ev, q, cfs, tgt)
                np.savez_compressed(os.path.join(cf_dir, f"s{seed}_r{rid}.npz"), q=np.asarray(q), cfs=np.asarray(cfs))
                row.update(config=config, seed=seed, row_id=rid, seconds=round(time.time() - t0, 2))
                pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

                fa_rows = [dict(config=config, seed=seed, row_id=rid, feature=name, score=float(score))
                           for name, score in exp.importance_ranking]
                pd.DataFrame(fa_rows).to_csv(fa_csv, mode="a", header=not os.path.exists(fa_csv), index=False)
            except Exception as e:  # keep a long run alive, but show the problem
                plt.close("all")
                msg = f"seed {seed} row {rid}: {type(e).__name__}: {e}"
                print("ERROR", msg)
                with open(err_txt, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                continue

            elapsed = time.time() - t_start
            eta = elapsed / i * (len(todo) - i)
            print(f"  [{i}/{len(todo)}] row {rid} -> class {tgt}  {time.time() - t0:5.1f}s   elapsed {elapsed / 60:5.1f} min   ETA {eta / 60:5.1f} min")
    print("\nFinished. Results in", out_csv)


def summarize(config, which="neg", subset="all"):
    tag = config if which == "neg" else f"{config}_all"
    path = os.path.join(RESULTS_DIR, f"full_{tag}.csv")
    df = pd.read_csv(path)
    if subset != "all":
        # rows the model predicts as rejected = the query rows of the default (neg) run of config A
        rejected_ids = set(pd.read_csv(os.path.join(RESULTS_DIR, "full_A.csv"))["row_id"].astype(int))
        df = df[df["row_id"].isin(rejected_ids)] if subset == "rejected" else df[~df["row_id"].isin(rejected_ids)]
    df["avg5"] = ((1 - df["prox"]) + (1 - df["spars"]) + (1 - df["plaus_eq5_k3"]) + df["div"] + df["conf"]) / 5
    cols = ["prox", "spars", "plaus_eq5_k3", "plaus_raw_k3", "div", "conf", "valid_frac", "avg5", "seconds"]
    per_seed = df.groupby("seed")[cols].mean()
    n_per_seed = df.groupby("seed").size()
    table = pd.DataFrame({"mean": per_seed.mean(), "std_across_seeds": per_seed.std(ddof=1)})
    table["paper (GradCFA, Credit)"] = pd.Series(PAPER_ROW)
    text = (f"Config {tag} (subset: {subset})\nSeeds: {list(per_seed.index)}\nQueries per seed: {n_per_seed.to_dict()}\n\n"
            f"{table.round(4).to_string()}\n")
    print(text)
    out = os.path.join(RESULTS_DIR, f"summary_{tag}.txt" if subset == "all" else f"summary_{tag}_{subset}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="A")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries (for a quick test)")
    ap.add_argument("--which", choices=["neg", "all"], default="neg",
                    help="neg = test rows the model predicts as rejected (default); all = every test row (138)")
    ap.add_argument("--subset", choices=["all", "rejected", "approved"], default="all",
                    help="with --summarize: restrict the summary to rows the model rejects / approves")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.config, args.which, args.subset)
    else:
        run(args.config, args.seeds, args.limit, args.which)