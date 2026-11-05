"""
Full test-set evaluation of GradCFA on the UCI Fetal Health dataset.

Paper settings for Fetal Health:
  * Rows, features: 2126 rows, 32 features (22 continuous, 10 categorical), 3 classes
  * Split: 60/20/20 train/validation/test (seed=42)
  * Target class: 0 (Normal)
  * Queries: test-set rows where predicted class != 0 (classes 1 and 2)
  * lambda (each term): 0.5
  * gamma_pen (penalty_scale): 0.1
  * gamma_pert (perturb_scale): 0.5
  * tau_prox = tau_spars: 0.4
  * tau_plaus: 0.5
  * tau_loss (high_loss_threshold): 0.6
  * tau_div: 0.9 (not stated in paper, 0.9 used as standard)
  * Paper GradCFA: Prox 0.34 / Spars 0.35 / Plaus 0.42 / Div 0.96 / Conf 0.79 / Avg 0.73
  * Without perturbation avg: 0.64
  * Without penalty avg: 0.65

Usage:
  python run_full_eval_fetal.py --config A --seeds 42 --limit 5
  python run_full_eval_fetal.py --config A --seeds 42 43 44
  python run_full_eval_fetal.py --config C --seeds 42
  python run_full_eval_fetal.py --config A --summarize
  python run_full_eval_fetal.py --config C --summarize
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

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while PROJECT_ROOT.parent != PROJECT_ROOT and not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "tmp_mpl")
os.makedirs(PROJECT_ROOT / "tmp_mpl", exist_ok=True)
import matplotlib
matplotlib.use("Agg")  # no pop-up windows
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

import numpy as np
import pandas as pd
import torch

from arc.Multiclass_Architecture import Model
from data.Data_Interface import DataInterface
from model.Model_Interface_MC import ModelInterface
from cf.GradCFA_MC import GradCFA
from eval.Evaluation import Evaluate

MODEL_PATH = str(PROJECT_ROOT / "pytorch_model_fetal.pth")
N_CF = 5
K = 3
RESULTS_DIR = str(PROJECT_ROOT / "phase1" / "results" / "fetal")
TARGET_CLASS = 0

CONT_COLS = ["LBE","LB","AC","FM","UC","ASTV","MSTV","ALTV","MLTV","DL","DS","DP",
             "Width","Min","Max","Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"]
CAT_COLS = ["A","B","C","D","E","AD","DE","LD","FS","SUSP"]

PAPER_HYPERPARAMS = dict(
    proximity_weight=1.5,
    sparsity_weight=1.0,
    plausibility_weight=0.5,
    diversity_weight=0.5,
    penalty_scale=0.1,
    perturb_scale=0.5,
    proximity_threshold=0.35,
    sparsity_threshold=0.35,
    plausibility_threshold=0.5,
    diversity_threshold=0.9,
    problem_type="multiclass",
    desired_value=TARGET_CLASS,
    min_iter=500,
    max_iter=2000,
    init_near_query_instance=False,
)

CONFIGS = {
    # Config A: perturbation off (fast baseline / paper ablation)
    "A": dict(PAPER_HYPERPARAMS, high_loss_threshold=1e9, loss_converge_maxiter=2),
    # Config C: full method with perturbation (loss_converge_maxiter=1 and high_loss_threshold=2.5)
    "C": dict(PAPER_HYPERPARAMS, high_loss_threshold=2.5, loss_converge_maxiter=1),
    # Ablations
    "A_nopert": dict(PAPER_HYPERPARAMS, high_loss_threshold=1e9, loss_converge_maxiter=2),
    "A_nopen": dict(PAPER_HYPERPARAMS, penalty_scale=0.0, high_loss_threshold=1e9, loss_converge_maxiter=2),
    "C_nopert": dict(PAPER_HYPERPARAMS, high_loss_threshold=1e9, loss_converge_maxiter=1),
    "C_nopen": dict(PAPER_HYPERPARAMS, penalty_scale=0.0, high_loss_threshold=2.5, loss_converge_maxiter=1),
}

PAPER_ROW = {
    "prox": 0.34,
    "spars": 0.35,
    "plaus_eq5_k3": 0.42,
    "div": 0.96,
    "conf": 0.79,
    "avg5": 0.73,
}


def seed_all(s):
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)


def prepare(which="neg"):
    df = pd.read_csv(PROJECT_ROOT / "datasets" / "fetal.csv")
    split = pd.read_csv(PROJECT_ROOT / "datasets" / "fetal_split.csv", index_col="row")["split"]

    d = DataInterface(dataframe=df, target="NSP", continuous_features=CONT_COLS)
    Z = d.norm_encoded_features
    input_size = Z.shape[1]
    output_size = 3

    te_idx = split[split == "test"].index
    Z_test = Z.loc[te_idx]
    y_test = df.loc[te_idx, "NSP"]

    model = Model(input_size, output_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits_test = model(torch.tensor(Z_test.to_numpy(dtype=np.float32)))
        pred_test = logits_test.argmax(dim=1).numpy()
        acc = (pred_test == y_test.values).mean()

        logits_all = model(torch.tensor(Z.to_numpy(dtype=np.float32)))
        pred_all = logits_all.argmax(dim=1).numpy()

    m = ModelInterface(model_path=MODEL_PATH, input_size=input_size, output_size=output_size)
    obs = torch.tensor(Z.to_numpy(dtype=np.float32), dtype=torch.float)

    queries = []
    for row_id in te_idx:
        pred = int(pred_all[row_id])
        true_cls = int(df.loc[row_id, "NSP"])
        if which == "neg" and pred == TARGET_CLASS:
            continue
        # queries target class 0
        queries.append((int(row_id), df.loc[row_id].drop("NSP").to_list(), TARGET_CLASS, true_cls, pred))

    print(f"Test rows: {len(Z_test)} | queries used ({which}): {len(queries)} | model test accuracy: {acc:.4f} (paper: 0.91)")
    return dict(df=df, d=d, m=m, model=model, obs=obs, queries=queries, acc=acc)


def plausibility(dist, k=K):
    vals, _ = torch.topk(dist, k, largest=False)
    raw = vals.mean().item()
    dmin = vals.min(dim=1, keepdim=True).values
    dmax = vals.max(dim=1, keepdim=True).values
    eq5 = ((vals - dmin) / (dmax - dmin + 1e-8)).mean().item()
    return raw, eq5


def metrics_for(ctx, ev, q, cfs, target=TARGET_CLASS):
    q_t = torch.tensor(q, dtype=torch.float32)
    c_t = torch.tensor(cfs, dtype=torch.float32)
    dist = torch.cdist(c_t, ctx["obs"], p=1.0)
    raw, eq5 = plausibility(dist, k=K)

    with torch.no_grad():
        logits = ctx["model"](c_t)
        probs = torch.softmax(logits, dim=1)
        conf = probs[:, target].mean().item()
        valid_frac = (logits.argmax(dim=1) == target).float().mean().item()

    prox = ev.evaluate_proximity(q_t, c_t).item()
    spars = ev.evaluate_sparsity(q_t, c_t).item()
    div = ev.evaluate_diversity(c_t).item()

    return dict(
        prox=prox,
        spars=spars,
        div=div,
        plaus_raw_k3=raw,
        plaus_eq5_k3=eq5,
        conf=conf,
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
    cf_dir = os.path.join(RESULTS_DIR, f"cfs_{tag}")
    os.makedirs(cf_dir, exist_ok=True)

    import json
    param_json = os.path.join(RESULTS_DIR, f"hyperparams_{tag}.json")
    with open(param_json, "w", encoding="utf-8") as pf:
        json.dump({k: (str(v) if isinstance(v, (np.integer, np.floating)) else v) for k, v in kwargs.items()}, pf, indent=2)

    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["seed"].astype(int), prev["row_id"].astype(int)))

    queries = ctx["queries"][:limit] if limit else ctx["queries"]

    for seed in seeds:
        todo = [(rid, ql, tgt, tc, pc) for rid, ql, tgt, tc, pc in queries if (seed, rid) not in done]
        print(f"\n[config {tag}] seed {seed}: {len(todo)} queries to run ({len(queries) - len(todo)} already done)")
        t_start = time.time()
        for i, (rid, ql, tgt, tc, pc) in enumerate(todo, 1):
            t0 = time.time()
            try:
                seed_all(seed * 1000 + rid)
                exp = GradCFA(ctx["d"], ctx["m"])
                exp.plot_loss_curve = lambda *args, **kwargs: None
                exp.plot_feature_importance = lambda *args, **kwargs: None

                with contextlib.redirect_stdout(io.StringIO()):
                    q, cfs = exp.generate_counterfactuals(ql, total_CFs=N_CF, **kwargs)

                row = metrics_for(ctx, ev, q, cfs, tgt)
                np.savez_compressed(os.path.join(cf_dir, f"s{seed}_r{rid}.npz"), q=np.asarray(q), cfs=np.asarray(cfs))
                row.update(
                    config=config,
                    seed=seed,
                    row_id=rid,
                    true_class=tc,
                    pred_class=pc,
                    target_class=tgt,
                    seconds=round(time.time() - t0, 2),
                )
                pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

                if hasattr(exp, "importance_ranking") and exp.importance_ranking:
                    fa_rows = [
                        dict(config=config, seed=seed, row_id=rid, true_class=tc, pred_class=pc, feature=name, score=float(score))
                        for name, score in exp.importance_ranking
                    ]
                    pd.DataFrame(fa_rows).to_csv(fa_csv, mode="a", header=not os.path.exists(fa_csv), index=False)

            except Exception as e:
                msg = f"seed {seed} row {rid}: {type(e).__name__}: {e}"
                print("ERROR", msg)
                with open(err_txt, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                continue

            elapsed = time.time() - t_start
            eta = elapsed / i * (len(todo) - i)
            print(f"  [{i}/{len(todo)}] row {rid} (class {pc}->{tgt}) {time.time() - t0:5.1f}s | elapsed {elapsed / 60:5.1f}m | ETA {eta / 60:5.1f}m")

    print("\nFinished. Results in", out_csv)


def summarize(config, which="neg"):
    tag = config if which == "neg" else f"{config}_all"
    path = os.path.join(RESULTS_DIR, f"full_{tag}.csv")
    if not os.path.exists(path):
        legacy_path = os.path.join(str(PROJECT_ROOT / "results_fetal"), f"full_{tag}.csv")
        if os.path.exists(legacy_path):
            path = legacy_path
        else:
            print(f"No results found at {path}")
            return

    df = pd.read_csv(path)
    df["avg5"] = ((1 - df["prox"]) + (1 - df["spars"]) + (1 - df["plaus_eq5_k3"]) + df["div"] + df["conf"]) / 5

    cols = ["prox", "spars", "plaus_eq5_k3", "plaus_raw_k3", "div", "conf", "valid_frac", "avg5", "seconds"]
    per_seed = df.groupby("seed")[cols].mean()
    n_per_seed = df.groupby("seed").size()

    table = pd.DataFrame({"mean": per_seed.mean(), "std_across_seeds": per_seed.std(ddof=1)})
    table["paper (GradCFA, Fetal)"] = pd.Series(PAPER_ROW)

    # Per-class breakdown
    lines = []
    lines.append("=" * 70)
    lines.append(f"GradCFA Reproduction Summary: Fetal Health (Config {tag})")
    lines.append(f"Seeds: {list(per_seed.index)} | Total rows in CSV: {len(df)}")
    lines.append(f"Queries per seed: {n_per_seed.to_dict()}")
    lines.append("=" * 70)
    lines.append("\nOVERALL METRICS TABLE:")
    lines.append(table.round(4).to_string())

    if "pred_class" in df.columns or "true_class" in df.columns:
        cls_col = "pred_class" if "pred_class" in df.columns else "true_class"
        lines.append(f"\n\nPER ORIGINAL CLASS ({cls_col}) BREAKDOWN:")
        per_cls = df.groupby(cls_col)[cols].mean()
        lines.append(per_cls.round(4).to_string())

    fa_path = os.path.join(RESULTS_DIR, f"fa_{tag}.csv")
    if os.path.exists(fa_path):
        fdf = pd.read_csv(fa_path)
        feat_rank = fdf.groupby("feature")["score"].mean().sort_values(ascending=False)
        lines.append("\n\nTOP 10 ATTRIBUTED FEATURES (mean score):")
        lines.append(feat_rank.head(10).round(4).to_string())
        lines.append(f"\nPaper top features: D, then A")

    summary_text = "\n".join(lines) + "\n"
    print(summary_text)

    out = os.path.join(RESULTS_DIR, f"summary_{tag}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="A")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries")
    ap.add_argument("--which", choices=["neg", "all"], default="neg")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.config, args.which)
    else:
        run(args.config, args.seeds, args.limit, args.which)

