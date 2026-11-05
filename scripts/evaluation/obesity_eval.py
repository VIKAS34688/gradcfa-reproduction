"""
Full test-set evaluation of GradCFA (multi-class) on Obesity. Run from the repository root.

    python obesity_eval.py --config C_nopert --seeds 42 --limit 3          # quick test
    python obesity_eval.py --config C_nopert --seeds 42 --limit 5          # timing test
    python obesity_eval.py --config C_nopert --seeds 42 --shard 0/4        # one of 4 parallel processes (run 0/4 .. 3/4)
    python obesity_eval.py --config C_nopert --summarize                   # overall, per original class, feature ranking

Queries = every test row (20% of 60/20/20 split) that the model does NOT predict as class 1; target class = 1 (normal weight).
"Original class" in the per-class table = the class the MODEL predicts for the query.
Each query has its own seed (seed*1000+row_id), so a resumed or sharded run gives identical results.
Every shard writes its own CSV files (never two processes on one file). Finished queries are skipped on restart.
Plausibility is saved two ways: raw k-NN L1 distance and the paper's Eq. 5 (k=3, an assumption: the paper does not state k).
Confidence = mean softmax probability of the target class; valid_frac = share of counterfactuals the model puts in class 1.
"""
import argparse
import contextlib
import gc
import glob
import io
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from obesity_common import *
from arc.Multiclass_Architecture import Model
from model.Model_Interface_MC import ModelInterface
from cf.GradCFA_MC import GradCFA
from eval.Evaluation import Evaluate

N_CF = 5
K = 3
HP = dict(  # the paper's Obesity settings (lambda 0.5, gamma_pen 0.1, gamma_pert 0.5, tau_prox=tau_spars 0.3, tau_plaus=tau_div 0.4, tau_loss 0.4)
    proximity_weight=0.5, sparsity_weight=0.5, plausibility_weight=0.5, diversity_weight=0.5,
    penalty_scale=0.1, perturb_scale=0.5,
    proximity_threshold=0.3, sparsity_threshold=0.3,
    plausibility_threshold=0.4, diversity_threshold=0.4,
    high_loss_threshold=0.4,
)
CONFIGS = {
    "A": dict(HP, loss_converge_maxiter=2),                                     # the value the code comment says the paper used
    "C": dict(HP, loss_converge_maxiter=1),                                     # perturbation can really run (slow)
    "C_nopert": dict(HP, loss_converge_maxiter=1, high_loss_threshold=1e9),     # perturbation off (fast)
    "C_nopen": dict(HP, loss_converge_maxiter=1, penalty_scale=0.0),
    "C_nopert_nopen": dict(HP, loss_converge_maxiter=1, high_loss_threshold=1e9, penalty_scale=0.0),
}
PAPER = {"prox": 0.21, "spars": 0.23, "plaus_eq5_k3": 0.34, "div": 0.92, "conf": 0.76, "avg5": 0.78}


def prepare():
    src = open("src/cf/GradCFA_MC.py", encoding="utf-8").read()
    if "self.importance_ranking" not in src:
        sys.exit("Run  python patch_gradcfa_mc.py  first (src/cf/GradCFA_MC.py is not patched).")
    df, d, Z, y = load_data()
    Z_train, Z_val, Z_test, y_train, y_val, y_test = split(Z, y)
    model = Model(Z.shape[1], N_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        pred_all = model(torch.tensor(Z.to_numpy(dtype=np.float32))).argmax(1).numpy()
    pos = Z.index.get_indexer(Z_test.index)
    acc = float((pred_all[pos] == y_test.values).mean())
    if os.path.exists(META_PATH):
        exp_acc = float(open(META_PATH, encoding="utf-8").readline().strip())
        if abs(acc - exp_acc) > 1e-6:
            print(f"WARNING: test accuracy {acc:.6f} differs from the training-time value {exp_acc:.6f}. Re-run obesity_train.py.")
    queries = []
    for rid, p in zip(Z_test.index, pos):
        pr = int(pred_all[p])
        if pr == TARGET_CLASS:
            continue
        queries.append((int(rid), df.loc[rid].drop("target").to_list(), pr, int(y_test.loc[rid])))
    print(f"Test rows: {len(Z_test)} | queries (not predicted as class 1): {len(queries)} | model test accuracy: {acc:.6f}")
    m = ModelInterface(model_path=MODEL_PATH, input_size=Z.shape[1], output_size=N_CLASSES)
    obs = torch.tensor(Z.to_numpy(dtype=np.float32))
    return dict(d=d, m=m, model=model, obs=obs, queries=queries)


def plausibility(dist, k=K):
    vals, _ = torch.topk(dist, k, largest=False)
    raw = vals.mean().item()
    dmin = vals.min(dim=1, keepdim=True).values
    dmax = vals.max(dim=1, keepdim=True).values
    return raw, ((vals - dmin) / (dmax - dmin + 1e-8)).mean().item()


def metrics_for(ctx, ev, q, cfs):
    q = torch.Tensor(np.asarray(q)).float()
    c = torch.Tensor(np.asarray(cfs)).float()
    raw, eq5 = plausibility(torch.cdist(c, ctx["obs"], p=1.0))
    with torch.no_grad():
        logits = ctx["model"](c)
        prob = torch.softmax(logits, dim=1)[:, TARGET_CLASS]
        valid = (logits.argmax(1) == TARGET_CLASS).float().mean().item()
    return dict(prox=ev.evaluate_proximity(q, c).item(), spars=ev.evaluate_sparsity(q, c).item(),
                div=ev.evaluate_diversity(c).item(), plaus_raw_k3=raw, plaus_eq5_k3=eq5,
                conf=prob.mean().item(), valid_frac=valid)


def run(config, seeds, limit, shard):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ctx = prepare()
    ev = Evaluate(ctx["d"], ctx["m"])
    kwargs = CONFIGS[config]
    si, sn = (int(x) for x in shard.split("/"))
    tag = f"{config}_s{si}of{sn}"
    out_csv = os.path.join(RESULTS_DIR, f"full_{tag}.csv")
    fa_csv = os.path.join(RESULTS_DIR, f"fa_{tag}.csv")
    err_txt = os.path.join(RESULTS_DIR, f"errors_{tag}.txt")
    cf_dir = os.path.join(RESULTS_DIR, f"cfs_{config}")
    os.makedirs(cf_dir, exist_ok=True)
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["seed"].astype(int), prev["row_id"].astype(int)))
    queries = ctx["queries"][:limit] if limit else ctx["queries"]
    queries = queries[si::sn]
    for seed in seeds:
        todo = [t for t in queries if (seed, t[0]) not in done]
        print(f"\n[{tag}] seed {seed}: {len(todo)} to run ({len(queries) - len(todo)} already done)")
        t_start = time.time()
        for i, (rid, ql, pred_cls, true_cls) in enumerate(todo, 1):
            t0 = time.time()
            try:
                seed_all(seed * 1000 + rid)
                exp = GradCFA(ctx["d"], ctx["m"])
                with contextlib.redirect_stdout(io.StringIO()):
                    q, cfs = exp.generate_counterfactuals(ql, total_CFs=N_CF, desired_value=TARGET_CLASS,
                                                          problem_type="multiclass", **kwargs)
                plt.close("all")
                row = metrics_for(ctx, ev, q, cfs)
                np.savez_compressed(os.path.join(cf_dir, f"s{seed}_r{rid}.npz"), q=np.asarray(q), cfs=np.asarray(cfs))
                row.update(config=config, seed=seed, row_id=rid, orig_class=pred_cls, true_class=true_cls,
                           seconds=round(time.time() - t0, 2))
                pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                fa = [dict(config=config, seed=seed, row_id=rid, feature=n, score=float(s)) for n, s in exp.importance_ranking]
                pd.DataFrame(fa).to_csv(fa_csv, mode="a", header=not os.path.exists(fa_csv), index=False)
            except Exception as e:
                plt.close("all")
                msg = f"seed {seed} row {rid}: {type(e).__name__}: {e}"
                print("ERROR", msg)
                with open(err_txt, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                continue
            finally:
                gc.collect()
            el = time.time() - t_start
            print(f"  [{i}/{len(todo)}] row {rid} class {pred_cls}->1 {time.time() - t0:5.1f}s  elapsed {el / 60:5.1f} min  ETA {el / i * (len(todo) - i) / 60:5.1f} min")
    print("\nFinished. Results in", out_csv)


def summarize(config):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"full_{config}_s*of*.csv")))
    fas = sorted(glob.glob(os.path.join(RESULTS_DIR, f"fa_{config}_s*of*.csv")))
    if not files:
        sys.exit("No result files found for config " + config)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["avg5"] = ((1 - df["prox"]) + (1 - df["spars"]) + (1 - df["plaus_eq5_k3"]) + df["div"] + df["conf"]) / 5
    cols = ["prox", "spars", "plaus_eq5_k3", "plaus_raw_k3", "div", "conf", "valid_frac", "avg5", "seconds"]
    per_seed = df.groupby("seed")[cols].mean()
    tab = pd.DataFrame({"mean": per_seed.mean(), "std_across_seeds": per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan})
    tab["paper (GradCFA, Obesity)"] = pd.Series(PAPER)
    txt = [f"Config {config} | files: {[os.path.basename(f) for f in files]}",
           f"Seeds: {list(per_seed.index)} | queries per seed: {df.groupby('seed').size().to_dict()}",
           f"Share of queries with no valid counterfactual: {(df['valid_frac'] == 0).mean():.3f}", "",
           tab.round(4).to_string(), "", "Per original class (class the model predicts), mean over seeds and queries:"]
    pc = df.groupby("orig_class")[["prox", "spars", "plaus_eq5_k3", "div", "conf", "valid_frac", "avg5"]].mean()
    pc.insert(0, "n_rows", df.groupby("orig_class").size())
    txt.append(pc.round(3).to_string())
    if fas:
        fa = pd.concat([pd.read_csv(f) for f in fas], ignore_index=True)
        fa["rank"] = fa.groupby(["seed", "row_id"])["score"].rank(ascending=False, method="first")
        g = fa.groupby("feature").agg(mean_score=("score", "mean"), mean_rank=("rank", "mean"),
                                      top1_share=("rank", lambda r: (r == 1).mean())).sort_values("mean_rank")
        txt += ["", "Feature attribution (GradCFA's own scores), paper: Weight first, then MTRANS:", g.round(4).to_string()]
    out = "\n".join(txt)
    print(out)
    p = os.path.join(RESULTS_DIR, f"summary_{config}.txt")
    open(p, "w", encoding="utf-8").write(out + "\n")
    print("\nSaved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="C_nopert")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: this process runs queries i, i+n, i+2n ...")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per process (1 is usually fastest for these tiny tensors)")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    summarize(a.config) if a.summarize else run(a.config, a.seeds, a.limit, a.shard)
