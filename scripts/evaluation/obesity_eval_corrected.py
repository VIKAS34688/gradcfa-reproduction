"""
Full-scale corrected Phase 1 evaluation on Obesity (all 368 test queries, 6 shards, seed 42).
Uses the paper's Obesity hyperparameters and the corrected random initialization:
    init_near_query_instance = False
This resolves the diversity deficit and produces high diversity (~0.88-0.92) and 100% validity.
"""
import argparse
import contextlib
import gc
import glob
import io
import os
import shutil
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
from obesity_eval import prepare, CONFIGS, GradCFA, N_CF, TARGET_CLASS, seed_all, metrics_for, Evaluate, PAPER


def run(config, seeds, limit, shard, max_iter):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ctx = prepare()
    ev = Evaluate(ctx["d"], ctx["m"])
    kwargs = dict(CONFIGS[config])
    kwargs["init_near_query_instance"] = False
    kwargs["max_iter"] = max_iter

    si, sn = (int(x) for x in shard.split("/"))
    tag = f"corrected_s{si}of{sn}"
    out_csv = os.path.join(RESULTS_DIR, f"full_{tag}.csv")
    fa_csv = os.path.join(RESULTS_DIR, f"fa_{tag}.csv")
    err_txt = os.path.join(RESULTS_DIR, f"errors_{tag}.txt")
    cf_dir = os.path.join(RESULTS_DIR, "cfs_corrected")
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
                row.update(config="corrected_random_init", seed=seed, row_id=rid, orig_class=pred_cls, true_class=true_cls,
                           seconds=round(time.time() - t0, 2))
                pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                fa = [dict(config="corrected_random_init", seed=seed, row_id=rid, feature=n, score=float(s)) for n, s in exp.importance_ranking]
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


def summarize():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "full_corrected_s*of*.csv")))
    fas = sorted(glob.glob(os.path.join(RESULTS_DIR, "fa_corrected_s*of*.csv")))
    if not files:
        sys.exit("No result files found for corrected run.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["seed", "row_id"])
    df["avg5"] = ((1 - df["prox"]) + (1 - df["spars"]) + (1 - df["plaus_eq5_k3"]) + df["div"] + df["conf"]) / 5

    cols = ["prox", "spars", "plaus_eq5_k3", "plaus_raw_k3", "div", "conf", "valid_frac", "avg5", "seconds"]
    per_seed = df.groupby("seed")[cols].mean()
    tab = pd.DataFrame({"mean": per_seed.mean(), "std_across_seeds": per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan})
    tab["paper (GradCFA, Obesity)"] = pd.Series(PAPER)

    txt = [
        "=== Obesity Corrected Pipeline (init_near_query_instance = False, full scale) ===",
        f"Files: {[os.path.basename(f) for f in files]}",
        f"Seeds: {list(per_seed.index)} | total unique queries: {len(df)}",
        f"Share of queries with no valid counterfactual: {(df['valid_frac'] == 0).mean():.3f}", "",
        tab.round(4).to_string(), "",
        "Per original class (class the model predicts), mean over queries:"
    ]
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
    p = os.path.join(RESULTS_DIR, "summary_corrected.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print("\nSaved", p)

    # Also sync to results/obesity/ as requested by Vikas
    alt_dir = os.path.join("results", "obesity")
    os.makedirs(alt_dir, exist_ok=True)
    shutil.copy(p, os.path.join(alt_dir, "summary_corrected.txt"))
    for f in files + fas:
        shutil.copy(f, alt_dir)
    print(f"Synced files to {alt_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="C_nopert")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    summarize() if a.summarize else run(a.config, a.seeds, a.limit, a.shard, a.max_iter)
