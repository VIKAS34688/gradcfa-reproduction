"""
Reproduction of the paper's Table VII (Fixed-Feature Regeneration Test) on Obesity.
Tests whether fixing the top feature (Weight) destroys counterfactual validity.
In the paper's Table VII: validity drops from ~0.92 down to 0.18 when the top feature is fixed.
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
from obesity_eval import prepare, CONFIGS, GradCFA, N_CF, TARGET_CLASS, seed_all, metrics_for, Evaluate

PAPER_TABLE7_VALIDITY = 0.18   # Table VII in the paper


def run(config, seeds, limit, fixed_feature, max_iter, shard=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ctx = prepare()
    ev = Evaluate(ctx["d"], ctx["m"])
    kwargs = dict(CONFIGS[config])
    kwargs["init_near_query_instance"] = False
    kwargs["max_iter"] = max_iter

    if shard:
        si, sn = (int(x) for x in shard.split("/"))
        tag = f"fixed_{fixed_feature}_s{si}of{sn}"
    else:
        tag = f"fixed_{fixed_feature}"

    out_csv = os.path.join(RESULTS_DIR, f"table7_{tag}.csv")
    err_txt = os.path.join(RESULTS_DIR, f"errors_{tag}.txt")

    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["seed"].astype(int), prev["row_id"].astype(int)))

    queries = ctx["queries"][:limit] if limit else ctx["queries"]
    if shard:
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
                                                          problem_type="multiclass", immutable_features=[fixed_feature],
                                                          **kwargs)
                plt.close("all")
                row = metrics_for(ctx, ev, q, cfs)
                row.update(config=config, seed=seed, row_id=rid, orig_class=pred_cls, true_class=true_cls,
                           fixed_feature=fixed_feature, seconds=round(time.time() - t0, 2))
                pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
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
            print(f"  [{i}/{len(todo)}] row {rid} class {pred_cls}->1 fixed={fixed_feature} {time.time() - t0:5.1f}s  elapsed {el / 60:5.1f} min  ETA {el / i * (len(todo) - i) / 60:5.1f} min")
    print("\nFinished. Results in", out_csv)


def summarize(fixed_feature):
    fixed_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"table7_fixed_{fixed_feature}*.csv")))
    if not fixed_files:
        sys.exit(f"No result file found matching: table7_fixed_{fixed_feature}*.csv")
    df_fix = pd.concat([pd.read_csv(f) for f in fixed_files], ignore_index=True)
    df_fix = df_fix.drop_duplicates(subset=["seed", "row_id"])

    base_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "full_corrected_s*of*.csv")))
    if not base_files:
        base_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "full_C_nopert_s*of*.csv")))
    if not base_files:
        sys.exit("No baseline Phase 1 files found.")
    df_base = pd.concat([pd.read_csv(f) for f in base_files], ignore_index=True)
    df_base = df_base.drop_duplicates(subset=["seed", "row_id"])

    merged = pd.merge(df_fix, df_base[["seed", "row_id", "valid_frac", "conf", "prox", "spars", "div"]],
                      on=["seed", "row_id"], suffixes=(f"_{fixed_feature}_fixed", "_baseline"))

    v_base = merged["valid_frac_baseline"].mean()
    v_fix = merged[f"valid_frac_{fixed_feature}_fixed"].mean()
    v_drop = v_base - v_fix

    txt = [
        f"=== Table VII Reproduction: Fixed-Feature Test on Obesity (Feature: {fixed_feature}) ===",
        f"Queries evaluated: {len(merged)} | Seed: {list(merged['seed'].unique())}",
        "",
        f"Baseline Validity (Phase 1, {fixed_feature} free):     {v_base:.4f}",
        f"Fixed-Feature Validity ({fixed_feature} locked):       {v_fix:.4f}",
        f"Validity Drop:                                        {v_drop:.4f}",
        f"Paper's Fixed-Feature Validity (Table VII):           {PAPER_TABLE7_VALIDITY:.4f}",
        "",
        "--- Per Original Class Breakdown ---"
    ]

    pc = merged.groupby("orig_class").agg(
        n_queries=("row_id", "count"),
        valid_baseline=("valid_frac_baseline", "mean"),
        valid_fixed=(f"valid_frac_{fixed_feature}_fixed", "mean"),
        conf_baseline=("conf_baseline", "mean"),
        conf_fixed=(f"conf_{fixed_feature}_fixed", "mean")
    )
    pc["validity_drop"] = pc["valid_baseline"] - pc["valid_fixed"]
    txt.append(pc.round(4).to_string())

    out = "\n".join(txt)
    print(out)
    out_txt = os.path.join(RESULTS_DIR, f"summary_table7_fixed_{fixed_feature}.txt")
    open(out_txt, "w", encoding="utf-8").write(out + "\n")
    print("\nSaved", out_txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="C_nopert")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="Shard in format i/N, e.g., 0/6")
    ap.add_argument("--fixed", default="Weight", help="Feature to fix as immutable (paper Table VII: Weight)")
    ap.add_argument("--max-iter", type=int, default=3000, help="Max optimization iterations per query")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    summarize(a.fixed) if a.summarize else run(a.config, a.seeds, a.limit, a.fixed, a.max_iter, a.shard)
