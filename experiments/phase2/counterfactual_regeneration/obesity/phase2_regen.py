"""
Phase 2, test T3: fixed-feature regeneration (extends the paper's Table VI to per-query rankings).

The paper fixes a feature and shows that counterfactual generation gets harder. Here, for every query,
we fix the top-k features of each ranking, GENERATE the counterfactuals again with GradCFA, and measure how
many of the 5 counterfactuals are still valid. A more faithful ranking points at features the counterfactuals
really depend on, so fixing them should destroy more validity. This test runs the real generator, so it does
not depend on the integrated-gradient path used by Path-IG (unlike T2).

Rankings: GradCFA, Shapley, Path-IG, Random (one random order per query).
Setting: the 69 queries the model rejects, seed 42, perturbation off (config A, fast: about 30 s per run).
The baseline (no feature fixed) is the already saved run results_di/full_A_all.csv, seed 42.

Run from the repository root, after phase2_attribution.py has produced results_di/phase2_shapley_cache.csv:
    python phase2_regen.py --limit 3 --k 1      # quick test (3 queries, k = 1)
    python phase2_regen.py                      # all 69 queries, k = 1 and 3 (about 4 to 5 hours, resumable)
    python phase2_regen.py --k 3 --rankings Path-IG Random   # resume only some rankings, in a fresh process
    python phase2_regen.py --summarize          # table with 95% bootstrap intervals
"""
import argparse
import contextlib
import gc
import io
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("src"))
import phase2_attribution as p2

SEED = 42
CF_DIR = "results_di/cfs_A_all"
FA = "results_di/fa_A_all.csv"
BASE = "results_di/full_A_all.csv"
CACHE = "results_di/phase2_shapley_cache.csv"
OUT = "results_di/phase2_regen.csv"
SUMMARY = "results_di/phase2_regen_summary.txt"
RANKINGS = ["GradCFA", "Shapley", "Path-IG", "Random"]


def load_cache(features):
    cache = {}
    if os.path.exists(CACHE):
        c = pd.read_csv(CACHE)
        for r, g in c.groupby("row_id"):
            s = g.set_index("feature")["phi"].reindex(features)
            if not s.isna().any():
                cache[int(r)] = s.to_numpy(dtype=float)
    return cache


def build_rankings():
    """Returns (feature names, {row: {ranking name: ordered list of feature indices}})."""
    model = p2.MLP(p2.load_weights(p2.MODEL_PATH))
    Ztr, groups = p2.build_space()
    features = list(groups.keys())
    groups_idx = [groups[f] for f in features]
    bg = Ztr[np.random.default_rng(42).choice(len(Ztr), p2.N_BG, replace=False)]
    sets = p2.load_sets(CF_DIR, SEED)
    fa = p2.load_fa(FA, SEED, features)
    cache = load_cache(features)
    out = {}
    for row, (q, cfs) in sets.items():
        if row not in fa:
            continue
        phi = cache[row] if row in cache else p2.exact_shapley(model, q, bg, groups_idx)
        scores = {
            "GradCFA": fa[row],
            "Shapley": np.abs(phi),
            "Path-IG": p2.path_ig(model, q, cfs, groups_idx),
        }
        order = {name: p2.order_of(s) for name, s in scores.items()}
        order["Random"] = list(np.random.default_rng(SEED * 100000 + row).permutation(len(features)))
        out[row] = order
    return features, out


def run(ks, limit, rankings_to_run=None):
    import matplotlib.pyplot as plt
    import run_full_eval as rfe

    features, rankings = build_rankings()
    ctx = rfe.prepare("neg")
    ev = rfe.Evaluate(ctx["d"], ctx["m"])
    kwargs = dict(rfe.CONFIGS["A"])
    queries = [(rid, ql, tgt) for rid, ql, tgt in ctx["queries"] if rid in rankings]
    if limit:
        queries = queries[:limit]
    rankings_to_run = rankings_to_run or RANKINGS
    print(f"{len(queries)} queries, rankings {rankings_to_run}, k = {ks}")

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev["ranking"], prev["k"].astype(int), prev["row_id"].astype(int)))

    for k in ks:
        for name in rankings_to_run:
            todo = [(rid, ql, tgt) for rid, ql, tgt in queries if (name, k, rid) not in done]
            print(f"\n[k={k}] ranking {name}: {len(todo)} queries to run ({len(queries) - len(todo)} done)")
            t_start = time.time()
            for i, (rid, ql, tgt) in enumerate(todo, 1):
                t0 = time.time()
                fixed = [features[j] for j in rankings[rid][name][:k]]
                exp = q = cfs = None
                try:
                    rfe.seed_all(SEED * 1000 + rid)
                    exp = rfe.DiPACE_FA(ctx["d"], ctx["m"])
                    with contextlib.redirect_stdout(io.StringIO()):
                        q, cfs = exp.generate_counterfactuals(ql, total_CFs=rfe.N_CF, immutable_features=fixed, **kwargs)
                    plt.close("all")
                    row = rfe.metrics_for(ctx, ev, q, cfs, tgt)
                    row.update(ranking=name, k=k, row_id=rid, fixed="|".join(fixed), seconds=round(time.time() - t0, 2))
                    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
                except Exception as e:
                    plt.close("all")
                    print(f"ERROR row {rid} {name} k={k}: {type(e).__name__}: {e}")
                    gc.collect()
                    continue
                finally:
                    del exp, q, cfs
                    gc.collect()   # free memory after every query (a very long single process can run out of RAM)
                elapsed = time.time() - t_start
                eta = elapsed / i * (len(todo) - i)
                print(f"  [{i}/{len(todo)}] row {rid} fixed {fixed}  {time.time() - t0:5.1f}s  ETA {eta / 60:5.1f} min")
    print("\nFinished. Results in", OUT)


def summarize():
    df = pd.read_csv(OUT)
    base = pd.read_csv(BASE)
    base = base[base.seed == SEED].set_index("row_id")["valid_frac"]
    lines = ["Phase 2, test T3: fixed-feature regeneration (perturbation off, seed 42, queries the model rejects)",
             "Drop = valid fraction without fixing minus valid fraction with the top-k features fixed (larger = more faithful ranking).",
             "95% intervals are bootstrap over queries.", ""]
    for k in sorted(df.k.unique()):
        d = df[df.k == k]
        sets_of_rows = [set(g.row_id) for _, g in d.groupby("ranking")]
        if len(sets_of_rows) < len(RANKINGS):
            missing = [n for n in RANKINGS if n not in set(d.ranking)]
            lines += [f"k = {k}: not summarised yet, no results for {missing}", ""]
            continue
        rows = sorted(set.intersection(*sets_of_rows))
        if not rows:
            continue
        idx = p2.boot_idx(len(rows))
        b = base.loc[rows].to_numpy(dtype=float)
        drops, fixed_valid, allfail = {}, {}, {}
        for name in RANKINGS:
            g = d[d.ranking == name].set_index("row_id").loc[rows]
            fv = g["valid_frac"].to_numpy(dtype=float)
            drops[name] = b - fv
            fixed_valid[name] = fv
            allfail[name] = float((fv == 0).mean())
        lines.append(f"k = {k} fixed features ({len(rows)} queries; valid fraction without fixing: {b.mean():.3f})")
        for name in RANKINGS:
            m, lo, hi = mean_ci(drops[name], idx)
            lines.append(f"  {name:8s} valid fraction {fixed_valid[name].mean():.3f}   drop {m:+.3f} [{lo:+.3f}, {hi:+.3f}]   queries with no valid counterfactual {allfail[name]:.2f}")
        for a, c in [("Path-IG", "GradCFA"), ("Shapley", "GradCFA"), ("Path-IG", "Shapley"), ("GradCFA", "Random")]:
            m, lo, hi = mean_ci(drops[a] - drops[c], idx)
            lines.append(f"  paired drop {a} - {c}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("Saved", SUMMARY)


def mean_ci(arr, idx):
    m, lo, hi = p2.mean_ci(arr, idx)
    return float(m), float(lo), float(hi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rankings", nargs="+", choices=RANKINGS, default=None,
                    help="run only these rankings (use to resume in a fresh process)")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args.k, args.limit, args.rankings)