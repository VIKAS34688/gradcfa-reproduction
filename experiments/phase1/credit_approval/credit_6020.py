"""
Credit Approval with the paper's 60/20/20 split (train / validation / test) and validation-based tuning.

Why: the paper uses a 60/20/20 split and tunes on the test set. Here we use the paper's split, and we tune
ONLY on the validation rows, then report the test result next to the untuned one.

Everything runs from the repository root. All results go to results_6020/.

  python credit_6020.py train
      60/20/20 split (414 / 138 / 138 rows), trains the 64-32 model in DataInterface space, prints train,
      validation and test accuracy next to the paper's 0.88 / 0.86 / 0.82, saves the model and the row ids.

  python credit_6020.py run --split test --config A --seeds 42 --which all
      Generate counterfactuals for every test row (target = opposite of the model's prediction), 5 metrics.
      --limit 3 for a quick test. Finished queries are skipped when you run it again.

  python credit_6020.py summarize --split test --tag A
      Mean and standard deviation across seeds, next to the paper's Credit row.

  python credit_6020.py compare
      Instant, no generation: paired comparison of every finished tuning setting with the base setting.

  python credit_6020.py tune
      Validation-only grid (rejected rows, seed 42, perturbation off, about 30 minutes per setting).
      Prints a table and the command to run the best setting on the test rows.

Settings can be changed on the command line, for example:
  python credit_6020.py run --split val --which neg --tag try1 --set plausibility_weight=1.0 penalty_scale=0.5
"""
import argparse
import contextlib
import gc
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pathlib import Path
# Locate the repository root regardless of this script's subdirectory.
from pathlib import Path

def _find_project_root(start):
    for candidate in (start, *start.parents):
        if (candidate / "README.md").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate the GradCFs repository root.")

PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT / "scripts/evaluation") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts/evaluation"))

# Preserve the repository's existing relative data/results paths.
import os
import sys
os.chdir(PROJECT_ROOT)

for _path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "src",
):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

OUT = "results_6020"
SPLIT_JSON = os.path.join(OUT, "split.json")
META_JSON = os.path.join(OUT, "model_meta.json")
MODEL = PROJECT_ROOT / "models" / "checkpoints" / "pytorch_model_6020.pth"
CONT = ["Age", "Debt", "YearsEmployed", "Income"]
PAPER_ACC = {"train": 0.88, "val": 0.86, "test": 0.82}
PAPER_ROW = {"prox": 0.15, "spars": 0.18, "plaus_eq5_k3": 0.37, "div": 0.96, "conf": 0.78, "avg5": 0.81}

# validation-only tuning grid (changes to the paper's Credit settings, perturbation off)
GRID = {
    "base": {},
    "plaus_w1": {"plausibility_weight": 1.0},
    "plaus_w2": {"plausibility_weight": 2.0},
    "plaus_w1_t1.5": {"plausibility_weight": 1.0, "plausibility_threshold": 1.5},
    "pen_0.5": {"penalty_scale": 0.5},
    "prox_spars_w1": {"proximity_weight": 1.0, "sparsity_weight": 1.0},
    "div_w1": {"diversity_weight": 1.0},
}


# ---------------------------------------------------------------- data and split
def load_df():
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv("data/raw/credit.csv").drop(columns="ZipCode")
    for col in ["Industry", "Ethnicity", "Citizen"]:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


def make_split(index, seed=42):
    """60/20/20 split of the row labels. Returns three lists of row labels."""
    from sklearn.model_selection import train_test_split
    train, tmp = train_test_split(list(index), test_size=0.4, random_state=seed)
    val, test = train_test_split(tmp, test_size=0.5, random_state=seed)
    return [int(i) for i in train], [int(i) for i in val], [int(i) for i in test]


def seed_all(s):
    import random
    import torch
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)


# ---------------------------------------------------------------- train
def cmd_train(epochs):
    import torch
    import torch.nn as nn
    from arc.Binary_Architecture import Model
    from data.Data_Interface import DataInterface

    df = load_df()
    d = DataInterface(dataframe=df, target="Approved", continuous_features=CONT)
    Z, y = d.norm_encoded_features, df["Approved"]
    train, val, test = make_split(Z.index)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"train": train, "val": val, "test": test}, open(SPLIT_JSON, "w"))
    print(f"split sizes: train {len(train)}, validation {len(val)}, test {len(test)}")

    seed_all(42)
    X = {k: torch.tensor(Z.loc[ids].to_numpy(dtype=np.float32)) for k, ids in [("train", train), ("val", val), ("test", test)]}
    Y = {k: y.loc[ids].to_numpy(dtype=np.float32) for k, ids in [("train", train), ("val", val), ("test", test)]}
    model = Model(Z.shape[1])
    crit = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    ytr = torch.tensor(Y["train"]).view(-1, 1)
    for epoch in range(epochs):
        opt.zero_grad()
        loss = crit(model(X["train"]), ytr)
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    model.eval()
    acc = {}
    with torch.no_grad():
        for k in X:
            acc[k] = float(((model(X[k]).view(-1).numpy() >= 0.5).astype(np.float32) == Y[k]).mean())
    print("\naccuracy        ours     paper")
    for k in ["train", "val", "test"]:
        print(f"  {k:6s}      {acc[k]:.4f}   {PAPER_ACC[k]:.2f}")
    torch.save(model.state_dict(), MODEL)
    json.dump({"accuracy": acc, "epochs": epochs}, open(META_JSON, "w"))
    print(f"\nSaved {MODEL}, {SPLIT_JSON}, {META_JSON}")


# ---------------------------------------------------------------- run
def prepare(split, which):
    import torch
    import run_full_eval as rfe  # noqa: F401  (also applies the non-interactive matplotlib backend)
    from arc.Binary_Architecture import Model
    from data.Data_Interface import DataInterface
    from model.Model_Interface import ModelInterface

    df = load_df()
    d = DataInterface(dataframe=df, target="Approved", continuous_features=CONT)
    Z = d.norm_encoded_features
    rows = json.load(open(SPLIT_JSON))[split]
    model = Model(Z.shape[1])
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(Z.loc[rows].to_numpy(dtype=np.float32))).view(-1).numpy()
    pred = (probs >= 0.5).astype(int)
    queries = []
    for rid, p in zip(rows, pred):
        if which == "neg" and p != 0:
            continue
        queries.append((int(rid), df.loc[rid].drop("Approved").to_list(), 1 - int(p)))
    m = ModelInterface(model_path=MODEL, input_size=Z.shape[1])
    obs = torch.tensor(Z.to_numpy(dtype=np.float32))
    print(f"split {split}: {len(rows)} rows, {len(queries)} queries ({which})")
    return dict(d=d, m=m, model=model, obs=obs, queries=queries)


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = float(v)
    return out


def cmd_run(split, config, seeds, which, limit, tag, overrides, grid_name=None):
    import matplotlib.pyplot as plt
    import run_full_eval as rfe

    kwargs = dict(rfe.CONFIGS[config])
    kwargs.update(GRID.get(grid_name, {}) if grid_name else {})
    kwargs.update(overrides)
    tag = tag or (grid_name or config)
    ctx = prepare(split, which)
    ev = rfe.Evaluate(ctx["d"], ctx["m"])
    out_csv = os.path.join(OUT, f"{split}_{tag}.csv")
    os.makedirs(OUT, exist_ok=True)
    json.dump({k: v for k, v in kwargs.items()}, open(os.path.join(OUT, f"{split}_{tag}_settings.json"), "w"))

    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["seed"].astype(int), prev["row_id"].astype(int)))
    queries = ctx["queries"][:limit] if limit else ctx["queries"]

    for seed in seeds:
        todo = [(rid, ql, tgt) for rid, ql, tgt in queries if (seed, rid) not in done]
        print(f"\n[{split}/{tag}] seed {seed}: {len(todo)} queries to run ({len(queries) - len(todo)} done)")
        t_start = time.time()
        for i, (rid, ql, tgt) in enumerate(todo, 1):
            t0 = time.time()
            exp = q = cfs = None
            try:
                rfe.seed_all(seed * 1000 + rid)
                exp = rfe.DiPACE_FA(ctx["d"], ctx["m"])
                with contextlib.redirect_stdout(io.StringIO()):
                    q, cfs = exp.generate_counterfactuals(ql, total_CFs=rfe.N_CF, **kwargs)
                plt.close("all")
                row = rfe.metrics_for(ctx, ev, q, cfs, tgt)
                row.update(seed=seed, row_id=rid, seconds=round(time.time() - t0, 2))
                pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            except Exception as e:
                plt.close("all")
                print(f"ERROR row {rid}: {type(e).__name__}: {e}")
                continue
            finally:
                del exp, q, cfs
                gc.collect()
            elapsed = time.time() - t_start
            print(f"  [{i}/{len(todo)}] row {rid} -> class {tgt}  {time.time() - t0:5.1f}s  ETA {elapsed / i * (len(todo) - i) / 60:5.1f} min")
    print("\nFinished. Results in", out_csv)


# ---------------------------------------------------------------- summaries and tuning
def read_results(path):
    df = pd.read_csv(path)
    df["avg5"] = ((1 - df["prox"]) + (1 - df["spars"]) + (1 - df["plaus_eq5_k3"]) + df["div"] + df["conf"]) / 5
    return df


def cmd_summarize(split, tag):
    df = read_results(os.path.join(OUT, f"{split}_{tag}.csv"))
    cols = ["prox", "spars", "plaus_eq5_k3", "plaus_raw_k3", "div", "conf", "valid_frac", "avg5", "seconds"]
    per_seed = df.groupby("seed")[cols].mean()
    table = pd.DataFrame({"mean": per_seed.mean(), "std_across_seeds": per_seed.std(ddof=1)})
    table["paper (GradCFA, Credit)"] = pd.Series(PAPER_ROW)
    text = (f"{split}/{tag}\nSeeds: {list(per_seed.index)}\nQueries per seed: {df.groupby('seed').size().to_dict()}\n\n"
            f"{table.round(4).to_string()}\n")
    print(text)
    with open(os.path.join(OUT, f"summary_{split}_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(text)


def cmd_compare(seed=42, base="base"):
    """Paired comparison of every tuning setting with the base setting on the same validation queries."""
    import glob
    base_path = os.path.join(OUT, f"val_{base}.csv")
    b = read_results(base_path)
    b = b[b.seed == seed].set_index("row_id")
    rng = np.random.default_rng(0)
    lines = []
    for f in sorted(glob.glob(os.path.join(OUT, "val_*.csv"))):
        name = os.path.basename(f)[4:-4]
        if name == base:
            continue
        d = read_results(f)
        d = d[d.seed == seed].set_index("row_id")
        common = sorted(set(d.index) & set(b.index))
        if len(common) < 5:
            continue
        diff = d.loc[common, "avg5"].to_numpy() - b.loc[common, "avg5"].to_numpy()
        vdiff = d.loc[common, "valid_frac"].to_numpy() - b.loc[common, "valid_frac"].to_numpy()
        boots = diff[rng.integers(0, len(diff), (2000, len(diff)))].mean(axis=1)
        lines.append(dict(setting=name, queries=len(common), avg5_setting=round(d.loc[common, "avg5"].mean(), 4),
                          avg5_base=round(b.loc[common, "avg5"].mean(), 4), diff=round(diff.mean(), 4),
                          ci_low=round(float(np.percentile(boots, 2.5)), 4), ci_high=round(float(np.percentile(boots, 97.5)), 4),
                          valid_diff=round(vdiff.mean(), 3)))
    if not lines:
        print("Nothing to compare yet.")
        return
    t = pd.DataFrame(lines).sort_values("diff", ascending=False)
    print("\nPaired comparison with the base setting, on the same validation queries (95% bootstrap interval of the difference):")
    print(t.to_string(index=False))
    print("A setting is only worth using if its interval excludes zero and its valid fraction does not drop.")
    t.to_csv(os.path.join(OUT, "tuning_paired.csv"), index=False)


def cmd_tune(seed):
    rows = []
    for name in GRID:
        cmd_run("val", "A", [seed], "neg", None, None, {}, grid_name=name)
        df = read_results(os.path.join(OUT, f"val_{name}.csv"))
        df = df[df.seed == seed]
        rows.append(dict(setting=name, queries=len(df), **df[["prox", "spars", "plaus_eq5_k3", "div", "conf", "valid_frac", "avg5"]].mean().round(4)))
    table = pd.DataFrame(rows).sort_values("avg5", ascending=False)
    print("\nValidation results (rejected rows, perturbation off, seed %d), best first:" % seed)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(OUT, "tuning_validation.csv"), index=False)
    best = table.iloc[0]["setting"]
    print(f"\nBest on validation: {best}. To report the test result run:")
    print(f"  python credit_6020.py run --split test --config A --which all --seeds 42 --tag {best}"
          + (f" --grid {best}" if best != "base" else ""))
    print("and compare it with the untuned setting (--tag base).")
    cmd_compare(seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "run", "summarize", "tune", "compare"])
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--config", default="A")
    ap.add_argument("--grid", default=None, help="name of a setting from the tuning grid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--which", choices=["neg", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", nargs="+", default=None, help="overrides such as plausibility_weight=1.0")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    if args.command == "train":
        cmd_train(args.epochs)
    elif args.command == "run":
        cmd_run(args.split, args.config, args.seeds, args.which, args.limit, args.tag, parse_overrides(args.set), args.grid)
    elif args.command == "summarize":
        cmd_summarize(args.split, args.tag or args.config)
    elif args.command == "compare":
        cmd_compare(args.seeds[0])
    else:
        cmd_tune(args.seeds[0])
