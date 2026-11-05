"""
Runs 6 parallel shards to evaluate the Table VII Control Experiment (Fixed Feature = NCP)
across all 368 Obesity queries, managing the worker processes and streaming summary stats.
"""
import os
import sys
import subprocess
import time
import glob
import pandas as pd

PYTHON = sys.executable
DIR = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(DIR, "logs")
RESULTS_DIR = os.path.join(DIR, "results_obesity")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE = "NCP"
NUM_SHARDS = 6

processes = []
print(f"Starting {NUM_SHARDS} parallel worker processes for feature '{FEATURE}'...")

for i in range(NUM_SHARDS):
    shard_str = f"{i}/{NUM_SHARDS}"
    log_file = os.path.join(LOGS_DIR, f"shard_t7_{FEATURE}_{i}.log")
    cmd = [PYTHON, "-u", "obesity_fixed_feature.py", "--fixed", FEATURE, "--shard", shard_str]
    f_out = open(log_file, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=DIR, stdout=f_out, stderr=subprocess.STDOUT)
    processes.append((i, p, f_out))
    print(f"  Worker {i} (shard {shard_str}) PID: {p.pid}")

print(f"\nAll {NUM_SHARDS} workers active. Monitoring execution...")

start_time = time.time()
while True:
    all_done = True
    for i, p, f in processes:
        if p.poll() is None:
            all_done = False
            break
    
    # Check total queries completed across shards
    csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"table7_fixed_{FEATURE}_s*of6.csv")))
    n_done = 0
    valid_mean = None
    if csv_files:
        try:
            dfs = [pd.read_csv(f) for f in csv_files if os.path.getsize(f) > 0]
            if dfs:
                comb = pd.concat(dfs, ignore_index=True)
                n_done = len(comb)
                valid_mean = comb["valid_frac"].mean()
        except Exception:
            pass

    el = time.time() - start_time
    status_str = f"Elapsed: {el/60:.1f}m | Completed queries: {n_done}/368"
    if valid_mean is not None:
        status_str += f" | Mean Validity: {valid_mean:.4f}"
    print(status_str)
    
    if all_done:
        print("\nAll workers completed!")
        break
    
    time.sleep(20)

for _, _, f in processes:
    f.close()

print(f"\nFinished control run for '{FEATURE}'. Summarizing results...")
subprocess.run([PYTHON, "obesity_fixed_feature.py", "--fixed", FEATURE, "--summarize"], cwd=DIR)
