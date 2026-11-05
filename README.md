# GradCFA: Reproduction and Extension on Three Datasets (DA351 Explainable AI)

We reproduce, and then extend, the following paper with the authors' official code:

> J. Sanderson, H. Mao, W. L. Woo, "GradCFA: A Hybrid Gradient-Based Counterfactual and Feature Attribution Explanation Algorithm for Local Interpretation of Neural Networks", *IEEE Transactions on Artificial Intelligence*, vol. 6, no. 10, pp. 2575-2587, 2025. DOI: [10.1109/TAI.2025.3552057](https://doi.org/10.1109/TAI.2025.3552057)
> Original code: https://github.com/jacob-ws/GradCFs (MIT license, kept in `LICENSE`). The authors' README is in `environment/README.md`.

GradCFA generates counterfactual explanations ("what would have to change to flip the prediction") and, in the same optimisation, a ranking of feature importance.

**Phase 1** reproduces the paper's main experiment at full scale on three datasets: Credit Approval (binary), Obesity Levels (7 classes) and Fetal Health (3 classes).
**Phase 2** asks a question the paper does not test: *is GradCFA's feature ranking trustworthy?* We compare it with exact Shapley values (the quantity SHAP estimates) using faithfulness tests, and propose a fix (Path-IG). The full study is on Credit Approval.

Team of five: see `CONTRIBUTIONS.md`. AI tool use: see `AI_USAGE.md`.

## Results at a glance

**Phase 1: our reproduction against the paper** (average of the five metrics: 1 - proximity, 1 - sparsity, 1 - plausibility, diversity, confidence)

| Dataset | Test queries | Model test accuracy (ours / paper) | Average (ours / paper) | Confidence (ours / paper) | Setting |
|---|---|---|---|---|---|
| Credit Approval | 138 (all test rows) | 0.833 / 0.82 | 0.753 / 0.81 | 0.828 / 0.78 | full method, 1 seed; perturbation off, 3 seeds: 0.749 |
| Obesity Levels | 368 (all test rows not predicted as class 1) | 0.903 / 0.85 | 0.639 / 0.78 | 0.664 / 0.76 | perturbation off, 1 seed |
| Fetal Health | 71 (all test rows not predicted as class 0) | 0.913 / 0.91 | 0.615 / 0.73 | 0.958 / 0.79 | full method, 1 seed; perturbation off: 0.612 |

None of the three datasets matches the paper's average. What we did reproduce: on Credit, the perturbation step raises the share of valid counterfactuals from 87% to 95% and confidence matches the paper; on Obesity and Fetal Health the pipeline runs end to end at full scale and the dominant feature on Obesity (Weight, then MTRANS) agrees with the paper. Per-dataset detail, tables and the reasons for the gaps are in the reports under `results/` (see the layout below).

**Phase 2 on Credit Approval: before and after** (GradCFA's own ranking against our Path-IG alternative; 135 queries; 95% bootstrap intervals; higher is better except for T3)

| Test | GradCFA (before) | Path-IG (after) | Shapley (reference) | Random | Path-IG minus GradCFA |
|---|---|---|---|---|---|
| T1: prediction drop when the top features are ablated (average over k) | 0.298 | **0.372** | 0.363 | 0.194 | +0.074 [+0.062, +0.087] |
| T2: counterfactual damage when the top features are reset (average over k) | 0.608 | **0.714** | 0.692 | 0.381 | +0.106 [+0.094, +0.119] |
| T3: share of valid counterfactuals left after fixing the top feature and regenerating (lower = more faithful) | 0.814 | **0.281** | 0.290 | 0.777 | -0.533 [-0.609, -0.449] |
| Top-1 feature agrees with Shapley | 24% | **92%** | n/a | 7% | |

GradCFA's top feature on Credit (usually Income) is no better than a random feature in the real regeneration test (T3). Path-IG matches Shapley and needs no extra optimisation. We make no claim that Path-IG beats Shapley (`results/Phase_2/Credit Approval/PHASE2_RESULTS.md`).

**Phase 2 on the other two datasets** (as reported in `results/Phase_2/`):

- **Fetal Health** (71 queries): GradCFA's ranking is negatively correlated with Shapley (Spearman -0.22, top-1 match 4%); Path-IG agrees with Shapley (Spearman 0.78, top-1 match 87%). Shapley values there are estimated with 300 random permutations per query, not computed exactly.
- **Obesity**: only the paper's fixed-feature test was run. Locking `Weight` lowers counterfactual validity from 0.98 to 0.57 (paper: 0.18); locking the lowest-ranked feature `NCP` leaves it at 0.98. Shapley, Path-IG and tests T1 to T3 were not run on Obesity.

## Repository layout

```
.
├── README.md                    this file
├── AI_USAGE.md                  tools, prompts, and how the output was checked and changed
├── CONTRIBUTIONS.md             who did what
├── LICENSE                      authors' MIT license
├── requirements.txt             pinned packages
├── pytorch_model_di.pth         the trained Credit model (input space of the counterfactual code)
├── notebooks/
│   ├── binary_notebook.ipynb    authors' Credit notebook, with our fixes
│   └── mc_notebook.ipynb        authors' multi-class notebook (Obesity and Fetal Health work started from it)
├── src/  eval/                  authors' code: counterfactual methods, data and model interfaces, metrics
│                                (src/cf/patch_gradcfa_mc.py is our patch for the multi-class generator)
├── environment/README.md        authors' original README
├── scripts/                     shared scripts (see the table below)
│   ├── training/                check_and_train_di.py, obesity_train.py
│   ├── evaluation/              run_full_eval.py, obesity_eval*.py, run_shards*.ps1, run_shards_control.py
│   └── analysis/                recompute_plausibility.py, diagnose_model.py
├── experiments/                 per-dataset experiment code
│   ├── phase1/                  credit_approval/ (credit_6020.py), fetal/ (data preparation, training, evaluation),
│   │                            obesity/ (obesity_common.py, obesity_fixed_feature.py)
│   └── phase2/
│       ├── feature_attribution/        Shapley, Path-IG, tests T1 and T2: Credit approval/, fetal/, obesity/
│       └── counterfactual_regeneration/ test T3 (regenerate with top features fixed): Credit approval/, fetal/, obesity/
├── Output_files/                raw outputs: summaries, settings (JSON), figures
│   ├── Phase_1/                 Credit Approval/ (60/20/20 split runs), fetal/, obesity/
│   └── Phase_2/                 Credit Approval/ (results_di/ incl. saved_counterfactuals.zip), fetal/, obesity/
└── results/                     written reports
    ├── Phase_1/                 Credit Approval/PHASE1_RESULTS.md, fetal/, obesity/PHASE1_RESULTS_OBESITY.md
    └── Phase_2/                 Credit Approval/PHASE2_RESULTS.md, fetal/, obesity/
```

| Script | What it does |
|---|---|
| `scripts/training/check_and_train_di.py` | measures the input-space mismatch of the original notebook and trains the Credit model in the correct space |
| `scripts/evaluation/run_full_eval.py` | runs GradCFA on every test query for several seeds; writes the five metrics and the feature scores to CSV; `--summarize` prints mean and standard deviation |
| `experiments/phase1/credit_approval/credit_6020.py` | the paper's 60/20/20 split, validation-only tuning, paired comparison of settings |
| `scripts/analysis/recompute_plausibility.py` | recomputes plausibility with several definitions from the saved counterfactuals |
| `experiments/phase2/feature_attribution/Credit approval/phase2_attribution.py` | Phase 2: exact Shapley values, Path-IG, agreement, tests T1 and T2 |
| `experiments/phase2/counterfactual_regeneration/Credit approval/phase2_regen.py` | Phase 2: test T3, regenerates counterfactuals with the top-ranked features fixed |

## Setup and how to reproduce Credit Approval (Windows PowerShell)

```powershell
git clone https://github.com/VIKAS34688/gradcfa-reproduction.git
cd gradcfa-reproduction
python -m venv gradcfa_env
.\gradcfa_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run every command from the repository root. Seeds are fixed, and finished queries are skipped when a command is run again, so an interrupted run can be resumed.

| Step | Command | Approximate time on a laptop |
|---|---|---|
| Train the model in the correct input space | `python scripts/training/check_and_train_di.py` | seconds |
| Quick test (3 queries) | `python scripts/evaluation/run_full_eval.py --config A --seeds 42 --limit 3` | 1 minute |
| Perturbation off, all 138 test rows, 3 seeds | `python scripts/evaluation/run_full_eval.py --config A --which all --seeds 42 43 44` | 5 to 6 hours |
| Full method, all 138 rows | `python scripts/evaluation/run_full_eval.py --config C --which all --seeds 42` | about 16 hours |
| Summaries | `python scripts/evaluation/run_full_eval.py --config C --which all --summarize` | seconds |
| Plausibility sensitivity | `python scripts/analysis/recompute_plausibility.py` | 1 minute |
| Phase 2 tests T1, T2 | `python "experiments/phase2/feature_attribution/Credit approval/phase2_attribution.py"` | a few minutes |
| Phase 2 test T3 | `python "experiments/phase2/counterfactual_regeneration/Credit approval/phase2_regen.py"`, then `--summarize` | 4 to 5 hours |
| Paper's 60/20/20 split | `python experiments/phase1/credit_approval/credit_6020.py train`, then `run_full_eval.py --split test --config A --which all` | about 1 hour |
| Validation-only tuning | `python experiments/phase1/credit_approval/credit_6020.py tune`, then compare | about 3.5 hours |

The raw counterfactual files needed by the Phase 2 scripts are many small files; they are packed in `Output_files/Phase_2/Credit Approval/results_di/saved_counterfactuals.zip`, or can be recreated with the "all 138 rows" commands above.
Obesity and Fetal Health: the exact settings are saved next to their results (`Output_files/Phase_1/obesity/hyperparameters_obesity.json`, `Output_files/Phase_1/fetal/hyperparams_A.json`, `Output_files/Phase_1/fetal/hyperparams_C.json`), and the run details are in their reports in `results/`. Their scripts are in `scripts/` and `experiments/` (Obesity: `scripts/training/obesity_train.py`, `scripts/evaluation/obesity_eval_corrected.py`, `scripts/evaluation/run_shards_*.ps1`; Fetal: `experiments/phase1/fetal/`).

## What we did differently from the paper, and what we did not reproduce

- **Baselines and tables.** Wachter, DiCE and CARE baselines were not run, so the paper's Table II comparison with other methods is not reproduced. The paper's full ablation table was run only on Credit.
- **Model input space.** The authors' notebook trains the model on columns in one order with every column z-scored, but the counterfactual code uses another order and scales only continuous features (25 of 61 Credit columns misaligned). We train the model in the counterfactual code's own space.
- **Perturbation.** With `loss_converge_maxiter=2` (the value the code comments attribute to the paper) the perturbation step never changed the output in our Credit runs. Our "full method" uses `loss_converge_maxiter=1`. On Obesity, perturbation was not run.
- **Plausibility** uses Eq. 5 with k = 3, which is our assumption (the paper does not state k). Depending on the definition, the Credit average ranges from 0.72 to 0.85 (`results/Phase_1/Credit Approval/PHASE1_RESULTS.md`, Table 5), so the plausibility definition alone could account for the Credit gap.
- **Splits and tuning.** Credit main runs use an 80/20 split; the paper's 60/20/20 split gives the same average (0.753) and is in `Output_files/Phase_1/Credit Approval/results_6020`. The paper tuned hyperparameters on the test set; we use the paper's values, and our own tuning uses validation rows only.
- **Seeds.** Credit: 3 seeds with perturbation off, 1 seed for the full method (16 hours per seed). Obesity and Fetal Health: 1 seed.
- **Phase 2 limits.** The three tests are proxies for faithfulness; T3 uses one seed, perturbation off, and the 69 rejected Credit queries; Path-IG is computed in the scripts and is not yet a method of the original classes. The full Phase 2 study is on Credit only; Fetal Health has T1, T2 and agreement; Obesity has only the fixed-feature test.

## Issues found in the original repository

- No `requirements.txt` (we added one).
- The notebooks need `src` added to `sys.path`.
- The notebook imports `GradCFA`, but the class in `src/cf/GradCFA.py` is named `DiPACE_FA`.
- The model path points to the author's local folder (`../DiPACE/pytorch_model.pth`).
- `DiPACE.py`, `get_feature_weights_list` is missing a `return` statement (we fixed it).
- The `evaluate()` call in `binary_notebook` is outdated: it omits `num_classes`, and `evaluate()` always uses the multi-class confidence function.
- No random seed is set anywhere (we added one).
- The Fetal Health dataset is not included.
- The notebook splits 80/20, while the paper states 60/20/20.
- The repository's plausibility score is a raw nearest-neighbour distance, not scaled to 0-1 as in the paper's Eq. 5.
- Column order and scaling differ between the notebook's model training and `DataInterface`; we retrain in `DataInterface` space.
- With `loss_converge_maxiter=2` the perturb-and-retry step appears not to run (our reading of the code, supported by identical outputs; not confirmed with the authors).
- `DiPACE_FA` only prints and plots the feature importance; we store it in `self.importance_ranking`.
- Multi-class generator (`GradCFA_MC.py`), as reported by the team: the default `init_near_query_instance=True` starts the five counterfactuals as near-copies of the query (Obesity diversity 0.31 until the start was changed to random), and a line near line 318 (`cf_instances = best_cf_instances`) overwrote the optimised counterfactuals in the Fetal Health runs (0% valid counterfactuals until patched). Which code version each run used is stated in its report in `results/`.

## Reproducibility notes

- Seed 42 for model training; Credit counterfactual generation uses one seed per query (`seed * 1000 + row id`), so resumed runs give identical results.
- Packages in `requirements.txt`.
- The numbers in this README come from the files in `Output_files/` and the reports in `results/`. Every table states its number of queries and seeds.
- Phase 2 uses a numpy copy of the model; the Credit scripts check that it reproduces the stored validity of every query before running.

## Team, AI use and credits

Who did what: `CONTRIBUTIONS.md`. AI tool use: `AI_USAGE.md`.

All credit for GradCFA and DiPACE goes to the original authors. Please cite:

```bibtex
@article{sanderson2025gradcfa,
  title   = {GradCFA: A Hybrid Gradient-Based Counterfactual and Feature Attribution Explanation Algorithm for Local Interpretation of Neural Networks},
  author  = {Sanderson, Jacob and Mao, Hua and Woo, Wai Lok},
  journal = {IEEE Transactions on Artificial Intelligence},
  year    = {2025},
  volume  = {6},
  number  = {10},
  pages   = {2575--2587},
  doi     = {10.1109/TAI.2025.3552057}
}

@inproceedings{sanderson2025dipace,
  author    = {Sanderson, Jacob and Mao, Hua and Woo, Wai Lok},
  title     = {DiPACE: Diverse, Plausible and Actionable Counterfactual Explanations},
  booktitle = {Proceedings of the 17th International Conference on Agents and Artificial Intelligence},
  year      = {2025},
  doi       = {10.5220/0013219100003890}
}
```
