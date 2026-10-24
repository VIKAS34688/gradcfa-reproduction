# GradCFA Reproduction and Extension (DA351 Explainable AI)

Course project for DA351 - Explainable AI. We reproduce, and then extend, the following paper:

> J. Sanderson, H. Mao, W. L. Woo, "GradCFA: A Hybrid Gradient-Based Counterfactual and Feature Attribution Explanation Algorithm for Local Interpretation of Neural Networks", *IEEE Transactions on Artificial Intelligence*, vol. 6, no. 10, pp. 2575-2587, 2025. DOI: 10.1109/TAI.2025.3552057

The code in this repository is the authors' official implementation (https://github.com/jacob-ws/GradCFs, MIT license). We keep their code and license, and add our own fixes, documentation and experiments. The authors' original README is kept as `README_ORIGINAL.md`.

## Project status

| Item | Status |
|---|---|
| Environment set up, original code running | Done |
| Fixed random seed (SEED = 42), checked on two runs | Done |
| GradCFA on all test queries (72 queries), Credit Approval: Config A (3 seeds), Config B (1 seed), ablation variants (3 seeds), notebook pipeline | Done, results in `results/` |
| Input-space mismatch in the notebook pipeline found, model retrained in DataInterface space | Done (`check_and_train_di.py`, test accuracy 0.833) |
| Config A rerun in the corrected pipeline (3 seeds) | Done, results in `results_di/` |
| Ablation reruns in the corrected pipeline, SHAP baseline | Not started |
| Obesity (multi-class) | Not started (if time permits) |
| Phase 2 extension | Not started |

Results are reported with the number of queries and seeds stated. They do not match the paper's Table II; see the analysis in the Phase 1 summary (to be added).

## Repository contents

- `src/`, `eval/`, `datasets/`, `binary_notebook.ipynb`, `mc_notebook.ipynb`: original code by the authors, with our fixes (see "Issues found" below)
- `requirements.txt`: package versions used (added by us)
- `run_full_eval.py`: runs GradCFA on every test query that the model predicts as class 0, for several seeds, and writes metrics to CSV
- `check_and_train_di.py`: measures the input-space mismatch and retrains the model in DataInterface space
- `results/`: results from the original notebook pipeline (model trained on a different column order and scaling than the counterfactual code uses)
- `results_di/`: results from the corrected pipeline (`pytorch_model_di.pth`)
- `AI_USAGE.md`: disclosure of AI tool use
- `CONTRIBUTIONS.md`: contribution log
- Branches: `main` (setup), `phase1-replication`, `phase2-extension`

## Setup (Windows, PowerShell)

Tested with Python 3.14.7.

```powershell
python -m venv gradcfa_env
.\gradcfa_env\Scripts\Activate.ps1
pip install -r requirements.txt
pip install jupyter ipykernel
```

Open `binary_notebook.ipynb` in VS Code from the repository root, select the `gradcfa_env` kernel, and run the cells in order. The training cells create `pytorch_model.pth` in the repository root; this file is not committed and is recreated each time.

## Datasets

| Dataset | Rows | File | Source |
|---|---|---|---|
| Credit Approval (binary) | 690 | `datasets/credit.csv` | UCI, DOI 10.24432/C5FS30 |
| Obesity Levels (multi-class) | 2,111 | `datasets/obesity.csv` | UCI, DOI 10.24432/C5H31Z |
| Fetal Health / Cardiotocography (multi-class) | 2,126 | not included in the original repo | UCI, DOI 10.24432/C51S4N |

## Reproducibility notes

- Random seed: `SEED = 42` for Python, NumPy and PyTorch, set at the top of the notebook.
- The notebook splits the data 80/20 with `random_state=42`. The paper states a 60/20/20 split.
- Results will be reported with the number of test instances and the seeds used, as mean and standard deviation over 3 seeds.

## Issues found in the original repository

1. No `requirements.txt` (we added one).
2. The notebooks need `src` added to `sys.path`.
3. The notebook imports `GradCFA`, but the class in `cf/GradCFA.py` is named `DiPACE_FA`.
4. The model path points to the author's local folder (`../DiPACE/pytorch_model.pth`).
5. `DiPACE.py`, `get_feature_weights_list` is missing a `return` statement (we fixed it).
6. The `evaluate()` call in `binary_notebook` is outdated: it omits `num_classes`, and `evaluate()` always uses the multi-class confidence function.
7. No random seed is set anywhere (we added one).
8. The Fetal Health dataset is not included.
9. The notebook splits 80/20, while the paper states 60/20/20.
10. The repository's plausibility score is a raw nearest-neighbour distance, not scaled to 0-1 as in the paper's Eq. 5, so it cannot be compared directly with the paper's table.
11. The notebook trains the model on columns ordered CreditScore before Employed and z-scores every column, but DataInterface orders Employed before CreditScore and normalises only the continuous features (25 of 61 column positions differ). The old model changes its prediction on 7.5% of rows between the two spaces. We retrained the model in DataInterface's space (test accuracy 0.833) and report both pipelines.
12. With `loss_converge_maxiter=2` (the value the code comments attribute to the paper), results were identical to the no-perturbation variant in all runs, so the perturb-and-retry step appears not to run in the repository's current code (hypothesis from reading the code, not yet confirmed).

## Credits and citation

All credit for GradCFA and DiPACE goes to the original authors. Please cite their work:

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
## Reproducing Phase 2
Unzip `results_di/saved_counterfactuals.zip` into `results_di/`, then run `python phase2_attribution.py`.