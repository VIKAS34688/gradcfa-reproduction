# Contribution Log

**Project:** GradCFA reproduction and extension on three datasets (DA351 Explainable AI)
**Last updated:** 2026-11-05

Rules for this file: only completed work is listed, with the member who really did it and where the evidence is (a file or a commit). If two members shared a package, write both names and what each did. The GitHub commit history shows who committed what.

## Team (5 members)

| Member | Main area | XAI (Phase 2) area |
|---|---|---|
| Vikas P | Credit Approval Phase 1, repository, integration, documents | Credit Approval Phase 2: exact Shapley values, Path-IG, tests T1 to T3 |
| EVS Harsha | Obesity Levels reproduction: model, data preparation and full-scale runs | Obesity Levels XAI: fixed-feature test (Weight and control feature) |
| Vishwa | Obesity Levels result analysis, per-class evaluation and hyperparameter verification | Obesity Levels XAI: result verification and interpretation |
| Kishan Kumar | Fetal Health reproduction: data preparation, model and full-scale runs | Fetal Health XAI: Shapley values and Path-IG |
| Dhanush | Fetal Health reproduction: query evaluation and generator verification | Fetal Health XAI: tests T1 and T2, agreement analysis |

## Work packages

| ID | Work package | Area | Done by | Evidence |
|---|---|---|---|---|
| 1-C | Credit Approval reproduction: full-scale runs on all 138 test rows, ablation grid, paper's 60/20/20 split, plausibility sensitivity | Reproduction | Vikas P | `scripts/`, `results/credit_80_20/`, `results/credit_60_20_20/`, `docs/PHASE1_RESULTS_CREDIT.md` |
| 2-C | Credit Approval XAI: exact Shapley values, Path-IG, tests T1 to T3, robustness on three seeds | XAI | Vikas P | `scripts/phase2_attribution.py`, `scripts/phase2_regen.py`, `docs/PHASE2_RESULTS_CREDIT.md` |
| 1-O | Obesity Levels reproduction: model, 368 queries, per-class results, hyperparameters | Reproduction | EVS Harsha, Vishwa | `results_obesity/`, `docs/PHASE1_RESULTS_OBESITY.md` |
| 2-O | Obesity Levels XAI: fixed-feature test (Weight and control feature) | XAI | EVS Harsha, Vishwa | `results_obesity/`, `docs/PHASE2_RESULTS_OBESITY.md` |
| 1-F | Fetal Health reproduction: data preparation, model, 71 queries, generator fix | Reproduction | Kishan Kumar, Dhanush | `results_fetal/`, `docs/FETAL_PHASE1_RESULTS.txt` |
| 2-F | Fetal Health XAI: Shapley values, Path-IG, tests T1 and T2, agreement | XAI | Kishan Kumar, Dhanush | `results_fetal/`, `docs/FETAL_PHASE2_RESULTS.txt` |
| 3 | XAI verification: independent cross-check of the Phase 2 numbers against the result files and of Shapley against the `shap` library | XAI | Vikas P | (add file or commit) |
| 4 | Report verification: check every table in the README and `docs/` against the result files, and the paper's values against the PDF | Reproduction | Vikas P, EVS Harsha | (add file or commit) |
| 5 | Fresh-clone test on a second laptop: run the quick tests from the README | Reproduction | EVS Harsha | (add file or commit) |
| 6 | Presentation: slides and question practice | Presentation | All five members, if participated | (add file) |
| 7 | Repository: README, `AI_USAGE.md`, this file, layout, final commit | Documentation | Vikas P | repository root |

Packages 3, 4 and 5 are each about one to two hours of real work. They are listed so that members who have no dataset package can take them; they are logged only once the work is done.

## Work log

| Date | Work | Done by |
|---|---|---|
| Before 2026-10-06 | Paper shortlist, venue and dataset checks, selection of GradCFA | Vikas P |
| 2026-10-06 to 2026-10-10 | Cloned the official repository, created the environment, fixed import path, class name, model path and the missing `return` in `DiPACE.py`, added the seed, `requirements.txt`, GitHub repository, README and logs | Vikas P, Dhanush |
| 2026-10-10 to 2026-10-12 | Ran the notebook on one query and the first full runs on the Credit test set | Vikas P, Dhanush |
| 2026-10-18 | Found the input-space mismatch of the notebook, retrained the model in the correct space | Vikas P, Vishwa |
| 2026-10-19 to 2026-10-21 | All-rows runs (Config A 3 seeds, full method 1 seed) and the ablation grid on Credit | Vikas P, EVS Harsha |
| 2026-10-22 to 2026-10-26 | Phase 2 on Credit: Shapley values, Path-IG, tests T1 to T3, robustness runs | Vikas P, Vishwa |
| 2026-10-27 | Fetal Health reproduction and XAI work (packages 1-F and 2-F): data preparation, DataInterface and `GradCFA_MC.py` fixes, model, 71-query evaluation, Phase 2 tests (details below) | Kishan Kumar |
| 2026-10-27 to 2026-10-29 | Credit with the paper's 60/20/20 split, validation-only tuning, plausibility sensitivity | Vikas P |
| 2026-10-30 | Reorganised the repository into `scripts/`, `docs/`, `models/` and `results/` | Vikas P |
| 2026-10-31 to 2026-11-02 | Obesity Levels reproduction and XAI work (packages 1-O and 2-O) | EVS Harsha, Vishwa |

## Obesity Levels: details 

| Package | Contribution | Evidence |
|---|---|---|
| 1-O Obesity Phase 1 | Analysed the per-class evaluation, checked the reported hyperparameters, and compared the corrected random-initialization run with the earlier run and the paper. The corrected run covers 368 queries; the report shows diversity increasing from 0.307 to 0.846 and records the per-class metrics and limitations. | `PHASE1_RESULTS_OBESITY.md`; `obesity_eval_corrected.py`; `obesity_eval.py` |
| 2-O Obesity Phase 2 | Contributed to verifying and interpreting the fixed-feature results for top-ranked `Weight` and control `NCP`. The report records validity of 0.5668 for `Weight` locked (0.5734 on clean shards) versus 0.9837 for `NCP` locked, against a 0.9815 baseline. | `PHASE2_RESULTS_OBESITY.md`; `obesity_fixed_feature.py`; `run_shards_table7.ps1`; `run_shards_table7_NCP.ps1`; `run_shards_control.py` |

### Supplemental work log 

| Date | Work | Evidence / attribution note |
|---|---|---|
| 2026-10-18 | Participated with Vikas P in investigating the notebook/DataInterface input-space mismatch and the corrected model-training activity. | Joint credit in the supplied contribution log; implementation context in `check_and_train_di.py` and `README.md`. The source does not specify the individual task split. |
| 2026-10-22 to 2026-10-26 | Contributed with Vikas P to the Credit Phase 2 work and robustness runs. | Joint credit in the supplied contribution log; `phase2_attribution.py`, `phase2_regen.py`, and `PHASE2_RESULTS.md`. The source does not specify the individual task split. |
| 2026-10-31 | Worked with EVS Harsha on the Obesity reproduction and fixed-feature analysis, including per-class result review and hyperparameter verification. | Joint credit in the supplied contribution log; details and result tables in `PHASE1_RESULTS_OBESITY.md` and `PHASE2_RESULTS_OBESITY.md`. |
| 2026-11-02 | Cross-checked the Obesity paper tables, updated the hyperparameter metadata, launched the six-shard NCP control run, and assembled submission documentation. | Recorded in `AI_USAGE.md`; relevant artifacts include `PHASE1_RESULTS_OBESITY.md`, `PHASE2_RESULTS_OBESITY.md`, and `run_shards_table7_NCP.ps1`. |

## Fetal Health: details

| Date | Task / milestone | Details |
|---|---|---|
| 2026-10-27 | Repository and environment setup (Step 0) | Created `fetal` branch, set up virtual environment `gradcfa_env`, installed requirements, verified required repo files. |
| 2026-10-27 | Dataset preparation (Steps 1 and 2) | Downloaded UCI Cardiotocography (CTG.xls, DOI 10.24432/C51S4N). Built `prepare_fetal.py` to extract 2126 rows and 32 features (22 continuous, 10 categorical morphological flags; excluded CLASS to avoid target leakage, dropped DR per UCI description). Created stratified 60/20/20 split (`datasets/fetal_split.csv`). |
| 2026-10-27 | Bug fix: DataInterface categorical indexing | Resolved a bug in `src/data/Data_Interface.py` where `startswith(col_parent)` wrongly grouped `AD` with `A` and `DE` with `D`. Added the delimiter check `col.startswith(col_parent + "_") or col == col_parent`. |
| 2026-10-27 | Bug fix: GradCFA_MC importance ranking | Fixed missing `self.importance_ranking = importance_ranking` assignment in `src/cf/GradCFA_MC.py`. |
| 2026-10-27 | Model training in DataInterface space (Step 3) | Implemented `check_and_train_fetal.py`. Trained a 64-32-3 MLP on `d.norm_encoded_features` (42 columns): train 0.9153 (paper 0.92), val 0.9153 (paper 0.93), test 0.9131 (paper 0.91). Saved `pytorch_model_fetal.pth`. |
| 2026-10-27 | Full evaluation pipeline (Steps 4 and 5) | Built `run_full_eval_fetal.py`. Timing benchmark (5 queries, about 17.5 s per query), then the full evaluation on all 71 test queries under Config A (seed 42), saving metrics, feature attributions and `.npz` counterfactuals to `results_fetal/`. |
| 2026-10-27 | Phase 1 summaries and analysis (Step 6) | Overall comparison table, breakdown by original class (class 1 vs class 2), feature ranking, and reasons for divergence from the paper. |
| 2026-10-27 | Phase 2 attribution trustworthiness (Step 7) | Implemented `phase2_attribution_fetal.py` and `phase2_regen_fetal.py`. Evaluated agreement with Shapley values (correlation -0.21), faithfulness test T1 (GradCFA below random ranking at k=1), test T2, and the T3 fixed-feature regeneration test for D and A. |