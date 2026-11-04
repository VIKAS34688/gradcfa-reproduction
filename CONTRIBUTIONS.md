# Contribution Log

**Project:** GradCFA reproduction and extension on three datasets (DA351 Explainable AI)  
**Last updated:** 2026-11-03

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
| 3 | XAI verification: independent cross-check of the Phase 2 numbers against the result files and of Shapley against the `shap` library | XAI | Vikas P | |
| 4 | Report verification: check every table in the README and `docs/` against the result files, and the paper's values against the PDF | Reproduction | Vikas P , EVS Harsha | |
| 5 | Fresh-clone test on a second laptop: run the quick tests from the README | Reproduction | EVS Harsha | |
| 6 | Presentation: slides and question practice | Presentation | All five members, if participated | |
| 7 | Repository: README, `AI_USAGE.md`, this file, layout, final commit | Documentation | Vikas P | repository root |

Packages 3, 4 and 5 are each about one to two hours of real work. They are listed so that members who have no dataset package can take them; they are logged only once the work is done.

## Work log

| Date | Work | Done by |
|---|---|---|
| Before 2026-10-06 | Paper shortlist, venue and dataset checks, selection of GradCFA | Vikas P |
| 2026-10-06 to 2026-10-10 | Cloned the official repository, created the environment, fixed import path, class name, model path and the missing `return` in `DiPACE.py`, added the seed, `requirements.txt`, GitHub repository, README and logs | Vikas P , Dhanush |
| 2026-10-10 to 2026-10-12 | Ran the notebook on one query and the first full runs on the Credit test set | Vikas P , Dhanush |
| 2026-10-18 | Found the input-space mismatch of the notebook, retrained the model in the correct space | Vikas P , Vishwa| 
| 2026-10-19 to 2026-10-21 | All-rows runs (Config A 3 seeds, full method 1 seed) and the ablation grid on Credit | Vikas P , EVS Harsha |
| 2026-10-22 to 2026-10-26 | Phase 2 on Credit: Shapley values, Path-IG, tests T1 to T3, robustness runs | Vikas P , Vishwa |
| 2026-10-27 to 2026-10-29 | Credit with the paper's 60/20/20 split, validation-only tuning, plausibility sensitivity | Vikas P |
| 2026-10-30 | Reorganised the repository into `scripts/`, `docs/`, `models/` and `results/` | Vikas P |
| 2026-10-31 | Obesity Levels reproduction and XAI work (packages 1-O and 2-O) | EVS Harsha, Vishwa |
| 2026-11-01 | Fetal Health reproduction and XAI work (packages 1-F and 2-F) | Kishan Kumar |