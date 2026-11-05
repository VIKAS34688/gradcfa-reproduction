# Phase 1 results: GradCFA on Credit Approval (corrected pipeline)

## Setup
- Dataset: Credit Approval, 690 rows, full dataset. Test split: 20% = 138 rows (`random_state=42`).
- Black-box model: 64-32 MLP trained in DataInterface space, test accuracy 0.833 (paper: 0.82).
- Queries: all 138 test rows. The target class is the opposite of the model's prediction (69 rows rejected, 69 rows approved).
- Hyperparameters: the paper's Credit values (lambda 0.5, gamma_pen 0.1, gamma_pert 0.5, tau_prox = tau_spars = 0.2, tau_plaus = 0.4, tau_div = 0.9, tau_loss = 0.5).
- Metrics: proximity, sparsity, diversity and confidence from the repository's `Evaluate` class. Plausibility uses the paper's Eq. 5 with k = 3 (k is not stated in the paper). The average is the mean of (1 - proximity), (1 - sparsity), (1 - plausibility), diversity and confidence.
- Seeds: 42, 43, 44 for configurations marked 3 seeds, fewer where stated. The model is fixed; only counterfactual generation changes with the seed.

## Table 1: paper versus this reproduction (all 138 test rows)

| Metric | Paper (GradCFA) | Full method (perturbation and penalty on), 1 seed | Perturbation off, 3 seeds |
|---|---|---|---|
| Proximity | 0.15 | 0.218 | 0.227 |
| Sparsity | 0.18 | 0.227 | 0.236 |
| Plausibility (Eq. 5, k = 3) | 0.37 | 0.509 | 0.517 |
| Diversity | 0.96 | 0.891 | 0.911 |
| Confidence | 0.78 | 0.828 | 0.813 |
| Fraction of valid counterfactuals | not reported | 0.930 | 0.898 |
| **Average of the five** | **0.81** | **0.753** | **0.749** |

## Table 2: effect of perturbation and penalty (69 rows the model rejects)

| Perturbation | Penalty | Config | Seeds | Prox | Spars | Plaus | Div | Conf | Valid | Average |
|---|---|---|---|---|---|---|---|---|---|---|
| off | on | A = C_nopert | 3 | 0.210 | 0.228 | 0.523 | 0.923 | 0.738 | 0.866 | 0.740 |
| off | off | C_nopert_nopen | 2 | 0.210 | 0.228 | 0.523 | 0.920 | 0.753 | 0.883 | 0.742 |
| on | off | C_nopen | 2 | 0.198 | 0.217 | 0.522 | 0.900 | 0.788 | 0.951 | 0.750 |
| on | on | C | 1 | 0.201 | 0.220 | 0.526 | 0.900 | 0.781 | 0.954 | 0.747 |

- Turning perturbation on raises the fraction of valid counterfactuals by about 7 to 9 points and confidence by about 3 to 5 points. It lowers diversity by about 2 points and raises the average by about 0.008.
- Turning the penalty on changes the average by about -0.003, which is within noise.
- Note: with `loss_converge_maxiter=2` (the value attributed to the paper in the code), perturbation never changed the output in our runs (configurations A and C_nopert are identical).

## Table 3: where the gap to the paper comes from (all 138 rows, full method)

| Metric | Contribution to the average gap (paper minus ours) |
|---|---|
| Plausibility | 0.028 |
| Diversity | 0.014 |
| Proximity | 0.014 |
| Sparsity | 0.010 |
| Confidence | -0.010 (ours is higher) |
| **Total** | **0.055** (0.808 against 0.753) |

## Summary of findings
1. Reproduced: perturbation increases validity and confidence; confidence matches the paper (0.83 against 0.78).
2. Not reproduced: the paper's average (0.81). Every variant we ran stays between 0.740 and 0.753, so perturbation, penalty and input-space choices do not explain the gap.
3. Not reproduced: a clear benefit from the penalty.
4. About half of the gap is plausibility, the metric whose exact definition (Eq. 5, value of k) the paper leaves ambiguous.
5. Slowest queries: three rows (132, 65, 6) took 15 to 39 minutes each even without perturbation; the full method took 15.8 hours for 138 queries.

## Not reproduced
Obesity and Fetal Health datasets; Wachter, DiCE and CARE baselines; the SHAP comparison and the fixed-feature test (paper's Table VI); the paper's full ablation table.
