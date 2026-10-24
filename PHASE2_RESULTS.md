# Phase 2: Is GradCFA's feature attribution trustworthy?

Code: `phase2_attribution.py`. Results: `results_di/phase2_C_all/` (full method) and `results_di/phase2_A_all_s42`, `_s43`, `_s44` (perturbation off, three seeds).

## The gap in the paper
GradCFA returns counterfactuals together with a ranking of the features, built from the gradient size of the model output accumulated while the counterfactuals are optimised. As far as we can tell from the paper, this ranking is supported by an aggregate comparison with the top feature of SHAP and by a fixed-feature test (Table VI). It is not tested, query by query, whether the ranking identifies the features that matter for each counterfactual, and it is not compared with random or simple baselines. A gradient size also ignores how far each feature actually moved between the query and the counterfactual.

## What we did
All of it is computed offline from the saved counterfactuals of Phase 1 (Credit Approval, 138 test rows, 135 with at least one valid counterfactual).

Rankings compared, over the 14 original features:
- **GradCFA**: the repository's attribution scores.
- **Shapley**: exact interventional Shapley values (the quantity SHAP estimates), 50-row training background. Checked against the `shap` library: maximum difference 7e-16 on three queries.
- **Path-IG (our proposed alternative)**: integrated gradients from the query to each counterfactual, summed per feature, absolute values averaged over the counterfactuals.
- **Largest edit**: reference ranking by how much the counterfactuals changed each feature.
- **Random**: average over 30 random rankings.

Tests (higher = more faithful ranking):
- **T1 (query level)**: replace the top-k features by background values, measure the drop in the predicted-class probability.
- **T2 (counterfactual level)**: reset the top-k features of a valid counterfactual to the query's values, measure the drop in target-class probability, and how often the counterfactual stops being valid (flip rate). T2 tests GradCFA's own claim.

Averages are over k = 1..13. Intervals are 95% bootstrap over queries. Differences are paired by query.

## Results: full method (perturbation and penalty on, seed 42, 135 queries)

| | GradCFA | Shapley | Path-IG | Largest edit | Random |
|---|---|---|---|---|---|
| T1, average drop | 0.298 | 0.363 | **0.372** | 0.333 | 0.194 |
| T2, average drop | 0.608 | 0.692 | **0.714** | 0.641 | 0.381 |
| T2, average flip rate | 0.796 | 0.899 | **0.941** | 0.843 | 0.506 |
| T2, flip rate with the single top feature | 0.217 | 0.653 | **0.680** | 0.273 | 0.074 |

Paired differences (95% interval):
- Path-IG minus GradCFA: T1 +0.074 [+0.062, +0.087]; T2 drop +0.106 [+0.094, +0.119]; flip rate +0.145 [+0.125, +0.165].
- Shapley minus GradCFA: T1 +0.065 [+0.057, +0.074]; T2 drop +0.084 [+0.072, +0.097].
- Largest edit minus GradCFA: T1 +0.035 [+0.021, +0.049]; T2 drop +0.033 [+0.019, +0.047].
- Path-IG minus Shapley: T1 +0.009 [-0.001, +0.021] (no difference); T2 drop +0.022 [+0.016, +0.028]; flip rate +0.042 [+0.025, +0.061].

Agreement with Shapley (chance: Spearman 0, top-1 0.071, top-3 overlap 0.214):

| | Spearman | Top-1 match | Top-3 overlap |
|---|---|---|---|
| GradCFA vs Shapley | 0.343 | 0.244 | 0.469 |
| Path-IG vs Shapley | 0.623 | 0.919 | 0.751 |

## Robustness: perturbation off, three seeds (the behaviour of the repository's current code)

| Quantity | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| GradCFA vs Shapley, top-1 match | 0.052 | 0.044 | 0.030 |
| T2 flip rate with the top feature: GradCFA | 0.046 | 0.040 | 0.031 |
| T2 flip rate with the top feature: Path-IG | 0.649 | 0.664 | 0.659 |
| T2 average drop: GradCFA | 0.558 | 0.553 | 0.556 |
| T2 average drop: Path-IG | 0.712 | 0.716 | 0.705 |
| Paired Path-IG minus GradCFA, T2 drop | +0.153 | +0.163 | +0.149 |
| Paired Path-IG minus Shapley, T2 drop | +0.025 | +0.024 | +0.022 |

The picture is the same in all three runs, and the gap to Path-IG is larger than with the full method.

## 1-page gap explanation (draft)
**Limitation.** GradCFA gives a ranking of feature importance together with its counterfactuals, but the ranking is based only on gradient size along the optimisation path. As far as we can tell, the paper checks it with an aggregate comparison to SHAP's top feature and a fixed-feature test, and never tests per query whether the top-ranked features are the ones that matter for the counterfactual.

**What we found.** On 135 test queries of Credit Approval, GradCFA's ranking is better than random but is the least faithful of the informed rankings. Resetting its single top feature breaks the counterfactual in only 22% of cases (random: 7%, Shapley: 65%). A simple ranking by how much the counterfactuals changed each feature is more faithful than GradCFA's. With the repository's default setting, where perturbation does not run, GradCFA's top feature matches Shapley's top feature in only 3 to 5% of queries, which is at or below chance.

**Our improvement.** We replace the gradient size by integrated gradients along the path from the query to each counterfactual (Path-IG). It needs no extra optimisation, only 64 gradient evaluations per counterfactual. It agrees with Shapley on the top feature in 92% of queries and improves faithfulness over GradCFA on both tests, by +0.074 (T1) and +0.106 (T2) with intervals far from zero. It matches Shapley on T1 and is slightly better on T2. The result repeats on three further seeds.

**Honest limits.** T1 is built around Shapley's own background data and T2 follows the path that Path-IG integrates over, so we do not claim Path-IG beats Shapley. Both tests are proxies for faithfulness. One dataset and one trained model. The full method was run with one seed. Path-IG needs the counterfactuals, so it is computed after generation.
