# Phase 1 Results: GradCFA on Obesity Levels (Corrected Pipeline)

## Setup
- **Dataset**: Obesity Levels (UCI, DOI 10.24432/C5H31Z), 2,111 rows, 7 classes, 16 features.
- **Black-box model**: 64-32 MLP with softmax.
  - Accuracy (train / validation / test): **0.9408 / 0.9076 / 0.9031**
  - Paper: **0.90 / 0.84 / 0.85**
  - *Note*: Our model is more accurate than the paper's on validation and test.
- **Target class**: 1 (`Normal_Weight`), as in the paper.
- **Queries**: 368 test rows (every test row the model does *not* predict as class 1).
  - Per original class distribution: 0 -> 60, 2 -> 76, 3 -> 60, 4 -> 63, 5 -> 60, 6 -> 49.
- **Hyperparameters**: The paper's Obesity values (`results_obesity/hyperparameters_obesity.json`):
  - All loss weights = 0.5, `penalty_scale` = 0.1, `perturb_scale` = 0.5.
  - Thresholds: tau_prox = tau_spars = 0.3, tau_plaus = 0.4, tau_div = 0.4, tau_loss = 0.4.
- **Configuration run**: Perturbation OFF (`high_loss_threshold = 1e9`, `loss_converge_maxiter = 1`), penalty ON, `init_near_query_instance = False` (random start, see Table 2). Same as `C_nopert` on Credit.
- **Seeds**: 1 only (`seed = 42`). No seed variance is available for Obesity.
- **Metrics**: Same as the Credit report. Proximity, sparsity, diversity, and confidence come from the repository's `Evaluate` class. Plausibility is the paper's Eq. 5 with k = 3 (k is not stated in the paper).
  - Average = mean of (1 - proximity), (1 - sparsity), (1 - plausibility), diversity, and confidence.
- **Run time**: ~27 seconds per query; 368 queries evaluated across 6 shards.

---

## Table 1: Paper versus Reproduction (368 queries, 1 seed, perturbation off)

| Metric | Paper (GradCFA) | Ours (Corrected) |
|---|---|---|
| Proximity | 0.21 | 0.407 |
| Sparsity | 0.23 | 0.414 |
| Plausibility (Eq. 5, k = 3) | 0.34 | 0.495 |
| Diversity | 0.92 | 0.846 |
| Confidence | 0.76 | 0.664 |
| Fraction valid | *not reported* | 0.982 |
| **Average of the five** | **0.78** | **0.639** |

---

## Table 2: Before and After the Initialisation Fix (Same Model, Same Hyperparameters)

The first Obesity run used the repository default `init_near_query_instance = True`, which starts the 5 counterfactuals as near-copies of the query. The corrected run starts them from random values, as the binary generator did on Credit. The "Before" numbers are confirmed from `results_obesity/summary_C_nopert.txt`.

| Metric | Paper | Before (Near-Query Start) | After (Random Start) |
|---|---|---|---|
| Proximity | 0.21 | 0.099 | 0.407 |
| Sparsity | 0.23 | 0.249 | 0.414 |
| Plausibility | 0.34 | 0.497 | 0.495 |
| Diversity | 0.92 | 0.307 | 0.846 |
| Confidence | 0.76 | 0.616 | 0.664 |
| **Average** | **0.78** | **0.616** | **0.639** |

- **Diversity jumped from 0.31 to 0.85**, close to the paper's 0.92. This confirms that the near-query start was the direct cause of the diversity collapse.
- **Proximity and sparsity worsened** (0.10 -> 0.41, 0.25 -> 0.41), approximately double the paper's reported values. The counterfactuals spread out to achieve diversity, moving further away from the query.
- The average rose only from 0.616 to 0.639: fixing diversity helps, but does not close the overall gap to the paper.
- Every query achieved at least one valid counterfactual (0.0% failure rate).

---

## Table 3: Where the Gap to the Paper Comes From (Corrected Run)

| Metric | Contribution to Average Gap (Paper minus Ours) |
|---|---|
| Proximity | 0.039 |
| Sparsity | 0.037 |
| Plausibility | 0.031 |
| Confidence | 0.019 |
| Diversity | 0.015 |
| **Total** | **0.141** (0.780 vs. 0.639) |

- **Proximity and sparsity together account for over half of the gap**. This is the inverse of Credit Approval, where plausibility formed the largest discrepancy.
- Diversity now constitutes only a minor fraction of the gap (0.015).

---

## Table 4: Per Original Class Breakdown (Corrected Run)

| Class | Queries | Proximity | Sparsity | Plausibility | Diversity | Confidence | Fraction Valid | Average |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 60 | 0.440 | 0.455 | 0.502 | 0.852 | 0.672 | 0.990 | 0.625 |
| 2 | 76 | 0.424 | 0.410 | 0.495 | 0.847 | 0.658 | 0.976 | 0.635 |
| 3 | 60 | 0.371 | 0.383 | 0.492 | 0.831 | 0.655 | 0.983 | 0.648 |
| 4 | 63 | 0.396 | 0.396 | 0.492 | 0.847 | 0.670 | 0.987 | 0.646 |
| 5 | 60 | 0.405 | 0.429 | 0.493 | 0.857 | 0.672 | 0.980 | 0.640 |
| 6 | 49 | 0.399 | 0.415 | 0.495 | 0.844 | 0.657 | 0.971 | 0.639 |

### Paper's Table II Multi-Class Breakdown (Confirmed from PDF)
- **Paper's per-class Proximity**: Class 0: 0.21, Class 1: -, Class 2: 0.21, Class 3: 0.22, Class 4: 0.26, Class 5: 0.29, Class 6: 0.35.
- **Paper's per-class Average**: Class 0: 0.75, Class 1: -, Class 2: 0.78, Class 3: 0.72, Class 4: 0.69, Class 5: 0.67, Class 6: 0.63.
- **Paper's per-class Sparsity**: Class 0: 0.25, Class 2: 0.23, Class 3: 0.27, Class 4: 0.23, Class 5: 0.26, Class 6: 0.31.
- **Paper's per-class Plausibility**: Class 0: 0.39, Class 2: 0.34, Class 3: 0.52, Class 4: 0.59, Class 5: 0.61, Class 6: 0.67.
- **Paper's per-class Diversity**: Class 0: 0.90, Class 2: 0.92, Class 3: 0.87, Class 4: 0.84, Class 5: 0.85, Class 6: 0.83.
- **Paper's per-class Confidence**: Class 0: 0.71, Class 2: 0.76, Class 3: 0.73, Class 4: 0.70, Class 5: 0.67, Class 6: 0.64.

### Key Observation:
- The paper reported that proximity strictly degrades as the query class gets further from normal weight (rising from 0.21 to 0.35).
- **Our reproduction does not exhibit this trend**: Proximity is flat (0.37 - 0.44) across all classes.
- Our average per class remains steady (0.625 - 0.648), whereas the paper drops monotonically from 0.78 to 0.63.

---

## Table 5: Feature Attribution (GradCFA's Own Scores)

*Paper finding: Weight ranked first, followed by MTRANS.*

| Feature | Mean Score | Mean Rank | Top-1 Share |
|---|:---:|:---:|:---:|
| **Weight** | 0.614 | 1.00 | 100% |
| **MTRANS** | 0.120 | 2.24 | 0% |
| Height | 0.086 | 4.91 | 0% |
| CAEC | 0.078 | 4.92 | 0% |
| SMOKE | 0.076 | 5.18 | 0% |
| CALC | 0.074 | 5.51 | 0% |
| Gender | 0.074 | 5.62 | 0% |
| SCC | 0.055 | 7.82 | 0% |
| FHWO | 0.040 | 10.11 | 0% |
| FAVC | 0.037 | 10.60 | 0% |
| Age | 0.033 | 11.44 | 0% |
| CH2O | 0.033 | 11.45 | 0% |
| FCVC | 0.028 | 12.38 | 0% |
| FAF | 0.021 | 13.93 | 0% |
| TUE | 0.019 | 14.17 | 0% |
| NCP | 0.017 | 14.72 | 0% |

- **Exact reproduction of ranking**: Weight is unanimously ranked #1 across all 368 queries, with MTRANS ranked #2 on average.
- Weight's score is over 5x higher than the runner-up, making it unequivocally dominant on this dataset.

---

## Summary of Findings

1. **Reproduced**: GradCFA feature ranking (**Weight #1, MTRANS #2**); 98.2% valid counterfactual generation rate.
2. **Reproduced after fix**: Diversity restored (**0.85 vs. paper's 0.92**) by initiating search from random values rather than query copies.
3. **Not reproduced**: The paper's average score (**0.78 vs. 0.64**). Proximity and sparsity are roughly double the paper's values.
4. **Not reproduced**: The distance-dependent per-class trend reported in the paper.
5. **Model validation**: Our classifier achieved higher test accuracy (0.903 vs. 0.85), proving that the gap does not stem from an underperforming predictive model.

---

## Stated Limitations

- **Single seed**: Run evaluated on seed 42 only (perturbation off).
- **Ablation grid**: Multi-seed and perturbation on/off grids not replicated due to multiclass compute overhead.
- **Initialisation choice**: `init_near_query_instance = False` was our architectural intervention required to unfreeze diversity; this setting is now documented explicitly in `results_obesity/hyperparameters_obesity.json`.
- **Shard verification**: Verified that all 6 shards (`full_corrected_s0of6.csv` through `s5of6.csv`) exist locally and aggregate to 368 queries matching `summary_corrected.txt` to 4 decimal places.
