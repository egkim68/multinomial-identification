# Protocol: choosing the reference category in multinomial logistic regression

This protocol covers the simulation reported in the paper "The Effect of the Reference Category on Wald Tests in Multinomial Logistic Regression" (Eungi Kim).

## Origin
The study grew out of a larger protocol (v1, frozen 16 October 2026), which also covered thresholding and lasso comparisons. The revised paper keeps only the reference-category part. Hypotheses H1 to H3 below take their criteria from H4, H6, and H7 of protocol v1. H4 below was written on 25 November 2026, before the results of the present run were seen. It is based on an earlier descriptive run of the smallest-atypicality reference (addendum B1 of protocol v1, seed 9911).

## Data-generating process (`reference_sim.py`)
- *p* = 10 predictors, x ~ N(0, Σ) with Σ_jl = γ^|j − l|.
- For each predictor, the common slope is 0. Each category deviates with probability ρ by an amount U(0.3, 0.9) with a random sign.
- Intercepts are balanced (all 0) or imbalanced (a random permutation of an even grid from 0 to −3).
- Factors, fully crossed: K ∈ {5, 10, 20}; n ∈ {1,000, 5,000}; ρ ∈ {0.1, 0.3, 0.5}; balanced or imbalanced intercepts; γ ∈ {0, 0.5}. That gives 72 cells × 100 data sets = 7,200 data sets. Seed 20261125.
- Data sets with a category count below 5 are skipped. Data sets in which a fit fails (singular information matrix) are recorded and excluded.

## References
Most frequent, first label (code 0), least frequent, and smallest atypicality. For the last, q_k = (β_k − m)ᵀ S (β_k − m) is estimated from the unpenalized fit under the most frequent reference. Here m is the average slope vector weighted by the baseline shares, and S is the sample covariance of the predictors.

## Fit and outcomes
Unpenalized maximum likelihood (statsmodels MNLogit, Newton, at most 100 iterations), with every contrast given a Wald test at 0.05. The outcomes are:
- whether the reference deviates on some predictor;
- the density of the true contrasts;
- the share of significant contrasts;
- Type I error and power;
- the median standard error;
- apparently pervasive predictors (at least half of the contrasts significant, while the predictor truly deviates for fewer than half of the categories).

## Hypotheses (data sets with ρ ≤ 0.3)
- **H1 (balanced intercepts):** the most frequent reference deviates in at least 90% of data sets; the least frequent reference deviates in at most 75%; and the density ratio (most / least frequent) is at least 1.25.
- **H2 (balanced intercepts):** the significance-share ratio (most / least frequent) is at least 1.25; the ratio of apparently pervasive predictors is at least 1.50; the median-SE ratio (least / most frequent) is at most 1.25; and Type I error lies in [0.04, 0.06] for the three fixed rules.
- **H3:** the significance-share ratio is smaller with imbalanced than with balanced intercepts, and the median-SE ratio (least / most frequent) is larger.
- **H4:** the ratio of apparently pervasive predictors (most frequent / smallest atypicality) is at least 1.50 with both intercept patterns; the median-SE ratio (smallest atypicality / most frequent) is at most 1.25 with imbalanced intercepts; and Type I error under the smallest-atypicality reference is at most 0.06 with balanced intercepts.

A hypothesis is supported only if every criterion is met. One-sided p values come from a binomial test for shares, a t test for means, and a bootstrap with 2,000 resamples of data sets for ratios.

## Numerical check and illustration
- `frequency_check.py` compares Equation 4 with exact frequencies (200,000 draws) for 8 configurations × 40 structures.
- `chile.py` refits the carData::Chile survey under all four possible reference categories (three of which the rules select). It is descriptive and is not a hypothesis test.

Addendum (26 November 2026, after the results were seen): data sets with a non-finite standard error under any reference (a sign of quasi-separation) are excluded in `analysis.py`, like failed fits. All hypothesis decisions are the same with and without this exclusion.
