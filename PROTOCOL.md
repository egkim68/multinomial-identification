# Protocol v1 (FROZEN 16 October 2026, 01:00 KST, before the confirmatory run; approved by the author)

Working title: Which Zeros? Identification Choices and Sparse Reporting in Multinomial Logistic Regression: A Simulation Study.
Status: three exploratory runs have been read. explore1 has 32 cells x 15 datasets, explore2 has 16 cells x 15 datasets with one-sided deviations, and explore3 is the Wald study with 24 cells x 50 datasets. A pilot of the new conditions (rho = .5 and AR(.5) predictors; 32 cells x 8 datasets, seed 5501) has also been read. Criteria below were set from these runs with a margin, so a criterion is met only if the confirmatory run shows a clearly stronger pattern than needed.

## Question
A multinomial logit coefficient table is one representative of an equivalence class: adding c_j to every category's coefficient on x_j leaves all probabilities unchanged. Statements of the form "x_j has no effect on category k" therefore depend on the representative. The paper asks how the representatives chosen by common software and practice affect sparse reporting:
- (A) the row centring applied before post-estimation thresholding of a ridge fit;
- (B) the reference category in reference-coded lasso and in unpenalized Wald tables, especially the most-frequent-category rule (the documented Stata mlogit default and common textbook advice).
Theory (THEORY_NOTE.md): Proposition 1 covers the bias of mean, median and shorth centring. Proposition 2 shows that marginal category frequency increases with the Sigma-norm distance of a category's coefficient vector from the average, so that the most frequent category tends to be the most atypical one.

## Data-generating process (multinom_sim.py, wald_sim.py)
p = 10 predictors, x ~ N(0, Sigma) with Sigma_jl = corr^|j - l|. For each predictor, every category shares a common effect N(0, .5^2), which is not identified and is irrelevant. Each category independently deviates from it with probability rho, by a deviation of size U(.3, .9). Signs are mixed (independent) or one-sided (same sign within a predictor). Intercepts are balanced (N(0, .2^2)) or imbalanced (a permutation of an even grid from 0 to -3, a 20:1 frequency range). The true sparsity pattern is the deviation matrix, evaluated on the standardized-predictor scale. For reference-coded methods, the truth is the reference-coded contrast matrix.

## Study 1-2 design (multinom_sim.py; same code as the pilot)
Factors: K in {5, 10, 20, 40}; training n in {500, 2,000}; rho in {.1, .3, .5}; intercepts balanced or imbalanced; signs mixed or one-sided; corr in {0, .5}. Full factorial of 192 cells x 50 datasets = 9,600 datasets, with a validation set of 2,000 and a test set of 5,000 per dataset.
Methods:
- ridge: symmetric, lambda chosen on the validation set from 9 values, no thresholding.
- thr_mean, thr_median, thr_shorth, thr_ref_first, thr_ref_freq, thr_ref_rare: the same ridge fit re-centred per predictor (mean = what ridge returns; median; shorth; or a reference category: first label, most frequent or least frequent in the training data). Then the smallest q-fraction of coefficients is set to zero, with q chosen on the validation set from {0, .05, ..., .95}. There is no refit.
- lasso_sym, lasso_ref_first, lasso_ref_freq, lasso_ref_rare: L1-penalized multinomial logit (FISTA), symmetric or reference-coded, lambda chosen on the validation set from 12 values spanning lambda_max to 10^-3 lambda_max.
Outcomes: test log loss and excess loss over the true model; support F1 against the truth in the method's own coding; share of non-zero coefficients; whether the reference category deviates on any predictor; density of the true reference-coded matrix.
Strata: sparse = rho <= .3; boundary = rho = .5; balanced and imbalanced; mixed and one-sided; corr 0 and .5.

## Study 3 design (wald_sim.py; same code as explore3 plus corr and rho = .5)
Unpenalized reference-coded MLE (statsmodels MNLogit, Newton), with the reference relabelled to code 0. Factors: K in {5, 10, 20}; n in {1,000, 5,000}; rho in {.1, .3, .5}; balanced or imbalanced intercepts; corr in {0, .5}. Mixed signs. 72 cells x 100 datasets. Datasets with any category count below 5 are skipped and reported.
Each dataset is fitted with three references: first label, most frequent and least frequent.
Outcomes: share of the coefficient table significant at .05; Type I error and power against the true contrasts in that coding; median SE; and "spurious pervasive" predictors. A predictor is spurious pervasive if at least half of its contrasts are significant while it truly deviates for fewer than half of the categories.

## Checked before freezing
- The FISTA solver matches scikit-learn 1.9.1 (lbfgs ridge and saga lasso) to within 6e-4 in coefficients and 1e-5 in log loss. Ridge rows have mean 0 and lasso rows have median 0, as the theory requires.
- The unpenalized reference-coded fit (lambda = 1e-9) matches statsmodels MNLogit to 1e-6, which confirms that statsmodels uses the lowest code as the base.
- theory_checks.py reproduces Propositions 1-2 numerically (THEORY_NOTE.md).
- Not done before freezing (R is not available in the run environment; to be reported as a post-freeze check): cross-check in R 4.3.3 that glmnet (alpha = 0 and 1, symmetric) returns mean- and median-centred rows, and that nnet::multinom matches statsmodels on 20 datasets.

## Hypotheses and numeric criteria (exploratory value in brackets; explore = explore1-2, pilot = new conditions)
H1 Centring before thresholding.
- In sparse cells, thr_median has lower test loss than thr_mean in at least 65% of paired datasets [.78 explore; .70 pilot].
- At n = 2,000 in sparse cells, mean support F1 of thr_median exceeds that of thr_mean by at least .08, separately for mixed and one-sided signs [mixed .72 to .84; one-sided .57 to .78 explore; .65 to .82 and .45 to .68 pilot].

H2 One-sided deviations.
- In cells with one-sided signs, rho = .3 and n = 2,000, mean F1 of thr_shorth is at least .75 and at least .25 above thr_mean [.88 vs .47 explore; .80 vs .47 pilot, corr .5].
- In the same cells, thr_shorth has lower test loss than thr_median in at least 75% of paired datasets [.97; .94].

H3 Support recovery versus prediction-tuned lasso.
- (a) At n = 2,000 in sparse cells, mean F1 of thr_shorth exceeds lasso_sym by at least .20, separately for corr 0 and .5 [.84 vs .52; .77 vs .53].
- (b) With corr 0, the mean excess loss of thr_shorth is no more than .002 above that of lasso_sym [-.0005].
- With corr .5, the same contrast is reported descriptively. The pilot favoured lasso (+.0044), and no claim of prediction parity is made there.

H4 The most-frequent category is atypical (Proposition 2). In balanced sparse cells:
- the training-sample most frequent category deviates on at least one predictor in at least 90% of datasets, and the least frequent in at most 75% [.98; .64 pilot];
- the mean density of the true reference-coded matrix under the most frequent reference is at least 1.25 times that under the least frequent [1.47 explore; 1.42 pilot].

H5 Reference-coded lasso. lasso_ref_rare has lower test loss than lasso_ref_freq in at least 70% of paired datasets in balanced cells and in at least 60% overall [.89 and .82 explore; .82 pilot].

H6 Wald tables. In balanced sparse cells:
- the share of significant coefficients under the most frequent reference is at least 1.25 times that under the least frequent [1.59];
- the mean number of spurious pervasive predictors is at least 1.5 times as large [2.36];
- the median SE under the least frequent reference is at most 1.25 times that under the most frequent [1.11];
- pooled Type I error lies in [.04, .06] under every reference [.050-.052].

H7 Boundary condition. With imbalanced intercepts, the ratio for the share of significant coefficients (most frequent / least frequent) is smaller than in balanced cells [1.26 vs 1.59]. The median-SE ratio (least / most frequent) is larger [1.89 vs 1.11].

A hypothesis counts as supported only if every criterion is met in the confirmatory run, with no allowance for Monte Carlo error. The same criteria must also be met in the held-out quarter (rep divisible by 4). With 50 datasets per cell and at least 1,600 paired datasets per stratum, the SE of a win rate near .7 is about .011.

## Descriptive only (no criterion)
- The boundary stratum rho = .5, where the median is no longer anchored to the mode.
- Results by K.
- The selected q.
- thr_ref_first and lasso_ref_first, the R nnet and statsmodels default.
- Whether thresholding or lasso predicts better at n = 500.

## Runs
Confirmatory Study 1-2: SIM_MODE=full SIM_SEED=8823 SIM_REPS=50 (exploratory seeds: explore1 base 20261015; explore2 same base with tag 2; pilot 5501). About 20 CPU hours, extrapolated from pilot timing: about 10 hours on 2 cores, or 2.5 hours on 8.
Confirmatory Study 3: SIM_SEED=8824 SIM_REPS=100 (exploratory seed 3). About 1.5 hours on 2 cores.
No outside registration; dates rest on the deposited protocol and code.

## Real-data illustration (planned, not a hypothesis test)
One public data set with a nominal outcome of at least 8 categories will be fitted with ridge and then reported under mean, median, shorth and most-frequent-reference coding. The illustration shows how the number of "zero" cells and the apparent pattern of effects change with each. The data set is to be chosen and named in an addendum before it is analysed.

## Fingerprints at freeze (unchanged from the draft)
mnl.py md5 f8918b318ad9bd156d2e46c2e92ed446
multinom_sim.py md5 d0447b6a4e6dc8bb523f33552a542542
wald_sim.py md5 1043484c4860e1953691062ee1571322
theory_checks.py md5 aa01161dc4791bc7d8217871d4c7c3c3

## Open at draft
- The R cross-checks listed above.
- The choice of data set for the illustration.
- Formal proofs for the appendix.
- Final novelty wording after a check of citing papers of Zahid and Tutz (2013) and Price et al. (2019); see LITERATURE_NOTE.md.


---
## Addendum A (16 October 2026, written before the sensitivity run; the confirmatory hypotheses above are unchanged)

Purpose: respond to the concern that the mode-sparse truth favours mode-type centres. Experiment 1 is repeated
under two truths that are not mode-sparse, with the same per-data-set code (multinom_sim.one) and a replaced truth
generator (sensitivity_sim.py):
- two_cluster: for each predictor, 60% of categories share one slope and 40% share another, 0.6 apart; scored
  against the representation centred at the larger cluster.
- dense: every category has its own slope drawn from N(0, 0.5^2); there is no shared value and no true zero.
Design: K in {10, 40}, n in {500, 2,000}, balanced intercepts, uncorrelated predictors, 50 data sets per cell
(800 data sets), seed 9901. Run after the confirmatory runs: python3 run_resumable.py sens 9901 50 output/sensitivity.
Descriptive only (no criteria): F1, share kept, and excess loss for mean, median, and shorth thresholding and
symmetric lasso. Expected: shorth thresholding recovers the larger cluster under two_cluster; under dense, no centre
can produce a correct sparse table, and the validated threshold should keep most coefficients.
Execution note: after the container restarted during the confirmatory run, the frozen scripts are now executed
through run_resumable.py, which saves results in batches and skips data sets already saved; the per-data-set code,
grid, and seeds are unchanged.
sensitivity_sim.py md5 3fcc1243e2e9b830cca8d8232a168de6; run_resumable.py md5 3f1d32f1425af6dc7940a13f53f35238


---
## Addendum B (16 October 2026, written before these runs; responds to a third review; descriptive except where noted)

B1. Data-driven reference (Wald). The Discussion recommends choosing the reference with a small atypicality q_k.
Because q_k is estimated from the same data, its effect on inference is tested. Design: K in {5, 10, 20},
n in {1,000, 5,000}, rho in {.1, .3}, balanced and imbalanced intercepts, uncorrelated predictors, 100 data sets per
cell (2,400), seed 9911. References: most frequent, least frequent, first label, and minq (smallest q_k estimated
from the unpenalized fit, with Sigma the sample covariance). Outcomes as in Experiment 2, plus whether the reference
deviates. Pre-stated expectation (not a protocol hypothesis): minq gives the lowest truth density and the fewest
apparently pervasive predictors among the four references in balanced cells, with Type I error within [.04, .06].
B2. Stronger lasso comparators. Design: K in {10, 40}, n in {500, 2,000}, rho in {.1, .3}, mixed and one-sided signs,
gamma in {0, .5}, balanced intercepts, 25 data sets per cell (800), seed 9912, 20-value penalty path. Methods: shorth
thresholding of ridge (as in Experiment 1); symmetric lasso with the penalty chosen by validation minimum, validation
1-SE rule, or BIC on the training data; adaptive lasso (weights 1/|median-centred ridge coefficient|); relaxed lasso
(penalty and mixing weight chosen on validation); and lasso followed by shorth thresholding. Outcomes: F1, share kept,
excess test loss. Descriptive only.
Script addB_sim.py md5 d7edc0dffe601a5c4e3648d86c2580d9. Run after the sensitivity run:
python3 addB_sim.py b1 9911 100 output/addB ; python3 addB_sim.py b2 9912 25 output/addB

Execution notes (16 October 2026, 17:40 KST, before any Addendum B result was inspected):
- Addendum A: the design as stated (2 truths x K {10, 40} x n {500, 2,000} x 50) gives 400 data sets, not 800 as
  written above; 400 were run.
- Addendum B: both parts run through addB_resumable.py (same grid, seeds, and per-data-set functions, saved in
  batches). B1 stopped at one data set whose unpenalized fit had a singular Hessian; following the rule of
  Experiment 2 (wald_sim.py), a data set is now skipped when any of its fits fails, and the count is reported.

---
## Addendum C (24 October 2026, 13:16 KST, written before these runs; responds to a fourth review; descriptive only)

C1. Continuous deviations. Experiment 1 is repeated with deviations drawn from a heavy-tailed continuous
distribution that includes small values: each category deviates with probability rho by |0.2 * t_3| (median about
0.15), with mixed or one-sided signs. Design: K in {10, 40}, n in {500, 2,000}, rho in {.1, .3}, both sign patterns,
balanced intercepts, uncorrelated predictors, 25 data sets per cell (400), seed 9931. Outcomes as in Experiment 1
for mean, median, and shorth thresholding and symmetric lasso. Expectation (not a criterion): all methods lose
support recovery against small deviations; the ordering of the centers under one-sided signs is the question.
C2. Smallest-atypicality reference under one-sided deviations. The B1 design (K in {5, 10, 20}, n in {1,000,
5,000}, rho in {.1, .3}, balanced and imbalanced intercepts, uncorrelated predictors, four references) with
one-sided deviations, 50 data sets per cell (1,200), seed 9921. Outcomes as in B1.
Script addC_sim.py md5 a9e4d77a3ae0eb5369cc4ad07907b091.
Execution note (24 October 2026): Addendum C ran as written (C2: 1,200 data sets, 22 skipped for a sparse category or
a failed fit; C1: 400 data sets).

Erratum (24 October 2026, after all runs): the Addendum A design line gives "(800 data sets)". The design it specifies
(two truths x K in {10, 40} x n in {500, 2,000} x 50 data sets per cell) has 400 data sets, which is what was run and
reported. The design, seed, and code are unchanged; only the stated count was wrong. The solver comparison is
reproduced by solver_check.py (scikit-learn ridge and lasso, statsmodels unpenalized fit; all within 6e-4).
