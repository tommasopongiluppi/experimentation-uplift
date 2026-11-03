# Methodology

Plain-language notes on every method in `abkit`: what it does, the key formula, its assumptions
and how it is tested. Numbers come from the generated reports, never from this file.

## Sample ratio mismatch (SRM) test

`abkit.frequentist.srm_test`, data from `sql/arm_counts.sql`.

**What it does.** Checks that the number of customers in each arm matches the planned split
(equal thirds). A mismatch suggests broken randomisation or lost data, which would bias every
comparison, so under the analysis plan a failure blocks the readout.

**Formula.** Pearson's chi-square goodness-of-fit statistic,
X² = Σ_arms (observed − expected)² / expected, where expected = total × planned share. Under a
correct split, X² follows a chi-square distribution with (number of arms − 1) degrees of
freedom, and the p-value is its upper tail. The check fails if p < 0.001 (`config.SRM_ALPHA`).
This strict threshold is used because the test is run once, on a very large sample, as a
guard against implementation errors rather than as a scientific hypothesis.

**Assumptions.** Customers are assigned independently with fixed probabilities, and expected
counts are large (here over 21,000 per arm), so the chi-square approximation is accurate.

**How it is tested.**
- A perfect 21,000 / 21,000 / 21,000 split gives X² = 0 and p = 1.
- 22,000 / 21,000 / 21,000 gives the closed-form X² = 31.25, with a p-value equal to
  `scipy.stats.chisquare`'s, and fails.
- An unequal planned allocation matches scipy.
- A slow Monte Carlo test draws 2,000 fair multinomial splits of 64,000 customers. At
  alpha = 0.05 it checks that the test rejects about 5% of them, within 3 Monte Carlo standard
  errors (MCSE = sqrt(0.05 × 0.95 / 2000)).

## Standardised mean difference (SMD)

`abkit.balance.smd` and `abkit.balance.balance_table`, data from `sql/balance.sql`.

**What it does.** Measures how different two arms are on a pre-treatment variable, in units of
standard deviations, so variables on different scales (dollars, months, 0/1 indicators) can be
compared. Randomisation should make every SMD close to zero. Each category level is its own
0/1 indicator row.

**Formula.** SMD = (mean_a − mean_b) / sqrt((var_a + var_b) / 2), with sample variances. The
SQL returns each arm's n, mean and variance for every variable and level; Python then forms the
SMD for mens vs control, womens vs control and mens vs womens. |SMD| > 0.1
(`config.SMD_THRESHOLD`) is flagged, a common rule of thumb for a meaningful imbalance. Unlike a
p-value, the SMD does not shrink toward "significant" as the sample grows, which is why it is
preferred for balance.

**Assumptions.** None beyond finite variances. It is a descriptive measure, not a test. If both
variances are zero, the SMD is 0 for equal means and ±infinity otherwise.

**How it is tested.**
- A hand calculation: [1, 2, 3, 4] vs [2, 4, 6, 8] gives −√6 / 2 ≈ −1.2247, both directly and
  end to end through the SQL on a toy table.
- Every mean and SMD in the balance table matches a pandas group-by on the test fixture.
- The table contains one row per variable or level, and only pre-treatment variables.

## Chi-square test of balance for categorical variables

`abkit.balance.categorical_balance_tests`.

**What it does.** For each categorical pre-treatment variable (`history_segment`, `zip_code`,
`channel`), tests whether its distribution differs across the three arms. This is one test per
variable, which complements the per-level SMDs.

**Formula.** Pearson's chi-square test of independence on the levels × arms contingency table:
X² = Σ (observed − expected)² / expected, where expected = row total × column total / grand
total. Degrees of freedom are (levels − 1)(arms − 1). No continuity correction is applied.

**Assumptions.** Independent customers, and expected cell counts large enough for the
chi-square approximation (all cells here hold hundreds or thousands of customers).

**How it is tested.** The statistic, p-value and degrees of freedom match
`scipy.stats.chi2_contingency` on a pandas crosstab of the fixture.

## Pooled correlation of history with outcomes

`sql/eda_correlations.sql`.

**What it does.** Measures, over all customers pooled across arms, how strongly past-year spend
(`history`) is linearly related to each outcome. This is the planning input for CUPED (analysis
plan §7): CUPED with covariate X multiplies the variance of the outcome by about 1 − r². A
small r therefore means little variance reduction.

**Formula.** The Pearson correlation,
r = cov(history, Y) / (sd(history) · sd(Y)), computed by DuckDB's `corr`.

**Assumptions.** It measures only linear association, and is sensitive to the long right tail
of `history` and to the point mass at its minimum. It is pooled across arms, so it reveals
nothing about treatment effects.

**How it is tested.** It matches `numpy.corrcoef` on the fixture for spend, visit and
conversion.

## Metric layer: per-arm sufficient statistics

`abkit.metrics.arm_metrics`, data from `sql/arm_metrics.sql`.

**What it does.** Reduces each arm to a few sums, from which every mean, variance and
covariance used in the readout is formed. This is how production experimentation platforms
work: aggregate in SQL, infer in Python. The database returns per arm: n; the sum and sum of
squares of `visit`, `conversion` and `spend`; the sum and sum of squares of `history` and its
cross-product with each outcome (for CUPED); and the number of converters with the sum and sum
of squares of their spend. Python turns these into a tidy table with one row per arm and metric
(`visit`, `conversion`, `spend`, and `spend_converters`, spend among converters).

**Formula.** With n rows, sum S, sum of squares Q, covariate sum Sx and cross-product P:
mean = S / n, unbiased variance = (Q − S²/n) / (n − 1), and covariance with `history` =
(P − S·Sx/n) / (n − 1). With fewer than two rows, variance and covariance are undefined (NaN).

**Winsorised spend.** The same query takes a `winsorize` parameter. When set, spend is replaced
by min(spend, c), where c is the 95th percentile of converters' spend pooled across all arms
(analysis plan §8), computed in SQL with linear interpolation between order statistics
(`quantile_cont`, the numpy/pandas default). Zero spend is unchanged. The cap is returned with
the result so it can be reported.

**Assumptions.** None beyond finite values. The sum-of-squares formula can lose precision when
the mean is very large relative to the spread; here the outcomes and `history` are on modest
scales and the results agree with pandas to 1e-9.

**How it is tested.**
- On the test fixture, every n, sum, sum of squares, mean, variance and covariance matches a
  pandas group-by to a tolerance of 1e-9, for raw and for winsorised spend.
- The cap equals `numpy.quantile` of converters' spend at 0.95, binds on the fixture, and is
  not the all-customer quantile (which is $0).
- A planted example with converter spend 1 to 100 split unevenly across arms gives the pooled
  cap 1 + 0.95 × 99 = 95.05 in every arm; only the values above it change, each to exactly
  95.05.
- A hand calculation with y = [0, 1, 1, 2] and x = [1, 2, 3, 4] gives mean 1, variance 2/3,
  covariance 1; one-row arms give NaN variance; winsorising with no converters raises an error.

## Difference in proportions: z-test and Newcombe interval

`abkit.frequentist.diff_in_proportions`, inputs from `abkit.metrics.arm_metrics`.

**What it does.** Compares a rate (conversion or visit) between two arms: the difference
p_t − p_c, a p-value and a 95% interval.

**Formula.**
- Test: two-proportion z-test with the pooled rate p̄ = (x_t + x_c) / (n_t + n_c) under the null
  hypothesis of no difference: z = (p_t − p_c) / sqrt(p̄ (1 − p̄) (1/n_t + 1/n_c)), two-sided
  normal p-value.
- Interval: Newcombe's hybrid score interval (Newcombe 1998, method 10). Each arm gets a Wilson
  score interval (l, u); the difference d = p_t − p_c then has
  lower = d − sqrt((p_t − l_t)² + (u_c − p_c)²) and upper = d + sqrt((u_t − p_t)² + (p_c − l_c)²).
  Unlike the simple Wald interval, it stays well behaved for rare events such as a 1% conversion
  rate.

**Assumptions.** Customers are independent and each arm's outcome is binomial. The normal
approximation for the test needs enough events in each arm (hundreds here).

**How it is tested.** The z statistic and p-value match `statsmodels` `proportions_ztest`, the
interval matches `confint_proportions_2indep(method="newcomb")` and the Wilson interval matches
`proportion_confint(method="wilson")`, all to 1e-8, including edge cases with 0 or n events. A
slow simulation of 2,000 experiments of 21,000 customers per arm, at visit and conversion rates
near the observed pooled rates, checks that the interval covers the true difference 95% of the
time, within 3 Monte Carlo standard errors.

## Welch difference in means

`abkit.frequentist.diff_in_means`, inputs from `abkit.metrics.arm_metrics`.

**What it does.** Compares mean spend per customer between two arms without assuming equal
variances. It is the primary analysis (analysis plan §6).

**Formula.** se = sqrt(s_t²/n_t + s_c²/n_c), t = (mean_t − mean_c) / se, with
Welch–Satterthwaite degrees of freedom
ν = se⁴ / ((s_t²/n_t)² / (n_t − 1) + (s_c²/n_c)² / (n_c − 1)), and the interval
diff ± t_{0.975, ν} · se. Everything comes from per-arm n, mean and unbiased variance, so the
row-level data is not needed.

**Assumptions.** Independent customers and a sample mean that is approximately normal. Spend
is far from normal (about 99% zeros and a long right tail), so this relies on the central limit
theorem with about 21,000 customers per arm. The bootstrap below and the coverage simulation
check that this is good enough.

**How it is tested.** The t statistic, degrees of freedom, p-value and interval match
`scipy.stats.ttest_ind(equal_var=False)` to 1e-8. A slow simulation of 2,000 experiments with
zero-inflated, heavy-tailed spend shaped like the data (about 1% converters; converter spend with
a $29.99 floor, median about $81 and a long right tail top-coded at $499) checks that the
interval covers the true difference 95% of the time, within 3 Monte Carlo standard errors. The
true mean of that spend comes from a closed-form expression, itself checked by Monte Carlo.

## Relative lift with a delta-method interval

`abkit.frequentist.relative_lift_ci`.

**What it does.** Expresses an effect as a percentage of the control mean,
(mean_t − mean_c) / mean_c, with a 95% interval.

**Formula.** The interval is built on the log ratio and transformed back:
Var(log(mean_t / mean_c)) ≈ s_t² / (n_t mean_t²) + s_c² / (n_c mean_c²) (delta method), and
the interval is exp(log ratio ± 1.96 · se) − 1. On the log scale the ratio is closer to normal,
and the interval is asymmetric and never falls below −100%. For rates, s² = p (1 − p). The lift
is undefined if either mean is zero.

**Assumptions.** Both means are positive and estimated precisely enough for a first-order
(delta-method) approximation to hold.

**How it is tested.** For rates it equals the Katz log interval for a risk ratio, and matches
`statsmodels` `confint_proportions_2indep(method="log", compare="ratio")` to 1e-8. A closed-form
case (means 2 and 1, variances 4 and 1, n = 100: Var = 0.02) is checked by hand. The coverage
simulation checks its coverage for visit, conversion and spend.

## Percentile bootstrap for spend

`abkit.frequentist.bootstrap_diff_in_means`.

**What it does.** A robustness check on the Welch interval for spend that makes no normal
approximation: resample each arm's customers with replacement 10,000 times, recompute the
difference in means, and take the 2.5% and 97.5% quantiles of the replicates (Deviation D1).

**Formula.** A resample of an arm is fully described by how many times each distinct value is
drawn, which is Multinomial(n, frequency of each value / n). Drawing these counts has exactly
the same distribution as drawing n row indices, but needs one number per distinct value instead
of one per customer. Spend is about 99% zeros, so this is fast and fully vectorised. Each
endpoint is reported with a Monte Carlo standard error from the order statistics at
B·q ± sqrt(B q (1 − q)).

**Assumptions.** Customers are independent and the sample represents the population. Percentile
intervals can under-cover slightly in small samples; with 21,000 customers per arm this is minor.

**How it is tested.** The standard deviation of the replicates matches its exact value,
sqrt(var_t/n_t + var_c/n_c) with n-denominator variances, within 3 Monte Carlo standard errors.
On normal data the interval agrees with the Welch interval up to Monte Carlo error. The same
seed gives identical results.

## Multiple-testing corrections: Holm and Benjamini–Hochberg

`abkit.frequentist.holm`, `abkit.frequentist.benjamini_hochberg`.

**What it does.** Adjusts p-values when several hypotheses are tested together. Holm controls
the family-wise error rate (the chance of any false positive) and is used for the primary family.
Benjamini–Hochberg controls the false discovery rate (the expected share of false positives
among rejections) and is used for the secondary family (analysis plan §6).

**Formula.** Sort the m p-values in ascending order, p_(1) ≤ … ≤ p_(m).
- Holm: p_adj(i) = max over j ≤ i of min(1, (m − j + 1) p_(j)).
- BH: p_adj(i) = min over j ≥ i of min(1, m p_(j) / j).
Reject when p_adj < 0.05.

**Assumptions.** Holm holds under any dependence between tests. BH holds under independence or
positive dependence, which is plausible here since the comparisons share arms and the metrics
are positively related.

**How it is tested.** Both match `statsmodels.stats.multitest.multipletests` (`holm`, `fdr_bh`)
to 1e-12 for families of 1 to 25 p-values, including ties. Invalid inputs raise an error.

## Frequentist readout

`abkit.readout.run_frequentist`, output `reports/tables/readout_frequentist.csv`, the
`frequentist` section of `reports/results.json` and `reports/figures/forest_relative_lift.png`.

**What it does.** Runs every pre-registered comparison (mens vs control, womens vs control,
mens vs womens; first arm minus second) on spend, conversion and visit, plus winsorised spend as
the sensitivity analysis. Each row reports the two means, the absolute difference with its
interval, the relative lift with its interval, the unadjusted and adjusted p-values and the
multiple-testing family. Families: `primary` (spend vs control, Holm), `secondary` (conversion
and visit vs control, and mens vs womens on all three, BH over 7 tests), `sensitivity`
(winsorised spend vs control, Holm) and `sensitivity_secondary` (winsorised mens vs womens,
unadjusted); the last two are set by Deviation D1. Spend rows also carry the bootstrap interval.
The readout refuses to run if the sample ratio mismatch check fails.

**How it is tested.** On the fixture: one row per comparison and metric, family sizes 2 / 7 /
2 / 1, adjusted p-values equal to Holm or BH applied within each family, and means, differences
and Welch p-values that match a row-level pandas and scipy calculation.
