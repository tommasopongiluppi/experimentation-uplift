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

## CUPED (variance reduction with a pre-experiment covariate)

`abkit.cuped.cuped`, inputs from `sql/arm_metrics.sql` via `abkit.cuped.moments_from_stats`.

**What it does.** Removes the part of each customer's outcome that was predictable from before
the experiment, so the same effect is estimated with less noise (analysis plan §7). The
pre-registered covariate is `history`, past-year spend in dollars, used as is.

**Formula.** For each comparison, on the customers of both arms pooled:
θ = cov(Y, X) / var(X), and Y_adj = Y − θ (X − mean(X)). The effect is the Welch difference in
means of Y_adj. Everything comes from per-arm sums (n, ΣY, ΣY², ΣX, ΣX², ΣXY), so no row-level
data is needed: per arm, mean(Y_adj) = mean(Y) − θ (mean_arm(X) − mean(X)) and
var(Y_adj) = var(Y) − 2θ cov(X, Y) + θ² var(X). Reported next to it:
- ρ, the pooled correlation of X and Y;
- the variance reduction, 1 − var(Y_adj) / var(Y), using within-arm variances pooled over the
  two arms (the quantity that drives the standard error); it is close to ρ²;
- the ratio of the CUPED interval width to the unadjusted Welch interval width, about
  sqrt(1 − ρ²).

"Pooled" means the two arms being compared (mens and control for mens vs control, and so on).
θ is the same for both arms, so the adjustment cannot create a difference between them. Pooling
over the two compared arms, rather than all three, is recorded as Deviation D2 in the plan.

**Assumptions.** X is measured before treatment, so the email cannot change it, and by
randomisation mean(X) is the same in expectation in both arms. The adjustment therefore
removes zero in expectation and the estimate stays unbiased, whatever the true relationship
between X and Y. The linear form only affects how much variance is removed, not validity. The
variance formula treats θ as known; the error from estimating it is of order 1/n.

**Why the reduction is small here.** The variance falls by a factor of about (1 − ρ²), so the
gain is governed by ρ², and a correlation of 0.1 would remove only 1% of the variance. Past-year
spend has little to say about this outcome:
- About 99% of customers spend nothing in the two-week window, whatever their history. The
  variance of spend is almost entirely the question of *whether* a customer buys in those two
  weeks, plus the size of a rare order, and a year's total says little about either.
- The window is short. A two-week purchase is a rare, largely random event, even for a good
  customer; the year-long total averages over many such windows, the outcome sees one.
- `history` is itself awkward for a linear fit: about an eighth of customers sit at its
  minimum, and its long right tail (into the thousands) drives θ.
- Spend is top-coded at $499 in the source, which cuts off the large orders that a big past
  spender might place.

So the achieved reduction is reported as it is (the `cuped` section of `reports/results.json`
and `reports/tables/readout_cuped.csv`); nothing was tuned, transformed or added to raise it.
The pooled correlations in the `eda.history_correlations` section of `results.json` predicted
this before any arm was compared. Visit, the most frequent outcome, gains the most.

**How it is tested.**
- On skewed synthetic data, θ, ρ, the adjusted difference, the Welch p-value and interval, the
  variance reduction and the width ratio all match a row-level calculation (adjust each row,
  then `scipy.stats.ttest_ind(equal_var=False)`) to 1e-8.
- With Y = 2X + 5D exactly, θ = 2, the estimate is exactly 5 and the variance reduction is 1.
- Sums taken from the SQL metric layer match pandas on the fixture.
- Monte Carlo, with a planted effect τ = 0.1 and a covariate with known within-arm correlation
  ρ (2,000 customers per arm):
  - unbiased: over 1,000 experiments the mean estimate is within 3 Monte Carlo standard errors
    of τ;
  - variance reduction ≈ ρ² for ρ = 0.3, 0.6, 0.9: the mean reported reduction is within 3
    MCSE of ρ², and the realised variance of the CUPED estimate across experiments, divided by
    that of the raw estimate, is within 3 MCSE of 1 − ρ² (MCSE of a ratio of variances from
    the ratio-estimator formula on paired experiments);
  - no harm: with ρ = 0, the variance ratio is within 3 MCSE of 1;
  - slow: the 95% interval covers τ in 95% of 2,000 experiments, within 3 MCSE
    (sqrt(0.95 × 0.05 / 2000)).

## Regression adjustment (Lin 2013)

`abkit.cuped.regression_adjustment`, covariates from `abkit.cuped.pretreatment_design`, rows
from `sql/adjustment_rows.sql`.

**What it does.** The sensitivity analysis to CUPED in plan §7. It adjusts for every
pre-treatment variable at once and lets their effects differ by arm, which can capture
non-linear relationships (through the `history_segment` bands) and heterogeneous effects that
CUPED's single linear term misses.

**Formula.** For each comparison, OLS on the customers of the two arms:
Y = a + τ D + β' Xc + γ' (D · Xc) + error, where D = 1 for the first arm and Xc are the
covariates centred at their pooled mean. Because of the centring, τ is the average treatment
effect over the pooled sample. Its standard error is HC2 heteroskedasticity-robust
(statsmodels), with a normal 95% interval. Covariates: `recency`, `history`, `mens`, `womens`,
`newbie`, and one indicator per level of `history_segment`, `zip_code` and `channel` except a
fixed reference level (15 columns). Columns that are constant in a sample are dropped.

**Assumptions.** Randomisation and pre-treatment covariates only. The linear model does not
need to be correct: with the interactions, Lin (2013) shows the estimator is consistent and,
asymptotically, never less precise than the unadjusted difference, and HC2 gives valid
standard errors under heteroskedasticity. In finite samples the extra 30 parameters cost a
little, so when the covariates explain almost nothing the interval can be marginally wider
than the raw one.

**How it is tested.**
- The estimate, HC2 standard error and p-value match `statsmodels` formula
  `y ~ d * (x1c + x2c)` with `cov_type="HC2"` to 1e-8.
- With one binary covariate (a saturated model), the estimate equals the stratified difference
  in means weighted by the pooled stratum shares.
- The design matrix has the fixed 15 columns, contains no outcome, and rejects any variable that
  is not pre-treatment.
- Monte Carlo, with an effect that varies with the covariate: unbiased within 3 MCSE over 300
  experiments; slow: 95% coverage within 3 MCSE over 2,000 experiments with heteroskedastic
  noise and a three-level covariate.

## Adjusted readout

`abkit.readout.run_cuped`, output `reports/tables/readout_cuped.csv`, the `cuped` section of
`reports/results.json` and `reports/figures/intervals_raw_cuped_lin.png`.

**What it does.** Applies CUPED and the Lin adjustment to every comparison and metric of the
frequentist readout (spend, winsorised spend, conversion, visit; mens vs control, womens vs
control, mens vs womens). Each row shows the raw Welch difference, the CUPED and Lin estimates
with their intervals, θ, ρ, the variance reductions and the interval-width ratios. Lin's
variance reduction is 1 − (se_Lin / se_raw)². The raw column here is the Welch difference for
rates too (not the Newcombe interval of the headline readout), so the three estimators are
compared on the same footing. Adjusted p-values use the families of the frequentist readout.
These estimates sit next to the headline estimate, never in its place (plan §6).

**How it is tested.** On the fixture: one row per comparison and metric, raw differences and ρ
that match pandas, and the table, figure and results section are written.

## Minimum detectable effect, sample size and power

`abkit.power.mde`, `abkit.power.sample_size`, `abkit.power.power_for_effect`; readout
`abkit.power.run_power`, outputs `reports/tables/power.csv` (MDE at the actual sample size),
`reports/tables/power_sample_size.csv`, `reports/tables/power_curve.csv`, the `power` section of
`reports/results.json` and `reports/figures/power_curve.png`.

**What it does.** Says what the experiment could detect, and what a future one would need,
without looking at the observed effects. For spend, conversion and visit it reports the minimum
detectable effect (MDE) at the actual size of the arms, at the pre-registered level (alpha = 0.05)
and at the level a primary-family test must meet under Holm (alpha / 2 = 0.025), with and without
CUPED; the customers per arm needed to detect a 5%, 10% and 20% relative lift; and the power
curve, power against the true relative lift, at the actual sample size.

**Formula.** For a difference in means or rates with a common variance σ² per customer, n_t and
n_c customers per arm and a two-sided test at level alpha with target power 1 − β:
- se = sqrt(σ² (1 − r) (1 / n_t + 1 / n_c)), where r is the fraction of the variance CUPED
  removes (0 without CUPED, about ρ² with it);
- MDE = (z_{1−alpha/2} + z_{1−β}) · se, and the relative MDE is MDE / baseline mean;
- customers per arm for a lift δ with equal allocation: n = 2 (z_{1−alpha/2} + z_{1−β})² σ² (1 − r)
  / δ², rounded up;
- power at a true difference δ: Φ(δ / se − z_{1−alpha/2}) + Φ(−δ / se − z_{1−alpha/2}). The
  second term, rejecting in the wrong direction, is what the closed-form MDE ignores; it is about
  1e-6 at 80% power and alpha = 0.05, so the curve passes through the MDE at the target power.
At 80% power and alpha = 0.05 the MDE is 2.80 standard errors; at the Holm level it is 3.08.
For a rate p, σ² = p (1 − p). Inputs come from the metric layer: the control arm's mean and
unbiased variance (for a rate, p (1 − p) up to n / (n − 1)); the sample size per arm is the
smallest arm; the CUPED variance reduction is the one achieved on the metric, averaged over the
two email-vs-control comparisons.

**Assumptions.** The normal approximation for the difference in means, the same as the readout's
tests. The variance is taken as common to both arms and equal to the control arm's. For a
zero-inflated outcome like spend the variance grows with the mean, so the treatment arm's variance
under a real lift is larger than the control's, and the MDE and sample sizes for spend are
optimistic by that amount. The MDE uses the control arm's observed baseline and variance, which is
a legitimate use of the data: they describe the noise, not the effect.

**Why "observed power" is not computed.** Post-hoc or "observed" power plugs the observed effect
into the power formula as if it were the true effect. It adds nothing: it is a one-to-one
transformation of the p-value (a p-value of 0.05 always maps to an observed power of about 50%,
whatever the experiment), so a non-significant result always shows "low power" and a significant
one "high power". It is also biased, because the observed effect is a noisy estimate that, when
significant, overstates the true effect (the winner's curse). Power is a property of a design
against a hypothesised effect, so this project computes it only for pre-specified effect sizes:
the MDE at the actual sample size, the sample sizes for 5%, 10% and 20% lifts, and the full
curve against the true lift. None of these use the observed difference between arms.

**How it is tested.** The MDE matches the hand formula with scipy's normal quantiles to 1e-12,
sample size inverts it to 1e-9, and CUPED scales the MDE by sqrt(1 − r). Against
`statsmodels.stats.power`: the power function matches `NormalIndPower.power` and
`TTestIndPower.power` (normal and non-central t) to 1e-6, and `NormalIndPower.power` evaluated at
our MDE and at our exact sample size returns the target power to 1e-6 (the solver's own output
is compared at its 1e-5 root-finding tolerance). A slow Monte Carlo check plants the MDE as the
true effect and confirms the Welch test, the two-proportion z-test and CUPED each reject in 80%
of 4,000 simulated experiments, within three Monte Carlo standard errors. On the fixture, the
tables have one row per metric, level, CUPED flag (and lift), the design inputs equal the
control arm's pandas mean and variance, and every output is written.
