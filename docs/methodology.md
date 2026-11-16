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

## Beta-Binomial Bayesian readout for rates

`abkit.bayes.posterior`, `abkit.bayes.compare_rates`, `abkit.bayes.prob_b_greater_a_exact`,
`abkit.bayes.decide`; readout `abkit.bayes.run_bayes_rates`, outputs
`reports/tables/readout_bayes_rates.csv`, the `bayes_rates` section of `reports/results.json` and
`reports/figures/bayes_p_better_by_prior.png`.

**What it does.** Answers the question a decision maker actually asks about conversion and visit:
how likely is it that the email beats control, and how much do we expect to lose if we ship it
and are wrong? Each arm's rate gets a Beta posterior; the two posteriors are compared on 200,000
posterior draws per arm (analysis plan §9, Deviation D3), and the pre-registered decision rule is
applied. This is the Bayesian counterpart of the two-proportion rows of the frequentist readout,
run on the same successes and customers per arm from the SQL metric layer.

**Formula.**
- Posterior per arm: with prior Beta(α0, β0), x successes in n customers, the posterior is
  Beta(α0 + x, β0 + n − x), with mean α / (α + β) and variance αβ / ((α + β)² (α + β + 1)).
- Monte Carlo summaries, from independent draws p_t and p_c (`config.SEEDS["bayes"]`): the
  posterior of the absolute lift p_t − p_c and of the relative lift p_t / p_c − 1, each with a
  95% equal-tailed credible interval (2.5% and 97.5% quantiles of the draws); P(treatment better)
  = P(p_t > p_c); the expected loss of shipping, E[max(p_c − p_t, 0)], the average shortfall if
  the email is shipped and is in fact worse; and the expected loss of keeping control,
  E[max(p_t − p_c, 0)]. The two losses differ by exactly the posterior mean of the difference.
  The exact posterior mean and standard deviation of the difference (from the Beta formulas, the
  arms being independent) are reported next to the Monte Carlo values.
- Monte Carlo standard errors: sd / sqrt(N) for a mean; sqrt(p̃ (1 − p̃) / N) for a probability,
  with the Agresti–Coull proportion p̃ = (k + 2) / (N + 4) so the error stays positive when all N
  draws agree (the plug-in formula gives exactly zero there, which is wrong); and for a quantile
  the order statistics at N q ± sqrt(N q (1 − q)), as for the bootstrap.
- Exact P(p_B > p_A). For independent Betas with integer α_B (Miller 2015):
  P(p_B > p_A) = Σ_{i=0}^{α_B − 1} B(α_A + i, β_A + β_B) / ((β_B + i) B(1 + i, β_B) B(α_A, β_A)).
  The terms are summed on the log scale. For any parameters, the same probability is
  ∫ f_B(x) F_A(x) dx by adaptive quadrature; every row of the table carries this exact value
  next to the Monte Carlo one.
- Decision rule (plan §9): **ship** if P(better) ≥ 0.95 and the expected loss of shipping is
  below 1% of the control arm's rate; **don't ship** if the mirror image holds (P(control better)
  ≥ 0.95 and the expected loss of keeping control is below the same threshold); **keep testing**
  otherwise. The threshold is 1% of the observed control-arm rate (`config.BAYES_P_BETTER`,
  `config.BAYES_LOSS_FRACTION`). For the rates the rule is reported for information; the primary
  decision is on spend.

**Assumptions.** Independent customers with a binomial outcome per arm, and independent
posteriors across arms (the arms are disjoint sets of customers and the prior is set per arm).
The conclusions depend on the prior, which is the point of the next section. With a flat prior
and about 21,000 customers per arm the posterior is driven by the data: the credible interval
coincides with the frequentist interval to a relative gap of order 1 / sqrt(n), under 1%.

**How it is tested.**
- The closed form gives 1/2 for two uniforms and 2/3 for Beta(2, 1) against a uniform (the mean of
  a density 2x); it is symmetric (P(B > A) + P(A > B) = 1 to 1e-10) and matches the quadrature to
  1e-8 for small, visit-sized and conversion-sized posteriors.
- The Monte Carlo P(B > A) is within 3 Monte Carlo standard errors of the closed form for the
  same six parameter sets; when every draw agrees, the adjusted standard error is positive and
  still brackets the exact value.
- Posterior parameters, means and variances equal the Beta formulas and `scipy.stats.beta`; the
  Monte Carlo mean and standard deviation of the difference are within 3 MCSE of the exact ones;
  the two expected losses differ by the Monte Carlo mean exactly.
- With a flat prior and 21,000 customers per arm, every endpoint of the 95% credible interval
  for the difference and for the relative lift is within 2% of the interval width of the
  Newcombe and delta-method confidence intervals, at visit-like and conversion-like rates.
- The decision rule returns each of the three outcomes on hand-chosen inputs, including the
  boundary cases.
- On the fixture: one row per metric, prior and comparison; successes, customers, posterior
  parameters and thresholds match pandas; Monte Carlo P(better) is within 4 MCSE of the exact
  value in all 30 rows at once (4 rather than 3 because 30 checks run together); the table,
  figure and results section are written as strict JSON.
- Slow, calibration: 2,000 experiments where the true rates of both arms are drawn from a
  Beta(3, 20) prior, 2,000 customers per arm are simulated, and the posterior uses that same
  prior. The 95% credible intervals for the difference and for the relative lift contain the
  truth 95% of the time, within 3 Monte Carlo standard errors, and the average P(better) equals
  the share of experiments where the treatment really is better, within 3 MCSE of the paired
  difference. This is the frequentist property a correct Bayesian interval must have when the
  prior is true.

## Prior sensitivity

`abkit.bayes.shared_prior`, `abkit.bayes.rate_priors`; the `prior` column of
`reports/tables/readout_bayes_rates.csv` and `reports/figures/bayes_p_better_by_prior.png`.

**What it does.** Repeats the whole Bayesian readout under five priors and reports, for each,
P(better), both expected losses, the credible interval and the decision. The primary analysis
uses the flat Beta(1, 1). The sensitivity analyses use a **shared informative prior**: the same
Beta for every arm, with mean equal to the rate pooled across all three arms and strength
n0 = α0 + β0 ∈ {100, 1,000, 10,000, 100,000} pseudo-customers (the first three pre-registered in
plan §9, the fourth added as a stress test in Deviation D3). The pooled mean uses the data, but
symmetrically, so it cannot favour either email.

**Formula.** α0 = n0 · p̄ and β0 = n0 · (1 − p̄). Because every arm gets the same prior, each
posterior mean is pulled toward p̄ by the same amount, and with equal arm sizes n the posterior
mean difference is exactly (x_t − x_c) / (n + n0): the observed difference shrunk by the factor
n / (n + n0). At n0 = 100 the shrinkage is under 1% and the prior is invisible; at
n0 = 100,000, about five arm-sizes of pseudo-customers, the difference is shrunk to about a sixth
and P(better) moves toward 1/2. The prior is sceptical by construction: it has to be overcome by
the data, and the strength at which a conclusion flips is the honest measure of how robust that
conclusion is.

**Why this matters.** A Bayesian conclusion is a function of the prior as much as of the data,
and a single posterior probability hides that dependence. Reporting P(better) across a ladder of
prior strengths turns "the email wins with probability 0.98" into "the email wins under any prior
weaker than N pseudo-customers", which is the statement a sceptical reader can check against
their own beliefs. This is the same question the author's MSc thesis addressed, Bayesian
inference and the sensitivity of conclusions to the prior: there, the question was how far a
conclusion survives as the prior is moved; here, the ladder of shared priors is that analysis in
its simplest conjugate form. The effect that survives the strongest sceptical prior is the one to
act on; an effect that vanishes at n0 = 1,000 was never more than a suggestion. The Bayesian
bootstrap for spend has no tunable prior, so its sensitivity check is the winsorised analysis
(plan §8).

**How it is tested.** The shared prior has the requested mean and strength and the labels
`flat`, `shared_100`, …; invalid rates and strengths raise. With equal arms the posterior mean
difference equals (x_t − x_c) / (n + n0) to 1e-12, decreases monotonically along the ladder
100 → 1,000,000, and P(better) falls toward 1/2 while staying above it. On the fixture the flat
prior rows have prior mean 1/2 and every shared row has the pooled rate as its prior mean.

## Bayesian bootstrap for spend per customer

`abkit.bayes.bayesian_bootstrap_means`, `abkit.bayes.bayesian_bootstrap_moments`,
`abkit.bayes.compare_means`, `abkit.bayes.recommend`; readout `abkit.bayes.run_bayes_spend`,
outputs `reports/tables/readout_bayes_spend.csv`, the `bayes_spend` section of
`reports/results.json` and `reports/figures/bayes_spend_diff_posterior.png` (its binned densities
are in `reports/tables/bayes_spend_posterior_hist.csv`).

**What it does.** The Bayesian answer on the primary metric, spend per customer (analysis plan
§9): the posterior of the difference in mean spend between each email and control (and mens vs
womens), its 95% credible interval, P(email better), the expected loss in dollars per customer of
shipping it when it is in fact worse, and the same numbers per 10,000 emails sent. The
pre-registered decision rule is applied, and if both emails qualify the one with the higher
posterior mean spend is recommended with P(mens > womens) reported alongside. Raw spend is the
primary analysis; the same model on winsorised spend (plan §8) is the sensitivity analysis, since
the Bayesian bootstrap has no tunable prior.

**Formula.** The Bayesian bootstrap (Rubin 1981) treats the n customers of an arm as the support
of the population and puts a flat Dirichlet(1, …, 1) posterior on the probability of each one:
a posterior draw of the arm's mean spend is Σ w_i y_i with w ~ Dirichlet(1, …, 1). It is the
Bayesian twin of the percentile bootstrap: the ordinary bootstrap draws multinomial counts with
mean 1, the Bayesian bootstrap draws continuous weights with mean 1/n.
- **Collapsing the zeros, exactly.** About 99% of customers spend nothing, so most of the n
  weights multiply a zero. If the value v occurs c_v times, only the total weight on v,
  W_v = Σ_{i: y_i = v} w_i, matters: Σ w_i y_i = Σ_v W_v v. By the aggregation property of the
  Dirichlet distribution (summing the components of a Dirichlet(α_1, …, α_n) over disjoint groups
  gives a Dirichlet with the group sums as parameters), (W_v)_v ~ Dirichlet(c_v)_v, so the
  zero-spend customers collapse into one component with parameter n_zero, and the ties among
  converters (the $29.99 floor, the $499 cap) collapse the same way. The implementation draws a
  Dirichlet over the distinct values with their counts as parameters: the same posterior as the
  n-dimensional one, at the cost of a few hundred numbers per draw instead of 21,000.
- **Exact posterior moments.** E[w_i] = 1/n, Var(w_i) = (n − 1) / (n² (n + 1)) and
  Cov(w_i, w_j) = −1 / (n² (n + 1)), so the posterior mean of the arm's mean is the sample mean ȳ
  exactly and its posterior variance is Σ (y_i − ȳ)² / (n (n + 1)), the squared standard error
  times (n − 1) / (n + 1). These are reported next to the Monte Carlo values; the Bayesian
  bootstrap is, to order 1/n, the sampling distribution of the mean.
- **Comparisons.** Each arm is drawn once (200,000 draws, `config.SEEDS["bayes"] + 1`) and
  reused across its comparisons, so the three comparisons share one joint posterior. The
  summaries and their Monte Carlo standard errors are those of the rate readout: credible
  interval from the 2.5% and 97.5% quantiles, P(better), the expected loss of shipping
  E[max(θ_c − θ_t, 0)] and of keeping control E[max(θ_t − θ_c, 0)] in dollars per customer, and
  the posterior median. The incremental revenue per 10,000 emails is the difference, its interval
  and the expected loss multiplied by 10,000 (`config.BUDGET`).
- **Decision rule** (plan §9): ship if P(better) ≥ 0.95 and the expected loss of shipping is
  below 1% of the observed control-arm mean spend; don't ship on the mirror image; keep testing
  otherwise (Deviation D3 for the two outcomes the plan left open). If both emails qualify, the
  higher posterior mean spend wins and P(mens > womens) is reported next to it.

**Assumptions.** Customers are independent and the sample is the population's support: the
posterior cannot put mass on spend values that nobody in the sample reached, so it says nothing
about orders above the $499 top-code. The flat Dirichlet is a prior on the empirical
distribution, not on a parameter, which is why there is no prior-strength ladder here. With
about 21,000 customers per arm and 120 to 270 converters the posterior of the mean is close to
normal and the credible interval tracks the Welch and percentile-bootstrap intervals; with a
handful of converters it would be noticeably skewed and the intervals would differ.

**How it is tested.**
- On 40 customers with 8 converters (including ties), the collapsed bootstrap and the naive
  40-dimensional Dirichlet agree: both means are within 3 Monte Carlo standard errors of the
  exact posterior mean, both standard deviations within 3 MCSE (kurtosis-based) of the exact
  posterior standard deviation, every quantile from 2.5% to 97.5% agrees within 3 combined
  quantile MCSEs, and a two-sample Kolmogorov–Smirnov test does not tell the two samples apart.
- With one converter among 200 customers the mean is v · W with W ~ Beta(1, 199), and the Monte
  Carlo quantiles match that closed form within 3 MCSE.
- At n = 20,000 with 200 converters the exact posterior mean equals the sample mean, the exact
  posterior standard deviation equals the standard error times sqrt((n − 1) / (n + 1)) (within
  0.01% of the SE), and the Monte Carlo mean and standard deviation match the exact values
  within 3 MCSE.
- A planted effect between two simulated arms is recovered: the exact difference is the
  difference of sample means, the Monte Carlo difference is within 3 MCSE of it, the two expected
  losses differ by the Monte Carlo mean exactly, and identical arms give P(better) = 1/2 within
  Monte Carlo error.
- The tie-break returns the higher posterior mean when both emails qualify, the single
  qualifying email when one does, and none when neither does, ignoring the winsorised and
  two-part rows.
- On the fixture: one row per model, metric and comparison; customers, converters, means,
  thresholds and decisions match pandas; the exact and Monte Carlo differences agree within
  4 MCSE across all rows; the per-10,000 columns are the per-customer ones times the budget;
  the binned densities integrate to 1; the tables, figure and results section are written as
  strict JSON.

## Two-part log-normal cross-check and posterior predictive check

`abkit.bayes.nig_posterior`, `abkit.bayes.draw_nig`, `abkit.bayes.two_part_draws`,
`abkit.bayes.posterior_predictive_check`; the `two_part_lognormal` rows of
`reports/tables/readout_bayes_spend.csv`, `reports/tables/bayes_spend_ppc.csv` and the `arms`
and `ppc` lists of the `bayes_spend` section.

**What it does.** A parametric second opinion on the Bayesian bootstrap, which is
non-parametric. Mean spend per customer is written as (probability of buying) × (mean spend
among buyers), each part gets a conjugate model, and the posterior of the product is compared
with the bootstrap's. A posterior predictive check then asks whether the log-normal actually
reproduces the observed converters' spend, in particular its tail, and the answer is reported
whatever it is.

**Formula.**
- Conversion: x converters in n customers, flat Beta(1, 1) prior, posterior Beta(1 + x,
  1 + n − x), as in the rate readout.
- Spend among converters: log spend ~ Normal(μ, σ²) with a Normal-Inverse-Gamma prior,
  σ² ~ Inverse-Gamma(α0, β0) and μ | σ² ~ Normal(μ0, σ² / κ0). With k converters of log-spend
  mean x̄ and sum of squares S: κ_n = κ0 + k, μ_n = (κ0 μ0 + k x̄) / κ_n, α_n = α0 + k / 2,
  β_n = β0 + S / 2 + κ0 k (x̄ − μ0)² / (2 κ_n). The marginal of μ is Student-t with 2 α_n degrees
  of freedom, centre μ_n and scale sqrt(β_n / (α_n κ_n)); the marginal of σ² is
  Inverse-Gamma(α_n, β_n). The prior is weak and the same for every arm: μ0 = 0, κ0 = α0 = β0 =
  0.01, a hundredth of a pseudo-observation, so the posterior sits on the maximum-likelihood fit.
- Mean spend per customer: p · exp(μ + σ² / 2), drawn as a product of independent posterior
  draws (p; then σ², then μ | σ²). The difference between arms is summarised exactly as for the
  bootstrap. One caveat is reported with it: under an Inverse-Gamma posterior for σ² the exact
  posterior mean of exp(σ² / 2) is infinite (the Inverse-Gamma tail decays polynomially), so the
  Monte Carlo mean is the mean of the bulk; the posterior median and the credible interval are
  unaffected, and the table carries both. The exact-moment columns are left empty for these rows.
- Posterior predictive check, per arm: for 4,000 posterior draws of (μ, σ²), a replicate
  dataset of the arm's k converters is simulated from the fitted log-normal, and the mean,
  median, 90th, 95th and 99th percentiles, maximum, share at or above the $499 cap and share at
  or below the $29.99 floor are computed for the replicates and for the data. Each statistic
  gets a posterior predictive interval and p-value, P(replicate ≥ observed); p-values near 0 or
  1 mean the model does not reproduce that feature. The model's implied P(spend ≥ 499) =
  E_post[1 − Φ((log 499 − μ) / σ)] is reported next to the observed share of converters at
  exactly $499.

**What the check shows on the real data** (the values are in `bayes_spend_ppc.csv`; read them
there). The log-normal reproduces the mean, the median and the upper percentiles of converters'
spend within their posterior predictive intervals in every arm, and its implied probability of
reaching $499 is of the same order as the observed share of converters sitting at the cap. Where
it fails, and fails clearly, is the left tail: the share of converters at or below the $29.99
floor is about twice what the log-normal predicts, with posterior predictive p-values at or near
zero in every arm. The data have a point mass at the floor (the cheapest order), and a continuous
model has no such mass. The cap shows up more gently: the observed maximum sits at the cap in
every arm and low in the predictive distribution of the maximum, and the 99th percentile is
pinned at $499 where the model expects a spread of values above it.

**What the top-coding does to the parametric model.** The log-normal is fitted to the capped
values as if they were exact, so every order above $499 enters the likelihood as $499. That
pulls μ and σ² down relative to the uncapped distribution, so exp(μ + σ² / 2) understates the
mean of true spend. At the same time the fitted log-normal keeps a tail above $499 that the data
cannot have, so as an estimate of the mean of **top-coded** spend, which is what the bootstrap
and the frequentist readout estimate, it is pulled the other way. The two biases partly offset,
and with 2% of converters at the cap both are small next to the posterior spread, which is why
the two models agree here. The proper fix is a censored likelihood for the values at $499
(treating them as "at least $499"), which is no longer conjugate and would still rest on the
log-normal shape, which the floor already contradicts. The bootstrap needs none of this: it
estimates the mean of spend as recorded, and the plan states the top-code as a limitation of
every spend effect (plan §8).

**Assumptions.** Independent customers; conversion and spend among converters independent given
the data (true of the likelihood factorisation, and a modelling choice for the priors);
log-normal spend among converters, which the check above rejects in the left tail. The
cross-check is reported next to the bootstrap, never in its place.

**How it is tested.**
- The conjugate update matches a hand-computed posterior (prior (0, 1, 1, 1) and data 1, 2, 3
  give κ_n = 4, μ_n = 1.5, α_n = 2.5, β_n = 3.5) and its derived moments; invalid priors and
  empty data raise.
- Monte Carlo draws of μ and σ² match their Student-t and Inverse-Gamma marginals from
  `scipy.stats` in mean and in the 2.5%, 50% and 97.5% quantiles, within 3 MCSE.
- On 400,000 simulated customers with p = 0.02 and log-normal spend (μ = 4.4, σ = 0.7), the
  posterior centres on the maximum-likelihood fit, the true p, μ, σ², mean spend per customer and
  mean spend among converters all lie within 4 posterior standard deviations of their posterior
  means, and the conversion posterior has the right Beta parameters.
- The posterior predictive check on truly log-normal data gives p-values between 0.01 and 0.99
  for every statistic, observed values inside every predictive interval and an implied tail
  probability within 25% of the truth; heaping a quarter of converters on the floor drives the
  floor statistic's p-value below 0.01.
- When spend really is log-normal and the arms are large (200,000 customers each), the bootstrap
  and two-part posteriors of the difference agree: centres within a tenth of a posterior
  standard deviation, spreads within 10%, P(better) within 0.02 and interval endpoints within
  0.15 posterior standard deviations. On the fixture, whose spend is Gamma with 10 to 30
  converters per arm, the two models are allowed to disagree, and the table carries both rows.

## Frequentist and Bayesian readouts: when they agree, when they could not, and what each answers

The spend readout now exists three times: Welch's interval with Holm-adjusted p-values
(`readout_frequentist.csv`), the percentile bootstrap next to it, and the Bayesian bootstrap
with P(better) and expected loss (`readout_bayes_spend.csv`). They are built to be compared.

**What question each one answers.** The frequentist p-value answers: *if the email had no
effect, how often would data this extreme appear?* It is a statement about the procedure's error
rates over imagined repetitions, and the Holm correction keeps the chance of any false claim in
the primary family at 5%. The confidence interval is the set of effects the data do not reject
at that level. The Bayesian posterior answers: *given these data and a flat prior, what is the
probability that the email is better, and by how much?* P(better) is a probability about the
effect, which the p-value is not, and the expected loss turns the posterior into a decision by
weighing how wrong shipping could be in dollars, which no p-value does. The credible interval
is the set of effects that hold 95% of the posterior probability.

**When they agree.** Here, and in any large randomised test with a flat prior. With about
21,000 customers per arm the posterior of the mean is driven by the data: the Bayesian
bootstrap's posterior mean is the sample mean exactly, its posterior standard deviation is the
standard error times sqrt((n − 1) / (n + 1)), and its credible interval, the Welch interval and
the percentile-bootstrap interval differ by a small fraction of their width (compare the spend
rows of the two tables). One minus P(better) is then numerically close to the
one-sided p-value, so P(better) ≥ 0.95 and a one-sided p below 0.05 pick out the same effects.
Under those conditions the two readouts are the same arithmetic read in two languages, and
reporting both costs nothing but adds the decision layer: the expected loss and the revenue per
10,000 emails are what a stakeholder can act on.

**When they could disagree.**
- *Priors with weight.* The rate readout shows it: under a shared prior worth a few arm-sizes of
  pseudo-customers the posterior difference shrinks toward zero and P(better) drops while the
  frequentist p-value, which has no prior, stays put. The Bayesian bootstrap has no such knob;
  its sensitivity check is winsorisation.
- *Multiple comparisons.* The frequentist readout adjusts p-values within families (Holm, BH);
  the posterior probabilities are not adjusted and each comparison is read on its own. An effect
  can be significant unadjusted, non-significant after Holm, and still have P(better) above 0.95.
  The decision rule's loss threshold, not a multiplicity correction, is what keeps the Bayesian
  readout from shipping on noise, and it does not control a family-wise error rate.
- *Small samples and skew.* With a handful of converters the posterior of a mean is skewed and
  the credible interval is asymmetric where the Welch interval is symmetric; the two can then
  cover different ranges and reach different conclusions near the boundary. The fixture, with
  10 to 30 converters per arm, is where the bootstrap and the parametric two-part model are
  allowed to disagree even on the sign of an effect.
- *Different estimands.* The two-part model estimates a log-normal mean; the bootstrap and Welch
  estimate the mean of the recorded, top-coded spend. When the model is wrong in the tail,
  as the posterior predictive check shows it is at the floor, they estimate different things
  and the gap is model error, not sampling error.
- *Sequential looks.* A posterior is the same whenever it is computed; a p-value is not, which
  is the subject of the peeking simulations. A Bayesian readout under optional stopping still
  shifts P(better) across the threshold more often than the posterior suggests, so the rule here
  is applied once, to the final snapshot.

**How it is reported.** Both readouts are written in full, the frequentist one is the
pre-registered headline for Q2 (plan §6), and the Bayesian one carries the decision rule for
Q1 (plan §9). Where they agree, the README says so with both numbers; where they would not,
the reason above is the explanation to give.

## Sequential testing: peeking and its valid fixes (simulation)

`abkit.sequential`; readout `abkit.sequential.run_sequential`, outputs
`reports/tables/sequential.csv` (one row per scenario, metric and rule),
`reports/tables/sequential_looks.csv` (A/A false positive rate against the number of looks),
`reports/tables/sequential_by_day.csv` (cumulative rejection by day, and the single-look rate of
the z-test on each day alone), `reports/tables/sequential_boundaries.csv`, the `sequential`
section of `reports/results.json` and the figures `sequential_fpr_vs_looks.png`,
`sequential_cumulative_rejection.png` and `sequential_boundaries.png`.

**This is a simulation, not a re-analysis.** The real experiment is a single final snapshot and
is read once (analysis plan §12). Everything in this section is generated data built on the
control arm's baselines: its visit and conversion rates, and its spend values resampled with
replacement. No decision about the real experiment is taken from it. The one place the real
outcome data enters is the size of the planted effect in one A/B scenario, the observed visit
lift of the better email, used only to say what the rules would do to an effect of that size.

**The setting.** A stylised rollout: the campaign runs for 14 days, each day the same number of
new customers joins each arm (the smallest real arm divided by 14, the remainder dropped), and
the analyst looks at the cumulative results every day. Arm A is the control baseline, arm B the
email. The A/A scenarios plant no effect and measure how often each rule declares a winner that
does not exist; the A/B scenarios plant a visit lift and measure power and the stopping day.
The looks curve repeats the A/A with K equally spaced looks over the same customers, for K from
1 to 30. Visit is the headline metric; conversion and spend are run under A/A as well.

**The rules compared**, each stopping at the first look where it rejects:
1. *Fixed horizon*: one two-sided test at level alpha on day 14, the benchmark. For a rate it
   is the pooled two-proportion z-test of the readout; for spend the Welch z-test.
2. *Naive peeking*: the same test at the same level at every daily look.
3. *Bayesian peeking*: Beta(1, 1) priors on each arm's rate; stop when P(p_B > p_A) leaves the
   interval (0.05, 0.95). The probability is exact: a Gauss–Legendre quadrature of
   ∫ f_B(x) F_A(x) dx, checked against the closed form of the Bayesian readout, and applied
   through a boundary on counts (for each count in A, the smallest count in B that crosses the
   threshold; the mirror rule handles the lower threshold). For spend the posterior of the mean
   difference is the large-sample normal with a flat prior, so P(B > A) = Φ(z) and the rule is
   |z| > 1.645, which is naive peeking at alpha = 0.10. The rate rules are numerically close to
   that too: a flat prior adds nothing, and peeking at a posterior is peeking.
4. *Pocock*: reject at look k when |Z_k| > c_P(K), the same boundary at every look.
5. *O'Brien–Fleming*: reject when |Z_k| > c_B(K) sqrt(K / k): very strict early, 1.96-like at
   the end.
6. *mSPRT* (Johari, Pekelis and Walsh 2017): the mixture likelihood ratio of a normal estimate
   d with variance V against a N(0, tau²) prior on the effect,
   Λ = sqrt(V / (V + tau²)) exp(tau² d² / (2 V (V + tau²))), and the always-valid p-value
   p_n = min over looks so far of min(1, 1 / Λ). Reject when p_n ≤ alpha. The p-value is valid
   at every look and under any stopping rule because Λ is a non-negative martingale under the
   null, so P(sup Λ ≥ 1 / alpha) ≤ alpha (Ville's inequality). V is the squared standard error
   of the fixed-horizon test at that look.

**Calibrating the group-sequential constants.** With equal increments, the z-statistics at
K looks satisfy Z_k = S_k / sqrt(k) with S a cumulative sum of independent standard normals,
the Brownian-motion correlation of a sequence of two-sample tests. The Pocock constant is the
(1 − alpha) quantile of max_k |Z_k| and the O'Brien–Fleming final boundary is the (1 − alpha)
quantile of max_k |S_k| / sqrt(K) (since |Z_k| > c sqrt(K / k) is |S_k| / sqrt(K) > c), both
read off `n_paths` simulated null paths, with the Monte Carlo standard error of the quantile.
There is no root finding: the constants are order statistics. They are calibrated for every K
from 1 to 30; for K = 1 both reduce to the normal quantile 1.96.

**The mSPRT mixing scale.** Tau is the standard deviation of the prior on the effect, and the
test is most powerful against effects of about that size. It is set per metric to the metric's
minimum detectable effect at 80% power and the full sample (`abkit.power.mde`, the project's own
"plausible effect"), computed from the control arm's variance and the customers per arm. Half
and double that tau are reported as a sensitivity; both move the A/A rate and the power only a
little.

**Monte Carlo error.** Every rate carries sqrt(p (1 − p) / runs) and every mean stopping day its
standard error. The stopping day is reported with runs that never reject stopping at day 14,
and separately among the runs that reject.

**What the tables show.** Naive peeking turns the nominal 5% into a false positive rate several
times larger with daily looks, and the rate keeps growing with the number of looks (it would
tend to 1 with infinitely many looks). Bayesian peeking under a flat prior is worse, as the
alpha = 0.10 equivalence predicts. Pocock, O'Brien–Fleming and mSPRT keep the A/A rate at or
below 5% for every number of looks. They pay in different coins: Pocock gives up the most power
at the horizon because its final boundary is far above 1.96; O'Brien–Fleming costs almost no
power at day 14 but rarely stops early; mSPRT is conservative under the null (its rate is well
under alpha) and has markedly lower power at the MDE than the fixed-horizon test, the price of
a p-value that stays valid at every look. At the observed visit lift of the better email, an
effect of several standard errors on the first day alone, every rule stops on day 1 or 2 with
power 1, which is why that scenario says nothing about the rules and the MDE scenario was added.

**Small samples at the early looks.** With one fourteenth of an arm arriving per day, the
first daily look holds a handful of converters and spenders per arm, and the normal
approximation behind the z-tests and the boundaries can fail there. The check, in
`results.json` under `sequential.small_sample_check` and in the `single_look` rows of
`sequential_by_day.csv`, is the fixed-horizon test applied once: at the full sample (K = 1) and
at the first daily look alone. At the full sample all three metrics reject at the nominal rate
within Monte Carlo error. At the first daily look visit is at the nominal rate; conversion is
slightly below it, within about two Monte Carlo standard errors; spend is clearly below it: the
Welch test on a zero-inflated, heavy-tailed outcome with a handful of positive values per arm is
conservative, because a large spend inflates the variance estimate more than the mean. The
single-look rate for spend climbs back to the nominal level within the first week. So the only
deviation from 5% that is not peeking is a conservative one, at the earliest looks of spend
(and marginally conversion), and it has the opposite sign to the peeking inflation. Its visible
effect is on Pocock for spend, whose constant boundary gives the early looks weight: its A/A
rate sits a little below 5%, and further below as the number of looks grows and the early looks
get smaller. The inflation from naive and Bayesian peeking on conversion and spend is of the
same size as on visit and is entirely peeking.

**Assumptions.** Equal arm sizes and equal daily batches; customers arrive independently with
the control arm's outcome distribution; the Z-statistics follow the Brownian-motion
correlation (true asymptotically for any sum-based statistic, and the calibration test checks
the constants against the published values for that structure); the mSPRT treats V as known.
The variance for the planted visit effect is the treatment arm's own, so the fixed-horizon power
at the MDE is slightly below 80%, the same optimism noted for the MDE itself.

**How it is tested.** The fixed-horizon A/A false positive rate is within three Monte Carlo
standard errors of 5%; the calibrated constants for K = 5 match the published values (Pocock
2.413, O'Brien–Fleming 2.040; Jennison and Turnbull 2000, Table 2.1) within 0.02, and for K = 1
both equal 1.96; Pocock, O'Brien–Fleming and mSPRT keep the A/A rate at or below 5% within
Monte Carlo error; naive peeking with 14 looks is clearly above 5% (above 10%, beyond three
standard errors). The quadrature matches the closed-form P(p_B > p_A) to 1e-8, the count
boundary equals a brute-force search, and the Bayesian rule on simulated paths equals the
cell-by-cell probabilities; the spend Bayesian rule equals naive peeking at alpha = 0.10 exactly.
The mSPRT likelihood ratio matches the hand formula, its p-values are running minima in (0, 1],
and the implied |z| boundary inverts the ratio. The spend simulator's sums and sums of squares
have the resampled values' moments, and the z-statistics equal the readout's tests on the same
counts. The fixed-horizon power at a planted visit lift matches the closed form for the pooled
z-test with unequal variances within three Monte Carlo standard errors. The pipeline test runs
the whole study in a fast mode (fewer runs and paths) on the fixture and checks every table,
figure and the results.json section; the full sizes are used only by the pipeline.

## Segment analysis (exploratory)

`abkit.segments`; readout `abkit.segments.run_segments`, outputs `reports/tables/segments.csv`
(one row per outcome, segment variable, level and email vs control),
`reports/tables/segments_interactions.csv` (one joint test per outcome, segment variable and
email), `reports/tables/segments_fdr_simulation.csv`, the `segments` section of
`reports/results.json` and `reports/figures/forest_segments.png`. Every row carries the label
`exploratory`.

**What it does.** Asks whether the emails worked differently for different kinds of customer,
on the seven pre-specified segment variables of analysis plan §10 (new customer, channel, zip
code, past men's buyer, past women's buyer, recency band and history band: 23 levels in all)
and never on any other. Segments are customer subgroups defined by pre-treatment variables
only; the treatment arm is never a segment. Results are exploratory: they generate hypotheses
for a future test and are not decisions, so the README reports them as such and the decision
rests on the pre-registered readouts.

**Per-level effects.** Within each level, each email vs control: spend by the Welch difference
in means with its interval; visit and conversion by the difference in proportions with the
pooled two-proportion z-test and the Newcombe interval, exactly the readout's tests (§6)
applied to the level's customers. The sums behind them come from `sql/segment_stats.sql`.

**Interaction tests.** For each outcome, segment variable and email, OLS on the customers of
that email arm and control: Y = a + τ D + Σ_l β_l 1{level = l} + Σ_l γ_l D · 1{level = l} +
error, with the first level as reference. τ is the effect in the reference level and γ_l is
how much the effect in level l differs from it; the within-level effects τ + γ_l are the raw
differences in means of the level (the model is saturated). The question "does the effect
differ across levels?" is the joint hypothesis γ = 0, tested with a Wald chi-square statistic
γ' V_γ⁻¹ γ on L − 1 degrees of freedom, where V is the HC2 heteroskedasticity-robust covariance
(X'X)⁻¹ X' diag(e_i² / (1 − h_ii)) X (X'X)⁻¹, computed in numpy and checked against
statsmodels. For visit and conversion this is a linear probability model, which is the right
model for differences in rates. One test per variable answers the heterogeneity question
directly, whereas comparing 23 separate p-values does not: two levels can both be significant
with the same effect, or one significant and one not with effects that do not differ.

**Multiple testing.** Benjamini–Hochberg at q = 0.05 over each outcome's whole segment family:
the 46 per-level tests and the 14 interaction tests together, 60 per outcome, as pre-specified.
Raw and adjusted p-values are reported side by side with the family size, and
`significant_fdr` flags adjusted p ≤ q. For each family `results.json` also records how many
of its tests are significant before correction (raw p ≤ 0.05, `n_significant_raw`) next to
the count after it (`n_significant_fdr`), so the cost of the correction is visible; an
adjusted p-value is never below its raw one, so the first count is at least the second, which
is tested together with a hand-worked family of six p-values. The Hillstrom challenge's own question, whether the
women's email worked better among past women's buyers and the men's email among past men's
buyers, is answered from the pre-specified `mens` and `womens` rows (the effect among past
buyers, among non-buyers and the interaction test), inside the family, with no extra test.

**Deviation D5.** The plan names spend and visit as the segment outcomes. Conversion is added
as a third outcome with its own family of 60 tests, so the spend and visit families are
unchanged. Recorded in the plan's Deviations section.

**The README simulation.** The reason the family matters, in one A/A experiment: 20
independent segments, each an A/B comparison of a visit-like rate with the real arm size
split twenty ways and the control arm's visit rate as baseline. Each segment gets the z-test;
then the run's 20 p-values are Benjamini–Hochberg adjusted. With no true effect anywhere, raw
testing finds at least one "significant" segment in a share of runs equal to 1 − 0.95²⁰
(about 64%), while Benjamini–Hochberg finds one in a share equal to q. A second scenario
plants a real lift in 5 of the 20 segments (the lift one segment's test detects with 80%
power): there the realised false discovery rate of raw testing is several times q, and
Benjamini–Hochberg's equals q · 15 / 20, its exact value under independence (Benjamini and
Hochberg 1995), at the cost of lower power on the real effects. Every rate carries its Monte
Carlo standard error next to the analytic reference.

**Assumptions.** Randomisation holds within every segment because the segment variables are
pre-treatment, so each within-level difference is an unbiased effect estimate. The interaction
test's HC2 covariance is valid under heteroskedasticity across levels and arms; the chi-square
reference is asymptotic and the smallest levels (the top history band) hold a few hundred
customers per arm. For spend, the Welch interval in a small level with a handful of converters
is wide and only approximately normal, as the readout section discusses. Benjamini–Hochberg
controls the false discovery rate under independence or positive dependence; the per-level and
interaction tests of one outcome are positively dependent, which the procedure tolerates.

**How it is tested.** The OLS coefficients and HC2 covariance match statsmodels
`cov_type="HC2"` to 1e-8; the joint statistic and p-value match the formula
`y ~ d * C(seg)` and its `wald_test` to 1e-6, with planted heterogeneity found at p < 1e-6;
the within-level effects equal the raw differences in means; with a binary segment the
statistic equals the squared interaction t-ratio; wrong inputs (one level, one arm, a level
missing from an arm) raise. The segment columns reproduce the SQL counts level by level and
reject outcome columns. Per-level rows equal the readout's `diff_in_means` and
`diff_in_proportions` on the level's customers. The adjusted p-values equal
`benjamini_hochberg` on the concatenated 60 p-values of each outcome. The row-wise BH used by
the simulation equals the scalar one row by row. The simulation reproduces 1 − 0.95²⁰ and q
under the global null, and q · 15 / 20 with five real effects, within three Monte Carlo standard
errors. The pipeline test on the fixture checks every table, the figure and the results.json
section, and that every row is labelled exploratory.

## Uplift feature matrix and the leakage guard

**What it does.** Builds the feature matrix for the uplift models from the pre-treatment
columns of the cleaned `customers` table and nothing else. `recency` and `history` stay
numeric; the 0/1 flags `mens`, `womens` and `newbie` are kept as they are; `zip_code` and
`channel` are one-hot encoded with every level from `config`, so the columns are the same
whichever subset of customers is passed in. `history_segment` is dropped because it is
`history` binned into seven dollar bands (analysis plan D6).

**Guard.** `assert_pre_treatment_only` raises if any feature name is, or starts with, an
outcome (`visit`, `conversion`, `spend`) or the treatment assignment (`arm`, raw `segment`),
which also catches derived columns such as `arm_mens`. `build_features` runs it on its own
output and `cross_fit` runs it on any feature frame it receives, so an outcome cannot reach
a model without raising.

**Tests.** The matrix built from the fixture has exactly the expected columns, no
`history_segment`, one-hot rows that sum to one, and numeric columns equal to the input. The
guard is tested on every forbidden name, on derived names, and on the allowed names.

## Treatment pairs and cross-fitting

**What it does.** Each email is modelled against control as its own binary problem: the
`mens` pair and the `womens` pair. Inside a pair the treatment probability is known by
design, so it is the realised share of the email arm, `p_treat = n_email / (n_email +
n_control)`, never a fitted propensity.

**Cross-fitting.** Customers are split into five folds, stratified jointly by arm and the
binary outcome (four cells), with scikit-learn's `StratifiedKFold` seeded from
`config.SEEDS["uplift"]`. For each fold the learner is fitted on the other four folds and
scores the held-out fold, so every customer gets exactly one out-of-fold score from models
that never saw that customer. `cross_fit` takes any `fit_predict(X_train, treated_train,
y_train, X_test)` callable, so the T-learner, X-learner and the response-model baseline share
the same folds and the same loop.

**Assumptions.** Randomisation inside each pair, which the SRM and balance checks support.
Stratifying on the outcome only balances the folds; the outcome is never a feature.

**Tests.** `p_treat` equals the arm count ratio; every (arm, outcome) cell is spread across
folds to within one customer; the same seed gives the same folds. With a `fit_predict` that
returns the mean outcome of its training rows, each fold's scores equal the mean over the
other folds exactly, which shows that no row scores itself.

## Uplift and Qini curves, their areas, uplift at k and calibration

**What it does.** Evaluates a ranking of customers by predicted uplift on out-of-fold
scores. Customers are sorted by descending score; the curves are evaluated at every
distinct score value, so tied customers enter together and the result does not depend on
row order. With cumulative counts over the top share of customers `n`, of which `n_t`
treated with `y_t` responders and `n_c` controls with `y_c` responders:

- **Uplift curve:** `(y_t / n_t − y_c / n_c) · n`, the incremental responses expected if the
  whole top share were treated.
- **Qini curve** (Radcliffe 2007): `y_t − y_c · n_t / n_c`, the incremental responses among
  the treated customers in the top share.

A rate whose group is still empty counts as zero, as in scikit-uplift.

**Areas.** Each area is the trapezoid integral over the share axis, so it is in incremental
responses averaged over targeting depths. Three numbers are reported per curve:

- `model`, the area under the curve;
- `gain = model − random`, where the random curve is the straight line from the origin to
  the curve's final point (the incremental responses random targeting finds at each depth);
- `normalised = (model − random) / (perfect − random)`, 0 for random targeting and 1 for the
  perfect ranking. The perfect ranking sorts by the observed outcomes exactly as scikit-uplift
  does (for Qini: treated responders, then everyone else, then control responders), so the
  normalised AUUC and Qini coefficient equal scikit-uplift's `uplift_auc_score` and
  `qini_auc_score`.

**Uplift at k.** The observed treated-minus-control response rate among the top `floor(k·n)`
customers by score (ties broken by row order).

**Calibration by decile.** Customers are split into ten equal-count bins of the predicted
score (decile 1 = highest). Each bin reports the mean predicted uplift and the observed
treated-minus-control rate with a Newcombe interval; a calibrated model lies on the diagonal.

**Bootstrap.** Customers are resampled with replacement (multinomial counts as weights) and
every metric is recomputed with the scores held fixed; bin membership is fixed from the
original scores. Percentile 95% intervals are reported with the Monte Carlo standard error
of each interval end. Because the scores are fixed, the intervals leave out model-fitting
variability (analysis plan §11).

**Tests.** A ten-customer example has its Qini and uplift curves, areas and uplift at k
worked out by hand. On synthetic data with a known individual effect, the oracle ranking
scores far above zero and random rankings average zero within Monte Carlo error; the perfect
ranking scores exactly 1. Resampling weights of one reproduce the unweighted values. A slow
test checks the bootstrap interval's coverage. When scikit-uplift is installed, the curves,
both normalised areas and uplift at k are compared with its implementation (skipped
otherwise).

## Uplift meta-learners: S, T and X

**What they do.** Each learner turns two outcome models into a per-customer uplift score
`tau(x)`, the predicted change in the response rate from sending the email. All use
scikit-learn's histogram gradient boosting (`HistGradientBoostingClassifier` for the binary
outcomes, `HistGradientBoostingRegressor` for the X-learner's second stage), with the 0/1
feature columns passed as categorical. Each email is modelled against control separately,
with the known treatment probability `g` = the realised share of the email arm in that pair.

- **S-learner:** one classifier on `(x, treated)`; `tau(x) = mu(x, 1) − mu(x, 0)`.
- **T-learner:** `mu_1` fitted on treated rows, `mu_0` on control rows;
  `tau(x) = mu_1(x) − mu_0(x)`.
- **X-learner** (Künzel et al., 2019): the T-learner models impute an individual effect for
  every training row, `D_i = y_i − mu_0(x_i)` if treated and `D_i = mu_1(x_i) − y_i` if
  control. Two regressors fit `tau_1(x)` on the treated rows' `D` and `tau_0(x)` on the
  control rows' `D`, and the score is `tau(x) = g tau_0(x) + (1 − g) tau_1(x)`. With the
  treated share `g` near one half the two estimates are weighted about equally.

**Hyperparameters.** A fixed grid of four points (learning rate 0.05 or 0.1; 15 or 31 leaves)
with early stopping (up to 500 iterations, stop after 10 without improvement on an internal
10% validation set). Inside each training fold, 20% is held out as an inner validation split;
the grid point with the lowest log loss (classifiers) or squared error (regressors) on that
split is refitted on the whole training fold. The evaluation fold is never used for any
choice, and nothing is tuned against the uplift metrics (analysis plan D7).

**Assumptions.** Randomisation within each pair, so the outcome models' difference is a
causal effect; enough signal in the pre-treatment variables for heterogeneity to be
learnable. The outcomes are rare (visits about 15%, conversions about 1%), so the conversion
scores are expected to be noisy.

**Tests.** On synthetic data with a known heterogeneous effect, the T- and X-learners' scores
have a clearly positive rank correlation with the true effect and beat random on the Qini
coefficient. A step-function effect that depends on one feature is recovered by all three
learners. The learners never see the evaluation fold: with a recording `fit_predict` every
training set is disjoint from its test set. The response model's score equals the treated
outcome model's probability.

## Targeting baselines: likely to respond versus persuadable

The uplift models are compared with three rankings that need no causal model:

- **Random targeting:** uniform random scores. Its Qini curve is the straight line to the
  overall effect; the normalised coefficient is 0 on average.
- **Recency rule:** most recent purchasers first (ties by higher `history`, then customer
  id). This is the R in RFM targeting and the strongest single rule in direct marketing.
- **Response model:** a classifier for `P(visit | emailed, x)` fitted on the emailed rows,
  ranking customers by how likely they are to visit *if emailed*.

The response model targets the **likely to respond**: customers with a high visit rate when
emailed. Many of them would have visited anyway, so emailing them adds little. An uplift
model targets the **persuadable**: customers whose visit rate the email raises the most,
`P(visit | emailed, x) − P(visit | not emailed, x)`. The two rankings agree only when the
baseline rate is flat across customers. When the baseline varies, the response model spends
budget on sure things and misses customers with a low baseline but a large lift; it can even
rank "do not disturb" customers (negative effect) at the top if their response rate under
email is still high. The Qini curve shows the difference directly: a ranking that chases
responders rather than lift flattens toward the random line.

The plan's fourth baseline, "A/B winner at random", differs from random targeting only in
which email is sent, so within one email-vs-control pair it is the random ranking; it is
evaluated as a policy in the budgeted targeting stage (analysis plan D7).

## Targeting policies and their value (inverse propensity weighting)

**What it does.** Answers the challenge's targeting questions. A policy `π` assigns every
customer an arm: the mens email, the womens email or no email. The experiment estimates
the value of any such policy without a model of the outcome, because each customer's arm
was drawn at random with known probability `p = 1/3`:

- **IPW:** `V(π) = (1/n) Σ_i Y_i · 1{A_i = π(x_i)} / p`.
- **Self-normalised IPW:** `V_sn(π) = Σ_i Y_i · 1{A_i = π(x_i)} / p ÷ Σ_i 1{A_i = π(x_i)} / p`,
  which replaces `n` by the realised weight and removes the small-sample noise of the
  arm shares.

Both are consistent because every policy is a fixed function of pre-treatment variables
and out-of-fold scores, so whether a customer's realised arm matches the policy is
independent of the outcome given `x`. Values are reported as increments over emailing
nobody: in total over the 64,000 customers, per email sent, and as a share of the
incremental effect of emailing everyone with the A/B winner. A cost per email
(`config.COST_PER_EMAIL`, default 0 because the challenge gives no cost) turns incremental
spend into a net value.

**Policies.** Email nobody; everyone gets the mens email; everyone gets the womens email;
a random 10,000 get the A/B winner; the recency rule sends the A/B winner to the 10,000
most recent purchasers; the response model emails the 10,000 most likely responders with
the email under which they are more likely to visit; the uplift policy gives each customer
the email with the higher predicted uplift and emails the 10,000 with the highest such
uplift, if positive; the exploratory mens-by-default policy holds the mens uplift at its
average effect and uses the womens model only where it predicts more; and the suppression
policy emails everyone with their better email except the 10,000 lowest predicted uplifts
(the do-not-email list). For that list the realised uplift of their better email is
estimated within the group by IPW, to check whether it is at or below zero.

**Scores.** The X-learner's visit scores are used for both emails (analysis plan D8). For
the mens email no ranking beat random, so no "best" mens learner is chosen by Qini: that
would select on noise. Because the uplift policy compares predicted levels across the two
models, the mean predicted uplift in each pair is checked against the observed average
effect before use, alongside the decile calibration plots. The scores were trained on
visits; the conversion and spend values therefore measure how well a visit-based targeting
carries over to those outcomes, not how well a conversion or spend model would do.

**Budget curve.** For the uplift policy, the response model and random targeting, the
incremental visits and spend are traced against the number of customers emailed, from 0 to
everyone in steps of 1,000, each ranking sending its own choice of email.

**Uncertainty and comparisons.** A customer-level bootstrap (1,000 resamples, the same
resampled customers for every policy) gives percentile intervals with the Monte Carlo
standard error of each end. Policies are compared by the paired bootstrap distribution of
their difference, never by whether two intervals overlap. The out-of-fold scores are held
fixed, so the intervals leave out model-fitting variability.

**Tests.** On synthetic potential outcomes with known means, IPW and self-normalised IPW
recover the closed-form value of a fixed policy within Monte Carlo error. Every budgeted
policy emails at most the budget, the suppression policy withholds exactly the budget, the
uplift policy emails the highest positive predicted uplifts with each customer's better
email, and identical policies give a paired difference of exactly zero. The stage runs end
to end on the synthetic fixture.

**What the experiment says about the policies.** Read from `reports/tables/policy.csv`,
`policy_comparison.csv` and the `policy` section of `results.json`; the numbers live there,
not here.

- The level-calibration check passes for both emails: the mean predicted uplift lies inside
  the Newcombe interval of the observed average effect, so the two models' levels can be
  compared.
- The uplift policy is **not clearly better than the recency rule** on visits, conversions
  or spend: every paired bootstrap difference between the two includes zero. Its spend point
  estimate is well above the rule's, but the interval is wide because spend is almost all
  zeros. The response model is nominally ahead of the uplift policy on visits, again without
  a meaningful paired difference. The only paired difference that excludes zero among the
  budgeted policies is uplift versus random targeting on conversions.
- The reason is visible in the uplift-model section: the mens email, which the A/B winner
  policies send, lifts visits for nearly everyone with no detectable heterogeneity, so any
  rule that sends it to 10,000 customers captures about the same number of incremental
  visits. The uplift model's advantage is confined to choosing *which* customers get the
  womens email, a minority of the budget.
- The suppression list (the 10,000 lowest predicted uplifts) does **not** identify
  customers who should not be emailed: their realised uplift under their better email is
  positive, with an interval that excludes zero. Withholding their email costs visits. The
  honest answer to "whom should we not email" is that this experiment finds no such group.
- The exploratory mens-by-default policy does not improve on the uplift policy.
- The budget curve shows the same thing at every budget: on visits the three rankings are
  close, with random targeting a little below the two models; on spend the uplift ranking
  is nominally ahead at small budgets, within noise. The curves meet at 64,000 only up to
  the choice of email, because each ranking sends its own choice.

## Repeated cross-fitting over fold seeds

**What it does.** Every bootstrap interval in the uplift-model and policy sections holds
the fold assignment fixed: it resamples customers with their out-of-fold scores as given.
That leaves out the variability from which customers land in which fold and from the base
models' own seeds. Chernozhukov et al. (2018) recommend repeating cross-fitting over
several random splits and reporting the spread (or the median) of the resulting
estimates. This stage re-runs the cross-fitting for the primary outcome (visit) and both
email pairs under ten fold seeds: seed 0 is the committed run, read from the scores the
uplift-model stage wrote, so its rows reproduce the headline tables exactly; seeds 1 to 9
are child generators spawned from the configured uplift seed. For each seed it records the
normalised Qini coefficient of the S-, T- and X-learners and the response model on each
pair, whether the X-learner beats the response model on the womens pair, and the uplift
policy's incremental visits at the 10,000-email budget. The committed headline tables are
left as they are; the spread is reported next to them in
`reports/tables/uplift_seed_robustness.csv` and the `seed_robustness` section of
`results.json`.

**Tests.** The seed generators are deterministic, distinct and leave the parent stream
untouched. The summary (median, min, max, win count) is checked on a toy table with a
known answer. The stage runs end to end with a fast stand-in scorer, and (slow) with the
real learners on the fixture.

**What the repeated cross-fitting shows.** Read from `reports/tables/uplift_seed_robustness.csv`
and the `seed_robustness` section of `results.json`.

- On the womens pair the ranking of the learners is stable: the X-learner beats the
  response model in every seed, and all three uplift learners stay well above it. The
  choice of the X-learner for the policy does not hinge on the committed seed.
- On the mens pair every uplift learner sits at zero in every seed, while the response
  model keeps a small positive coefficient. The conclusion that the mens effect has no
  detectable heterogeneity is not a seed artefact.
- The uplift policy's incremental visits at the budget move across seeds by an amount
  comparable to the width of the paired bootstrap interval, and the committed seed is at
  the low end of the range. The bootstrap intervals therefore understate the total
  uncertainty: fold-assignment and model-seed variability add about as much again. The
  headline verdict is unchanged, in that no seed puts the uplift policy outside the
  sampling uncertainty of the recency rule, but a reader should take the seed-to-seed
  range as the honest spread of the policy's value and the median across seeds, not the
  committed seed, as its central estimate.

## Pipeline: stages, cache and fast mode

**What it does.** `python -m abkit.pipeline run-all` runs every stage in dependency order,
from the download to the validated `results.json`: data, validate, checks, frequentist,
cuped, bayes, power, sequential, segments, uplift, policy, then report. The seed-robustness
stage is left out unless `--with-robustness` is given, because it re-fits the cross-fitting
nine more times. Each stage is cached: a fingerprint of its input files (the cleaned
`customers` parquet, the uplift scores), the `results.json` sections it reads, the source
files it depends on, the package version and its settings is stored in
`data/pipeline_cache.json` with the runtime, and the stage is skipped when the fingerprint
is unchanged and every output file and `results.json` section it writes still exists.
`--force` ignores the cache. The report stage is never cached: it writes the `pipeline`
section (mode, data, stages, sizes) and validates the whole file against the schema in
`src/abkit/results_schema.py`, which also renders `docs/results_schema.md`.

**Fast mode.** `--fast` replaces the simulation, bootstrap and seed counts with
`config.FAST_SIZES`, models visit only in the uplift stage and searches a single
hyperparameter point. Its numbers are for checking that the pipeline runs, not for the
memo: the `pipeline` section of `results.json` records which sizes were used, so a fast run
can never be mistaken for the full one. `--data` runs the same pipeline on another CSV,
such as the synthetic fixture, with the row-count check off and the email budget scaled
to the rows; `--out` moves every output under another directory so a fixture run never
overwrites the real artifacts.

**Figures.** Every figure's title is a sentence computed from the table behind it at plot
time (counts of significant rows, the quoted probabilities and rates), never typed by hand,
so a title can only state what the artifact says. One palette (the reference categorical
slots for the arms, validated for colour-vision deficiency, with neutral inks and line
styles once a figure has more than three series) and one type style are set in
`abkit.plots`.

**Tests.** The fast test suite runs `run-all --fast` on the fixture end to end, checks that
every stage output exists, that `results.json` validates, that the budget scaled with the
rows, that a second run skips every cached stage, and that removing an output or passing
`--force` reruns only what it should. The schema tests check the committed `results.json`
and that the rendered document matches the schema.

## The app and its live peeking simulator

**What it does.** The Streamlit app in `app/` is a reader of the committed artifacts. Its
Home, Readout, Targeting and Method pages show values from `reports/results.json` and the
CSV tables that file names, and nothing else: the app never opens the raw data or the
database, so it can be deployed from the repository alone. Lookups go through
`abkit.appdata.find_row`, which fails unless exactly one record matches, so a page cannot
show an ambiguous or stale number.

The Peeking simulator page is the one place where numbers are computed on screen. It
simulates an A/B test on a rate that is checked once a day: the user sets the baseline rate,
the true relative lift, the users per day (split evenly between two arms), the number of
days and the stopping rule. It reuses the pipeline's own code (`abkit.sequential`): the same
simulated experiments, the same calibration of the Pocock and O'Brien–Fleming constants, the
same six rules and the same summary. The mSPRT mixing scale is the minimum detectable effect
of the fixed-horizon design at the chosen size, as in the pipeline.

**Key quantities.** The share of simulated experiments that declare a winner is the false
positive rate when the lift is zero and the power otherwise; it is shown with its Monte
Carlo standard error sqrt(r (1 − r) / runs). The stopping-day chart is the share of
experiments that stop with a declared winner on each day, plus the share that run to the
end without one.

**What differs from the pipeline's simulation.** Only the sizes, which are capped so a
change of input answers in about two seconds: fewer simulated experiments and fewer null
paths for the boundary constants than the study in the sequential section, so its Monte
Carlo error is larger and is always printed next to the result. The Bayesian rule
evaluates P(B > A) for each run and day directly instead of through the count boundary
the pipeline uses; this is the same rule computed another way, cheaper for a few thousand
runs because its cost does not grow with the sample size, and its runs are capped further
because each evaluation is a numerical integral. The seed is the configured simulation
seed, so the same inputs always give the same answer.

**Assumptions.** Those of the sequential section: equal arm sizes, equal daily batches,
independent customers and a binary outcome.

**How it is tested.** Streamlit's `AppTest` runs every page and fails on any exception. The
Readout page's headline effect is compared with `results.json` for every comparison and
metric, the Home page's three numbers and decision with the sections they come from, and
the Targeting page's table with the budget-curve row at several budgets. Every link on the
Method page must point at a heading that exists in this document. For the simulator: the
fixed-horizon false positive rate is within three Monte Carlo standard errors of 5% and
naive peeking is more than five above it; the fixed-horizon power at a planted lift matches
the closed-form power; for each of the six rules the wrapper's result equals the
pipeline's rule applied to the same simulated experiments (which also checks the
direct Bayesian evaluation against the count boundary, tested separately on four
designs); and the largest allowed inputs answer within the time limit.
