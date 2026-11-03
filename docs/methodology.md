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
