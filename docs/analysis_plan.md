# Analysis Plan

Pre-registered plan for the Hillstrom email experiment. Written and committed before any outcome
is compared between arms. Once committed, this plan is frozen: any change is recorded in
[Deviations](#13-deviations) with its reason, and the original text stays as it is.

**What had been seen when this plan was written.** Only the data validation report
(`reports/validation.md`): schema, ranges, category labels, arm sizes, duplicate counts, and
outcome distributions pooled across all arms. No outcome had been broken down by arm.

## 1. Business questions

| # | Question | Analysis that answers it |
|---|---|---|
| Q1 | Which email performed better, and on which metric (visit, conversion, spend)? | Each email vs control on spend (primary, §4, §6), conversion and visit (secondary); mens vs womens on all three metrics (secondary comparison, §3); Bayesian readout (§9). |
| Q2 | How much incremental spend per customer did each email drive? | Difference in mean spend per customer, each email vs control, with 95% CI (§6); CUPED and regression-adjusted estimates (§7); winsorised sensitivity analysis (§8); posterior and credible interval (§9). |
| Q3 | If we could email only 10,000 customers, who should get which email? | Uplift models for each email vs control, out-of-fold scores, a policy that assigns each of the top 10,000 customers the email with the higher predicted uplift, evaluated by policy value against four baselines: random targeting, the A/B winner at random, a response model and a recency rule (§11). |
| Q4 | Which 10,000 customers should we NOT email? | The 10,000 customers with the lowest predicted uplift under the better of the two emails, checking whether their estimated realised uplift is at or below zero (§11). |
| Q5 | Do the effects differ across customer segments? | Pre-specified, exploratory segment analysis of spend and visit, with interaction tests and per-segment effects (§10). |

## 2. Design

- **Unit of randomisation and analysis:** the customer (one row per customer).
- **Population:** customers who purchased in the last twelve months.
- **Arms:** `mens` (Mens E-Mail), `womens` (Womens E-Mail), `control` (No E-Mail).
- **Randomisation:** simple randomisation with equal probability 1/3 per arm, as described by the
  source. These known assignment probabilities are used for inverse propensity weighting in §11.
- **Outcome window:** the two weeks after the campaign.
- **Sample sizes** (source: `reports/validation.json`, `categories.segment`):

| Arm | Customers |
|---|---:|
| mens | 21,307 |
| womens | 21,387 |
| control | 21,306 |
| **Total** | **64,000** |

- **Analysis data:** the cleaned table `customers` (all 64,000 rows; no rows dropped; duplicates
  kept, see the validation report).

## 3. Comparisons

- **Primary family:** `mens` vs `control` and `womens` vs `control`.
- **Secondary:** `mens` vs `womens`.

Differences are always defined as the first arm minus the second, so a positive difference
favours the email.

## 4. Metrics

- **Primary: spend per customer.** Mean of `spend` over all customers in the arm, with zeros
  included. The business question is incremental sales, and the mean over everyone (not only
  buyers) is the quantity that scales to revenue per email sent.
- **Secondary: conversion rate** (share with `conversion = 1`) and **visit rate** (share with
  `visit = 1`).
- **No guardrail metric.** The data has no unsubscribes, spam complaints or other harm
  measures, so a "win" cannot be checked against a cost of emailing. This is a limitation of every
  conclusion here, and in particular of Q4.

## 5. Validity checks (run before any outcome comparison)

1. **Sample ratio mismatch (SRM).** Chi-square goodness-of-fit test of the three arm counts
   against equal thirds (2 degrees of freedom). **p < 0.001 fails the check and blocks the
   readout**: no treatment effect is reported until the cause is understood.
2. **Covariate balance.** Standardised mean difference (SMD) for every pre-treatment covariate,
   for each email vs control and for mens vs womens. Categorical covariates are expanded into one
   indicator per level. SMD = (mean_a − mean_b) / sqrt((var_a + var_b) / 2). **|SMD| > 0.1 is
   flagged.** A flag does not block the readout, but it is reported, and the regression-adjusted
   estimate (§7) is reported next to the primary one.

## 6. Frequentist analysis

- **Tests:** two-sided, alpha = 0.05 (`config.ALPHA`).
- **Spend (primary):** Welch's difference in means, with Welch–Satterthwaite degrees of freedom and
  a 95% CI. With about 21,000 customers per arm and a heavily zero-inflated outcome, Welch's test
  relies on the central limit theorem for the mean, not on normal spend.
- **Rates (secondary):** two-proportion z-test (pooled variance under H0) for the p-value, with a
  95% **Newcombe hybrid score interval** for the difference in proportions (Newcombe 1998,
  method 10).
- **Relative lift:** (mean_t − mean_c) / mean_c for every metric, with a 95% interval from the
  **delta method on the log ratio**, back-transformed:
  Var(log(mean_t / mean_c)) ≈ var_t / (n_t · mean_t²) + var_c / (n_c · mean_c²).
- **Multiple testing:**
  - **Holm** correction within the primary family (the two spend comparisons vs control). An
    email beats control on the primary metric if its Holm-adjusted p < 0.05 and the difference is
    positive.
  - **Benjamini–Hochberg** (q = 0.05) across the secondary family: conversion and visit, each
    email vs control (4 tests), plus mens vs womens on spend, conversion and visit (3 tests),
    for 7 tests in total.
  - **Benjamini–Hochberg** separately across the segment family (§10).
  - Confidence intervals are reported unadjusted at 95%; decisions use the adjusted p-values.
- **Headline estimate for Q2:** the unadjusted Welch difference in means. It is the simplest and
  most assumption-light estimate. The CUPED and regression-adjusted estimates (§7) and the
  winsorised estimate (§8) are reported next to it, never in its place.

## 7. Variance reduction

- **CUPED.** The pre-specified covariate is `history` (past-year spend in dollars), used as is
  with no transformation. Y_adj = Y − θ (X − mean(X)), where θ = cov(Y, X) / var(X) is estimated
  on all customers pooled across arms, and X is centred on the pooled mean. Because X is measured
  before treatment and θ is pooled, the adjusted difference in means stays unbiased for the same
  effect, and its variance falls by a factor of about (1 − ρ²), where ρ is the correlation of
  `history` with the outcome. CUPED is applied to all three metrics, and the variance reduction
  achieved is reported.
- **Regression adjustment (sensitivity).** Lin (2013) estimator: OLS of the outcome on the arm
  indicators, all pre-treatment covariates centred at their pooled means, and arm × covariate
  interactions, with HC2 robust standard errors. Covariates: `recency`, `history`,
  `history_segment` (indicators), `mens`, `womens`, `zip_code` (indicators), `newbie`, `channel`
  (indicators). This captures non-linear relationships with `history` that CUPED, being linear,
  misses.

## 8. Outliers and data features

- **Primary analysis uses raw spend.** No value is dropped, capped or transformed.
- **Top-coding in the source.** Among converters, the 99th percentile of spend equals its
  maximum, \$499.00, with a heap of converters exactly at that value (validation report). Spend
  therefore appears to be capped at \$499 in the source file. Effects on spend are effects on
  this top-coded spend, and may understate the effect on true spend if large orders respond to
  email. This is stated as a limitation.
- **Sensitivity analysis: winsorised spend.** Spend is capped at the **95th percentile of
  converters' spend, pooled across arms**. Customers with zero spend are unchanged. The
  percentile is taken over converters only: about 99% of customers spend nothing, so any
  percentile up to the 99th over all customers would be \$0 and would wipe out the signal. The
  cap is computed once on the pooled converters, never per arm. The primary (§6) and CUPED (§7)
  analyses are repeated on winsorised spend. **Raw and winsorised results are both reported,
  whatever they show.**
- **Point mass in `history`.** The most common `history` value is the minimum, \$29.99, and it
  accounts for about an eighth of all customers (validation report). It is possibly a single
  lowest-priced purchase or a floor value in the source. Consequences:
  - **CUPED:** the covariate is not continuous. A large block of customers shares one value, and
    the right tail is long (up to \$3,345.93). θ is a linear fit and can be driven by the tail.
    CUPED stays unbiased, but the variance reduction may be small. The Lin adjustment, which
    includes `history_segment` indicators, is the check on this.
  - **Segments:** the `usd_0_100` history band contains the whole cluster, so its per-segment
    effect largely reflects customers at the minimum. This is noted when the band is
    interpreted; no extra segment is added for it.

## 9. Bayesian analysis

- **Rates (conversion, visit): Beta-Binomial.** Each arm's rate has a Beta(α0, β0) prior and the
  conjugate Beta(α0 + successes, β0 + failures) posterior. Effects are summarised from 20,000
  posterior draws per arm.
- **Spend: Bayesian bootstrap** (Rubin 1981). For each draw, Dirichlet(1, …, 1) weights over the
  customers in each arm give a posterior draw of that arm's mean spend; 4,000 draws per arm.
- **Reported for each email vs control** (and for mens vs womens):
  - the posterior mean of the difference and its 95% equal-tailed credible interval;
  - **P(treatment better)** = P(θ_t > θ_c);
  - the **expected loss of shipping** = E[max(θ_c − θ_t, 0)], the average shortfall if the email
    is shipped and is in fact worse.

  Monte Carlo quantities are reported with their Monte Carlo standard errors. Draws use the seed
  in `config.SEEDS["bayes"]`.
- **Decision rule (primary metric, spend).** Ship an email if **P(better) ≥ 0.95 AND the
  expected loss < 1% of the control arm's mean spend per customer**. If both emails qualify, ship
  the one with the higher posterior mean spend, and report P(mens > womens) alongside.
  - Why this threshold: without cost or unsubscribe data, the natural yardstick is what a
    customer is worth over two weeks without an email. Accepting an expected downside of at most
    1% of that means the worst we expect to give up is small next to a typical email campaign
    lift, which is several percent or more. The threshold is relative, so it does not depend on
    the currency or the period length.
  - The control mean is computed at readout time. The rule fixes the threshold's definition now,
    not its value.
  - For the secondary rates, the same rule (1% of the control rate) is reported for information
    only.
- **Prior sensitivity (rates).**
  - The primary analysis uses a flat Beta(1, 1) prior.
  - Sensitivity analyses use a **shared informative prior**: the same Beta prior for every arm,
    with mean equal to the rate pooled across all arms and strength n0 = α0 + β0 ∈ {100, 1,000,
    10,000} pseudo-customers. Because it is shared, it pulls every arm toward the same value and
    so shrinks differences toward zero, which makes it sceptical. The pooled mean uses the data,
    but symmetrically across arms, so it cannot favour either email.
  - Results are reported for all four priors.
  - For spend, the Bayesian bootstrap has no tunable prior; its sensitivity check is the
    winsorised analysis (§8).

## 10. Segments (exploratory)

- **Segment variables** (pre-treatment only; levels fixed now):

| Segment variable | Levels |
|---|---|
| `newbie` | 0, 1 |
| `channel` | phone, web, multichannel |
| `zip_code` | urban, suburban, rural |
| Past men's buyer (`mens`) | 0, 1 |
| Past women's buyer (`womens`) | 0, 1 |
| Recency band (`recency`) | 1–3, 4–6, 7–9, 10–12 months |
| History band (`history_segment`) | the 7 source bands, `usd_0_100` … `usd_1000_plus` |

- **Outcomes:** spend per customer (primary metric) and visit rate, each analysed for both
  emails vs control.
- **Interaction tests:** for each outcome, segment variable and email, OLS of the outcome on the
  arm, the segment levels and arm × level interactions, using customers in that email arm and
  control. For visit this is a linear probability model. A joint Wald test of the interaction
  terms uses HC2 standard errors. That gives 7 variables × 2 emails = 14 tests per outcome.
- **Per-segment effects**, within each level:
  - spend: the Welch difference in means with a 95% CI;
  - visit: the difference in proportions, with a two-proportion z-test and a 95% Newcombe
    interval (as in §6).

  That gives 23 levels × 2 emails = 46 tests per outcome.
- **Multiple testing:** Benjamini–Hochberg (q = 0.05) applied **separately for each outcome's
  family**: 14 + 46 = 60 tests for spend and 60 tests for visit.
- **Labelling:** all segment results are **exploratory**. They generate hypotheses for a future
  test; they are not decisions. Every segment is reported, not only the significant ones.

## 11. Uplift modelling and targeting

- **Outcome:** `visit` as the primary uplift outcome, because it carries the most signal.
  `conversion` is secondary. Spend is not modelled: with 99% zeros it is too noisy for
  individual-level models.
- **Features:** pre-treatment variables only, never an outcome: `recency`, `history`,
  `history_segment`, `mens`, `womens`, `zip_code`, `newbie`, `channel`. No outcome and no arm
  indicator is used as a feature.
- **Learners:**
  - **T-learner** and **X-learner**, each with scikit-learn `HistGradientBoostingClassifier` base
    models using fixed default hyperparameters. There is no tuning against the uplift metrics.
  - Each email is modelled against control separately, which gives τ_mens(x) and τ_womens(x).
- **Baselines.** Each baseline selects 10,000 customers and an email for each, and is evaluated
  exactly like the uplift policy. Here the "A/B winner" is the email that beats control on spend
  in §6. If both do, it is the one with the larger estimated difference; if neither does, it is
  the one with the higher posterior mean spend (§9). Random draws use `config.SEEDS["uplift"]`.
  1. **Random targeting:** 10,000 customers drawn at random, each given one of the two emails at
     random.
  2. **A/B winner at random:** the A/B winner sent to 10,000 customers drawn at random.
  3. **Response model:** a `HistGradientBoostingClassifier` predicting P(visit | emailed, x),
     fitted separately for each email arm on the same 5 cross-fitting folds, so every score is
     out of fold.
     - Each customer gets the email with the higher predicted visit probability.
     - The 10,000 customers with the highest predicted probability are emailed.
     - This is the common industry practice of targeting likely responders. It ignores whether
       the customer would have visited without an email, which is exactly what uplift modelling
       corrects for, so it is the key comparison for the uplift models.
  4. **Recency rule:** the A/B winner sent to the 10,000 customers who purchased most recently
     (lowest `recency`). Ties are broken by higher `history`, then by `customer_id`.
     - Recency is the strongest and most widely used single predictor in direct-marketing
       targeting (the R in RFM), and it needs no model.
     - Keeping the A/B winner as the email isolates the value of choosing *whom* to email with
       a simple rule.
     - It was chosen before any outcome by arm was seen, and over rules based on past category
       (for example, past men's buyers get the mens email). A category rule would require
       guessing in advance which email suits customers who bought both categories.
- **Cross-fitting:** 5 folds, stratified by arm and outcome. Every customer gets an out-of-fold
  score from models that never saw that customer. The fold assignment uses
  `config.SEEDS["uplift"]`.
- **Policy (Q3), with a budget of 10,000 customers (`config.BUDGET`):**
  1. For each customer, take the email with the higher predicted uplift, max(τ_mens, τ_womens).
  2. Rank customers by that value and email the top 10,000, provided their predicted uplift is
     positive. Everyone else gets no email.
- **Do-not-email list (Q4):** the 10,000 customers with the lowest max(τ_mens, τ_womens). We
  report whether their realised uplift, estimated by inverse propensity weighting (IPW), is at or
  below zero.
- **Evaluation** (all on out-of-fold scores):
  - **Qini curves and uplift curves**, with the area under each, for each email vs control.
  - **Policy value** of each targeting policy, estimated by IPW with the known assignment
    probabilities of 1/3: V(π) = mean over customers of Y_i · 1{A_i = π(x_i)} / (1/3). It is
    reported as incremental visits (and conversions) relative to emailing nobody.
  - **95% bootstrap intervals** from 1,000 customer-level resamples (`config.SEEDS["bootstrap"]`),
    with Monte Carlo standard errors. These intervals treat the out-of-fold scores as fixed, so
    they leave out model-fitting variability; this is stated as a limitation.

## 12. What we will not do

- **No stopping early.** The data is a single final snapshot. Peeking and sequential methods are
  shown by simulation only, never applied to choose when to stop this experiment.
- **No dropping outliers in the primary analysis.** Winsorisation is a pre-specified sensitivity
  analysis only.
- **No new segments after seeing results.** Only the segments in §10 are analysed. Any other
  segment would be a deviation, labelled post hoc.
- **No reporting only the significant results.** Every pre-specified comparison, metric,
  sensitivity analysis and segment is reported, including null and unfavourable results.
- **No changing metrics, covariates, priors, thresholds or the budget after seeing outcomes**
  without an entry in Deviations.

## 13. Deviations

Each deviation records the date, the section changed, what changed, why, and whether
it was made before or after seeing the outcome results.

### D1. Multiplicity for the winsorised sensitivity rows; frequentist bootstrap for spend

- **Date:** 2026-11-03. **Sections:** §6, §8. **Made before** any outcome was compared between
  arms (written before the frequentist readout was first run).
- **What changed.**
  1. §8 repeats the primary analysis on winsorised spend but does not say how its p-values are
     adjusted. They mirror the raw-spend rows and are never pooled with them: the two
     winsorised email-vs-control tests form their own Holm family (`sensitivity`), and the
     winsorised mens-vs-womens test is a family of one (`sensitivity_secondary`, adjusted
     p = unadjusted p), so it does not change the size of the pre-registered secondary family.
     Sensitivity rows inform robustness only; decisions use the primary and secondary families.
  2. A percentile bootstrap interval (10,000 resamples, `config.SEEDS["bootstrap"]`) is
     reported next to the Welch interval for every spend row, raw and winsorised, as a
     robustness check on the normal approximation for a zero-inflated, heavy-tailed outcome. It
     does not replace the Welch interval or change any decision.
  3. Implementation detail for §8: the 95th percentile of converters' spend uses linear
     interpolation between order statistics (the numpy / pandas default).
- **Why.** The plan left these unspecified; fixing them before seeing results keeps them from
  being chosen by their outcome.


### D2. CUPED θ pooled over the two arms of each comparison

- **Date:** 2026-11-04. **Section:** §7. **Made after** outcomes were compared between arms:
  the frequentist readout and the CUPED readout had both been run when this entry was written.
- **What changed.** §7 says θ is "estimated on all customers pooled across arms", which can be
  read as all three arms or as the two arms being compared. The implementation estimates θ, and
  centres `history`, on the two arms of each comparison (mens and control for mens vs control,
  and so on), so each comparison is a self-contained two-arm analysis.
- **Why.** Each comparison uses only its own two arms everywhere else in the readout (Welch
  differences, the Lin adjustment), and a two-arm θ keeps CUPED consistent with that: an
  estimate for mens vs control does not depend on the womens arm. Either reading keeps the
  estimate unbiased, since θ is shared by both arms and `history` is pre-treatment.
- **Impact.** Practically none. Re-running CUPED with θ pooled over all three arms changes every
  adjusted estimate by a negligible fraction of its standard error, and no interval, p-value
  or conclusion changes materially. The choice could not have been made to favour a result,
  because `history` correlates so weakly with every outcome that CUPED barely moves any
  estimate under either reading.


### D3. Bayesian rates: draws, a fourth prior strength, and the two outcomes the rule left open

- **Date:** 2026-11-04. **Section:** §9. **Made after** outcomes were compared between arms:
  the frequentist and CUPED readouts had been run when this entry was written; the Bayesian
  readout had not.
- **What changed.**
  1. Each comparison uses **200,000 posterior draws per arm** instead of 20,000. This is a
     precision choice: every Monte Carlo quantity is reported with its Monte Carlo standard
     error, and more draws shrink that error by about a factor of three. Nothing else moves.
  2. A **fourth shared-prior strength, n0 = 100,000** pseudo-customers, is reported after the
     three pre-registered ones (100, 1,000, 10,000). It is a stress test, about five arm-sizes
     of scepticism, added so that the ladder reaches the point where the shrinkage visibly bites.
     The pre-registered priors are still reported in full.
  3. §9 defines when to **ship** but not the other outcomes. They are fixed symmetrically:
     **don't ship** if P(control better) ≥ 0.95 and the expected loss of keeping control is
     below 1% of the control arm's mean; **keep testing** otherwise. The expected loss of
     keeping control, E[max(θ_t − θ_c, 0)], is reported next to the expected loss of shipping.
  4. For the rates the loss threshold is 1% of the **observed** control-arm rate (x_c / n_c),
     the same number under every prior, rather than the posterior mean of the control rate.
- **Why.** Items 1, 2 and 3 were requested at implementation time to make the readout more
  precise and the sensitivity ladder longer; item 4 fixes an ambiguity. None of them changes a
  pre-registered quantity, prior or threshold, and none could favour a result: the extra prior
  is more sceptical, not less, and the ship criterion is untouched.



### D4. Bayesian spend: draws, a parametric cross-check and the per-10,000 rescaling

- **Date:** 2026-11-04. **Section:** §9. **Made after** outcomes were compared between arms:
  the frequentist, CUPED and Bayesian-rate readouts had all been run when this entry was written;
  the Bayesian spend readout had not.
- **What changed.**
  1. The Bayesian bootstrap uses **200,000 posterior draws per arm** instead of 4,000, for the
     same reason as D3 item 1: every Monte Carlo quantity carries its standard error and the
     extra draws shrink it. Each arm is drawn once and reused across its comparisons, so the
     three comparisons share one joint posterior rather than being drawn separately.
  2. A **parametric two-part cross-check** is reported next to the bootstrap, never in its
     place: conversion ~ Beta-Binomial with the flat prior, log spend among converters ~ Normal
     with a weak Normal-Inverse-Gamma prior (μ0 = 0, κ0 = α0 = β0 = 0.01, the same for every
     arm), mean spend per customer = p · exp(μ + σ² / 2), with a posterior predictive check of the
     log-normal against the observed converters, including the share at the \$499 top-code. It
     runs on raw spend only. It has no role in the decision; it tests whether a parametric model
     agrees with the non-parametric one, and the check is reported whatever it shows.
  3. The decision rule's loss threshold is 1% of the **observed** control-arm mean spend of the
     same metric (the sample mean, as D3 item 4 fixed for the rates), rather than its posterior
     mean. Under the Bayesian bootstrap the two are identical, so this only matters for the
     cross-check rows, where the same threshold is used.
  4. The posterior of the difference, its interval and the expected loss are **also expressed
     per 10,000 emails sent** (the Q3 budget), by multiplying the per-customer values by 10,000.
     This is a rescaling for the README, not a new quantity.
  5. Raw versus winsorised spend is handled as the plan says: raw spend is the primary Bayesian
     analysis and the same bootstrap on winsorised spend is its sensitivity analysis (§8, §9).
     No change there.
- **Why.** Items 1, 2 and 4 were requested at implementation time; item 3 fixes an ambiguity
  the same way D3 did. None of them changes a pre-registered quantity, prior, threshold or
  metric, and none could favour a result: the cross-check cannot overrule the bootstrap, and the
  threshold definition is the one the plan describes ("the control mean is computed at readout
  time").


### D5. Segments: conversion added as a third exploratory outcome

- **Date:** 2026-11-04. **Section:** §10. **Made after** outcomes were compared between arms:
  every readout of §6, §7 and §9 had been run when this entry was written; the segment
  analysis had not.
- **What changed.** §10 names spend and visit as the segment outcomes. Conversion is analysed
  as well, with the same per-level test as visit (difference in proportions, pooled z-test,
  Newcombe interval), the same interaction test, and its **own** Benjamini–Hochberg family of
  46 + 14 = 60 tests. The spend and visit families are exactly as pre-specified; nothing is
  pooled across outcomes. Every conversion segment row is labelled exploratory like the rest.
- **Why.** Requested at implementation time so that all three metrics are read within segments.
  Adding a family cannot change the adjusted p-values of the pre-specified families, so it
  cannot favour a result in them; conversion's own family is adjusted at the same q.
- **Note.** The challenge's question (did each email work better among past buyers of its own
  merchandise?) is answered from the pre-specified `mens` and `womens` segment rows and
  interaction tests inside the family; it needs no new test and is exploratory, not post hoc.

### D6. Uplift features: `history_segment` dropped

- **Date:** 2026-11-04. **Section:** §11. **Made before** any uplift model was fitted; the
  feature matrix was being written when this entry was added.
- **What changed.** §11 lists `history_segment` among the uplift features. It is not used.
  The feature matrix holds `recency`, `history`, `mens`, `womens`, `newbie` and one-hot
  indicators for `zip_code` and `channel`.
- **Why.** `history_segment` is `history` binned into seven dollar bands, so it is redundant:
  it carries no information the numeric column does not already carry, and the tree models
  can place their own splits on `history`. Removing a redundant copy cannot be chosen by its
  effect on a result because no uplift model had been fitted.

### D7. Uplift learners: S-learner added; hyperparameters chosen on an inner validation split

- **Date:** 2026-11-04. **Section:** §11. **Made before** any uplift model was fitted on the
  experiment data (the learners were being written when this entry was added).
- **What changed.**
  1. An **S-learner** (one classifier with the treatment indicator as a feature) is fitted
     and evaluated next to the pre-specified T- and X-learners, on the same folds.
  2. §11 says the base models use "fixed default hyperparameters". Instead, each base model
     picks one of four fixed grid points (learning rate 0.05 or 0.1, 15 or 31 leaves) on an
     inner validation split of its training fold, by log loss for classifiers and squared
     error for the X-learner's second-stage regressors, with early stopping. The evaluation
     fold is never used for any choice. There is still no tuning against the uplift metrics.
  3. The baselines evaluated as rankings are random targeting, the recency rule and the
     response model. The plan's fourth baseline, "A/B winner at random", selects customers at
     random like baseline 1 and differs only in which email is sent, so as a ranking within
     one email-vs-control pair it coincides with random targeting; it is evaluated as its own
     policy in the budgeted targeting stage, where the choice of email matters.
- **Why.** The S-learner is the simplest meta-learner and a useful reference. Choosing
  hyperparameters on an inner split by the base models' own losses guards against a poorly
  fitted default without touching the uplift metrics or the evaluation fold.

### D8. Targeting policies: mens-by-default variant, level-calibration check, learner choice

- **Date:** 2026-11-04. **Section:** §11. **Made after** the uplift-model evaluation had
  been run and read.
- **What changed.**
  1. The uplift policy uses the **X-learner's visit scores for both emails**. For the womens
     email it was the best ranking and meaningfully above every baseline. For the mens email
     no ranking beat random, so choosing a "best" mens learner by its Qini coefficient would
     select on noise; the same learner is used for both pairs instead.
  2. A **level-calibration check** is added before the policy is used: the mean predicted
     uplift in each pair against the observed average effect (with its Newcombe interval),
     next to the decile calibration plots. The uplift policy compares predicted levels across
     two models, so both must be on the observed scale.
  3. An **exploratory mens-by-default policy**: because the mens effect shows no detectable
     heterogeneity, its uplift is held at the pair's average effect; the womens email is sent
     only where the womens model predicts more than that constant, the mens email otherwise;
     customers are ranked by the chosen uplift and the top 10,000 are emailed. It is labelled
     exploratory and is not a decision.
  4. The response-model baseline follows §11 (each customer gets the email under which the
     model predicts the higher visit probability; the 10,000 highest are emailed).
  5. Policy values are reported both by plain IPW with the design probability 1/3 and by the
     self-normalised estimator; increments are given in total, per email sent and as a share
     of the effect of emailing everyone with the A/B winner. A cost-per-email parameter
     (`config.COST_PER_EMAIL`, default 0) gives a net value. Policies are compared by paired
     bootstrap differences on the same resamples, not by overlapping intervals.
- **Why.** Items 1 to 3 respond to what the uplift evaluation showed and are recorded for
  that reason; 3 is exploratory. Items 4 and 5 fix details the plan left open.
- **Clarification (added 2026-11-16; the text above is unchanged).**
  - **Items 1, 2 and 3 are results-driven (post hoc).** All three were decided after the
    uplift evaluation had been run and read: the Qini and uplift curves, their areas and the
    paired bootstrap comparisons of every ranking were known when they were chosen. None of
    them is part of the pre-registered analysis.
  - **Item 1 is a post-hoc model choice.** The frozen §11 pre-specified two learners, the
    T-learner and the X-learner, each on `HistGradientBoostingClassifier` base models, and
    defined the policy on "predicted uplift" without saying which learner supplies it or how
    one would be selected; D7 then added the S-learner before any model was fitted. The
    X-learner was picked from these three because of how it ranked in the evaluation, so the
    policy built on it is a post-hoc analysis, not a confirmatory one.
  - **The X-learner's reported Qini is optimistic.** It was selected on the same out-of-fold
    evaluation that reports its Qini coefficient, so that coefficient is the largest of
    several noisy estimates and is biased upward as an estimate of how the model would rank
    new customers. The same holds for every policy value computed from its scores. No
    sample was held out from the selection, so the size of this optimism is not estimated
    here. The evaluation itself (`uplift_models.summary` in `reports/results.json`) does not
    separate the X-learner from the S-learner on the womens pair, which is a further reason
    to read "best" as "chosen", not as "shown to be better".
  - **The repeated cross-fitting is supporting, not independent, evidence.** The ten fold
    seeds (`seed_robustness` in `reports/results.json`) re-split and re-fit on the same
    64,000 customers and the same outcomes. They show that the ranking on the womens pair
    does not depend on one fold assignment or model seed. They cannot show that it would
    hold on new customers, and they do not remove the selection effect, because every seed
    reuses the data the choice was made on. Confirming the choice needs a new experiment or
    a sample kept aside before any model is compared.

### D9. Random targeting: one shared random draw for the fixed budget and the budget curve

- **Date:** 2026-11-17. **Section:** §11. **Made after** the policy results had been seen.
- **What changed.** §11 defines random targeting by customers "drawn at random" with
  `config.SEEDS["uplift"]`. The policy stage used to make two draws from that seed: a
  permutation whose first 10,000 customers formed the fixed-budget random policy
  (`reports/tables/policy.csv`), and a second, independent ordering for the random line of
  the budget curve (`reports/tables/policy_budget_curve.csv`). The two artifacts therefore
  gave different point estimates for the same policy at the same budget. The stage now draws
  one permutation of the customers. The fixed-budget policy emails its first 10,000, and the
  budget curve emails its first *k* at every budget *k*, so the curve at 10,000 reproduces
  the policy table exactly.
- **Which draw is kept.** The permutation behind `policy.csv`. The policy table, its
  bootstrap intervals and every paired comparison are unchanged; only the random line of the
  budget curve (table and figure) changes. The draw was kept because the published
  comparisons already rest on it, not because of its value.
- **Alternative not taken.** The expected value of random targeting needs no draw: it is the
  budget times the A/B winner's average treatment effect, with an interval from that
  effect's standard error. It was not adopted because §11 pre-registered a random draw, and
  because a drawn policy enters the paired bootstrap comparisons on the same footing as
  every other policy. A single draw still differs from that expectation by sampling noise.
- **Why.** One policy should have one value at one budget. Nothing else in the analysis
  changes.

### D10. Decision rule for choosing a targeting policy

- **Date:** 2026-11-17, 19:54 CET. **Section:** §11. **Written after** an independent review
  of the repository **and before** the corrected results exist. The review found that the
  model-based policies select on the realised arm (corrected in a later deviation) and that
  the comparison with random targeting rested on a single random draw. When this rule was
  written, the uncorrected policy results and the review's own estimates had been seen; the
  results that follow the corrections had not been computed.
- **What changed.** §11 pre-registered how the policies are built and evaluated but no rule
  for choosing one. The rule is:

  > A ranking is recommended for a 10,000 budget only if its paired difference against
  > random targeting (the exact expectation, budget x ATE) excludes zero on spend or on
  > visits. Among rankings that qualify, choose the simplest (recency rule, then response
  > model, then uplift model) unless a more complex one is separated from it. If none
  > qualifies, the memo says targeting is not shown to add value at this budget, and that
  > any of the rankings or a random 10,000 is defensible.

- **Why.** Without a rule fixed in advance, the recommendation could follow whichever
  comparison happened to look best. The rule is fixed before the corrected numbers so that
  it cannot be fitted to them. It is not a pre-registered rule in the sense of §1 to §12: it
  was written with knowledge of the uncorrected results, and the memo says so.

### Formatting note (2026-11-17)

Dollar signs in this file are now written with a backslash so that GitHub does not render
amounts as mathematics. No wording, number or decision changed.
