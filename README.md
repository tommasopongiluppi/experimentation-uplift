# Which email should we send, and to whom? The Hillstrom email experiment

[![CI](https://github.com/tommasopongiluppi/experimentation-uplift/actions/workflows/ci.yml/badge.svg)](https://github.com/tommasopongiluppi/experimentation-uplift/actions/workflows/ci.yml)

A decision memo on a randomised email experiment, backed by a tested Python package (`abkit`)
that produces every number below.
<!-- RESULTS:intro -->
64,000 customers were randomised to a Mens email, a Womens email or no email, and their visits, purchases and spend were tracked for 14 days (Hillstrom's MineThatData challenge, 2008).
<!-- /RESULTS:intro -->

## TL;DR

<!-- RESULTS:summary -->
**Decision: send the Mens email, to everyone who can be emailed.** Both emails beat sending nothing on spend, and the Mens email beats the Womens email. This assumes an email costs nothing: the data hold no cost, margin or unsubscribe figures.

| Headline (two-week window) | Estimate | 95% interval | Adjusted p | Correction |
|---|---|---|---|---|
| Mens vs no email: spend per customer | +\$0.77 | \$0.49 to \$1.05 | <0.001 | Holm, 2 primary tests |
| Womens vs no email: spend per customer | +\$0.42 | \$0.17 to \$0.68 | 0.001 | Holm, 2 primary tests |
| Mens vs Womens: spend per customer | +\$0.35 | \$0.03 to \$0.66 | 0.030 | Benjamini-Hochberg, 7 secondary tests |
| Mens vs no email: visit rate (10.6% → 18.3%) | +7.7 pp | 7.0 to 8.3 pp | <0.001 | Benjamini-Hochberg, 7 secondary tests |

Holm is the correction across the 2 primary tests (each email against no email on spend); Benjamini-Hochberg is the correction across the 7 secondary tests (every other comparison of spend, conversion and visits). Intervals are unadjusted.

- **Value.** Sent to all 64,000 customers, the Mens email brings about \$49,000 of extra revenue in two weeks (\$32,000 to \$67,000). Profit cannot be computed: the data hold no cost or margin.
- **Targeting under a budget, and whom not to email: under revision.** The targeting results were corrected after an independent review (analysis plan D11 to D13). The corrected numbers and the decision rule applied to them are in [Who to email](#5-who-to-email); this summary will be rewritten from them.
<!-- /RESULTS:summary -->

The analysis plan was written and committed before any outcome was compared between arms
([docs/analysis_plan.md](docs/analysis_plan.md)); every later change is listed there as a
deviation. Numbers in this file are generated from `reports/results.json`, never typed.

## Contents

1. [The business questions](#1-the-business-questions)
2. [Is the experiment valid?](#2-is-the-experiment-valid)
3. [Results](#3-results)
4. [Two pitfalls, quantified](#4-two-pitfalls-quantified)
5. [Who to email](#5-who-to-email)
6. [How it is verified](#6-how-it-is-verified)
7. [Limitations](#7-limitations)
8. [Reproduce](#8-reproduce)
9. [Repository structure](#9-repository-structure)
10. [References](#10-references)

## 1. The business questions

These are the questions of the original challenge. Each answer is generated from the results.

<!-- RESULTS:questions -->
| # | Question | Answer |
|---|---|---|
| Q1 | Which email performed better, and on which metric? | The Mens email, on all three metrics. Against the Womens email: visit rate +3.1 pp (2.4 to 3.8 pp); conversion rate +0.37 pp (0.17 to 0.57 pp); spend per customer +\$0.35 (\$0.03 to \$0.66), each significant after adjustment. |
| Q2 | How much incremental spend per customer did each email drive? | Mens +\$0.77 (\$0.49 to \$1.05), a relative lift of +118%; Womens +\$0.42 (\$0.17 to \$0.68), a relative lift of +65%, on a no-email baseline of \$0.65 per customer. |
| Q3 | If only 10,000 customers can be emailed, who should get which email? | By the decision rule (analysis plan D10), use the uplift model: it is the only ranking that qualifies. A ranking qualifies when its paired difference against random targeting excludes zero on spend or on visits; here that holds for the uplift model (on spend). |
| Q4 | Which 10,000 customers should not be emailed? | No such group was found. The 10,000 customers with the lowest predicted uplift still respond to their better email: +7.0 pp visits (5.2 to 9.0 pp). |
| Q5 | Do the effects differ across customer segments? | For the Womens email, yes: it lifts visits by +7.3 pp (6.4 to 8.2 pp) among past buyers of women's merchandise and by +1.1 pp (0.2 to 2.0 pp) among the rest (interaction adjusted p <0.001). 6 of its 21 interaction tests are significant after Benjamini-Hochberg, against 0 of 21 for the Mens email, whose effect is close to uniform. Exploratory, not a decision. |
<!-- /RESULTS:questions -->

## 2. Is the experiment valid?

Before reading any outcome, check that the randomisation worked: the arms should have the
planned sizes and look alike on everything measured before the email.

<!-- RESULTS:validity -->
- **Sample ratio mismatch: none.** Arm sizes are 21,307 (Mens), 21,387 (Womens) and 21,306 (no email) against an expected 21,333 each: chi-square 0.20 on 2 degrees of freedom, p = 0.90 (alarm threshold 0.001).
- **Balance: no variable flagged.** The largest absolute standardised mean difference between any two arms, over every pre-treatment variable and level, is 0.017 (`channel` = `phone`), against a threshold of 0.1; 0 are flagged. Chi-square tests of the categorical variables give p = 0.36 (`history_segment`), 0.58 (`zip_code`) and 0.46 (`channel`).
<!-- /RESULTS:validity -->

The data validation report is in [reports/validation.md](reports/validation.md). Only
variables measured before the email are used as covariates, features or segment definitions.

## 3. Results

Differences are the first arm minus the second, per customer, over the two weeks after the
email.

<!-- RESULTS:readout -->
| Metric | Comparison | Difference | 95% interval | Relative lift | Adjusted p |
|---|---|---|---|---|---|
| Spend per customer | Mens vs no email | +\$0.77 | \$0.49 to \$1.05 | +118% (63% to 192%) | <0.001 |
| Spend per customer | Womens vs no email | +\$0.42 | \$0.17 to \$0.68 | +65% (22% to 124%) | 0.001 |
| Spend per customer | Mens vs Womens | +\$0.35 | \$0.03 to \$0.66 | +32% (3% to 70%) | 0.030 |
| Conversion rate | Mens vs no email | +0.68 pp | 0.50 to 0.86 pp | +119% (77% to 171%) | <0.001 |
| Conversion rate | Womens vs no email | +0.31 pp | 0.15 to 0.47 pp | +54% (23% to 94%) | <0.001 |
| Conversion rate | Mens vs Womens | +0.37 pp | 0.17 to 0.57 pp | +42% (18% to 71%) | <0.001 |
| Visit rate | Mens vs no email | +7.7 pp | 7.0 to 8.3 pp | +72% (64% to 81%) | <0.001 |
| Visit rate | Womens vs no email | +4.5 pp | 3.9 to 5.2 pp | +43% (36% to 50%) | <0.001 |
| Visit rate | Mens vs Womens | +3.1 pp | 2.4 to 3.8 pp | +21% (16% to 26%) | <0.001 |

Spend against control is the primary family (2 tests, Holm); the other 7 comparisons are secondary (Benjamini-Hochberg). Spend uses Welch's t-test, rates a two-proportion z-test with Newcombe intervals.

- **Sensitivity.** With converters' spend winsorised at \$362.20, the differences against control are +\$0.74 (\$0.47 to \$1.00) for Mens and +\$0.41 (\$0.18 to \$0.65) for Womens. A percentile bootstrap of raw spend (10,000 resamples) gives \$0.49 to \$1.07 for Mens and \$0.18 to \$0.68 for Womens, so the normal approximation is not doing the work.
- **Bayesian readout** (Bayesian bootstrap on spend, 200,000 draws). P(Mens better than no email) = 1.000, credible interval \$0.49 to \$1.06. P(Womens better than no email) = 0.999, credible interval \$0.17 to \$0.68. P(Mens better than Womens) = 0.985, credible interval \$0.03 to \$0.66. The pre-registered rule (ship if P(better) is at least 0.95 and the expected loss is below 1% of the control mean) recommends the Mens email. The Bayesian bootstrap puts a uniform Dirichlet prior on the weights of the observed spend values and no prior on their size, so with arms this large it is expected to agree with the frequentist readout, and it does.
<!-- /RESULTS:readout -->

![Differences between arms with 95% intervals for spend, winsorised spend, conversion and visits, each shown raw, with CUPED and with the Lin regression; the three intervals coincide](reports/figures/intervals_raw_cuped_lin.png)

### Covariate adjustment

<!-- RESULTS:cuped -->
**CUPED and the regression adjustment barely reduce variance here.** CUPED removes a share of the variance equal to the squared correlation between the outcome and the pre-experiment covariate, and that correlation is tiny:

| Metric | Correlation with past-year spend | Variance removed by CUPED | Variance removed by Lin regression (15 covariates) |
|---|---|---|---|
| Spend per customer | 0.020 to 0.024 | 0.04% to 0.06% | -0.22% to -0.04% |
| Conversion rate | 0.026 to 0.032 | 0.07% to 0.10% | -0.24% to 0.27% |
| Visit rate | 0.063 to 0.067 | 0.40% to 0.44% | 2.81% to 2.91% |

- For spend, the standard error of the Mens effect goes from \$0.14525 to \$0.14522 with CUPED and to \$0.14528 with the regression. A negative value in the table means the adjustment added a little noise.
- **Minimum detectable effect.** With each arm's own variance, at the Holm level and 80% power, the smallest spend effect the experiment could detect is \$0.45 for Mens (\$0.45 with CUPED; observed \$0.77) and \$0.40 for Womens (\$0.40 with CUPED; observed \$0.42). CUPED does not move it. The emails raise the variance of spend in their own arm, so these are larger than a planning figure based on the control variance alone.
- **Why.** 99.1% of customers spend nothing in the two weeks, so spend is almost entirely a rare, noisy purchase event that last year's spend cannot predict (correlation at most 0.024). Visits are more predictable, and there the full regression removes up to 2.9%. CUPED pays off when the covariate is the same metric measured before the experiment on returning users; a one-off campaign with a two-week window is not that setting.
<!-- /RESULTS:cuped -->

## 4. Two pitfalls, quantified

The experiment itself was read once, at the end, with pre-registered families of tests. The
two simulations below show what the usual shortcuts would have cost.

### Peeking

<!-- RESULTS:peeking -->
Simulated experiments shaped like this one (21,306 customers per arm arriving evenly over 14 days, visit rate as in the control arm, 20,000 runs). Looking every day and stopping at the first p < 0.05 declares a winner in 21.9% of experiments in which the email does nothing, against 4.8% for a single test at the end.

| Rule | False positive rate with a single look (± Monte Carlo SE) | False positive rate with 14 daily looks | Power at a 7.9% relative lift | Mean stopping day at that lift |
|---|---|---|---|---|
| One test at the end (no peeking) | 5.2% (±0.2%) | 4.8% (±0.2%) | 79.0% (±0.3%) | 14.0 |
| Unadjusted test at every look, stop at the first p < 0.05 | 5.2% (±0.2%) | 21.9% (±0.3%) | 87.0% (±0.2%) | 6.7 |
| Bayesian rule at every look, stop when P(better) leaves (0.05, 0.95) | 10.5% (±0.3%) | 38.4% (±0.3%) | 93.0% (±0.2%) | 5.3 |
| Pocock boundary | 5.1% (±0.2%) | 4.9% (±0.2%) | 65.8% (±0.3%) | 9.7 |
| O'Brien-Fleming boundary | 5.1% (±0.2%) | 4.9% (±0.2%) | 77.3% (±0.3%) | 10.8 |
| Mixture SPRT (always-valid p-value) | 0.2% (±0.0%) | 0.9% (±0.1%) | 46.5% (±0.4%) | 11.7 |

Compare each rule with its own single-look rate. The Bayesian rule is two-sided at 10% by construction, so it starts from about twice the rate of the 5% test; peeking multiplies both. The single-look column uses 10,000 runs.

The group-sequential boundaries are calibrated by simulation; for five looks they come out at 2.416 (Pocock) and 2.040 (O'Brien-Fleming), against the published 2.413 and 2.04.
<!-- /RESULTS:peeking -->

![False positive rate against the number of looks for each stopping rule: unadjusted and Bayesian peeking climb well above the nominal level, the corrected rules stay at or below it](reports/figures/sequential_fpr_vs_looks.png)

The valid rules trade power for the option to stop early, each in its own way; the last two
columns of the table show what each one gives up and gains.

### Multiple testing

Cutting the customers into many segments and testing each one finds "effects" by chance.

<!-- RESULTS:multiple_testing -->
Simulation: 20 segments of 1,065 customers per arm, 10,000 runs.

| Scenario | Unadjusted tests at 0.05 | Benjamini-Hochberg at 0.05 |
|---|---|---|
| No segment has an effect: chance of at least one false discovery | 63.5% (theory 64.2%) | 4.7% |
| 5 of 20 segments have an effect: share of discoveries that are false | 14.8% | 3.9% |
| Same scenario: power to find a true effect | 74.7% | 47.8% |

**On the real data the correction changes little, because most of the effects are real.** Across 23 segment levels and two emails, the significant tests are 33 of 60 unadjusted and 26 after Benjamini-Hochberg for spend; 49 of 60 unadjusted and 47 after Benjamini-Hochberg for visits; 38 of 60 unadjusted and 36 after Benjamini-Hochberg for conversions. Both emails lift visits in nearly every segment, so there are few true nulls for the correction to protect against. The correction matters for the claim of a *difference between* segments, which is what the interaction tests in Q5 address.
<!-- /RESULTS:multiple_testing -->

## 5. Who to email

### Uplift models

An uplift model ranks customers by how much the email changes their behaviour, not by how
likely they are to respond anyway. Three meta-learners (S, T and X, on gradient-boosted
trees, cross-fitted so every score is out of fold) are compared with three baselines: a
response model, a recency rule and a random order.

<!-- RESULTS:uplift_models -->
| Ranking | Mens email vs no email | Womens email vs no email |
|---|---|---|
| S-learner | 0.010 (-0.005 to 0.025) | 0.063 (0.047 to 0.079) |
| T-learner | 0.009 (-0.006 to 0.025) | 0.060 (0.044 to 0.077) |
| X-learner | 0.008 (-0.006 to 0.024) | 0.062 (0.046 to 0.078) |
| Response model (baseline) | 0.025 (0.011 to 0.039) | 0.037 (0.022 to 0.052) |
| Recency rule (baseline) | 0.013 (0.000 to 0.027) | -0.004 (-0.020 to 0.013) |
| Random order (baseline) | -0.008 (-0.023 to 0.007) | -0.007 (-0.021 to 0.008) |

Normalised Qini coefficient on visits, out of fold, with 95% bootstrap intervals (1,000 resamples): 0 is a random ranking, 1 the best ranking the outcomes allow.

- **Womens email.** The S-learner ranks best, ahead of the random order, recency rule and response model by a paired bootstrap difference, and is not separated from the T-learner and X-learner.
- **Mens email.** The response model ranks best, ahead of the random order, S-learner and X-learner by a paired bootstrap difference, and is not separated from the recency rule and T-learner. No uplift learner beats the baselines: the email's effect on visits is close to the same for everyone, so there is little to rank.
- **Stability.** Repeating the cross-fitting under 10 fold seeds, the X-learner's coefficient on the Womens pair has median 0.061 (range 0.056 to 0.064) and is above the response model in 10 of 10 seeds.
- **Policy value across fold seeds.** Incremental visits at the 10,000 budget over the same 10 seeds: uplift policy 807 in the committed seed, median 882, range 771 to 909; response model 954 in the committed seed, median 984, range 942 to 1,077. The bootstrap intervals hold the folds fixed, so this spread comes on top of them; read the median as the central value.
- **The choice of the X-learner for the policy is post hoc (deviation D8).** It was picked from three learners after an earlier evaluation on the same data, which the correction of deviation D11 has since replaced; the choice was not revisited, so that the policy is not selected a second time on the corrected results. Read "chosen", not "shown to be better": in the table above it is not the top-ranked learner on the Womens pair, and it is not separated from the other learners that rank near it.
<!-- /RESULTS:uplift_models -->

### The budgeted policy

Each policy picks the customers to email and the email each one gets. Three rankings are
compared: the recency rule (the most recent purchasers get the A/B winner; no model), the
response model (the customers most likely to visit, each with the email they are more likely
to respond to) and the uplift model (the customers with the highest predicted effect of an
email, each with their better email). Two random baselines are shown next to them.

<!-- RESULTS:policy -->
| Policy, 10,000 emails | Mens / Womens emails | Incremental visits | Incremental conversions | Incremental spend |
|---|---|---|---|---|
| Uplift policy (X-learner) | 7,786 / 2,214 | 807 (594 to 1,032) | 129 (72 to 183) | \$16,806 (\$7,601 to \$26,996) |
| Response model | 7,707 / 2,293 | 954 (717 to 1,188) | 99 (42 to 159) | \$9,373 (\$965 to \$18,308) |
| Recency rule | 10,000 / 0 | 879 (681 to 1,095) | 87 (39 to 144) | \$9,683 (\$1,817 to \$18,121) |
| Random customers, A/B winner (one draw; plan baseline 2) | 10,000 / 0 | 822 (645 to 990) | 27 (-12 to 66) | \$7,138 (\$1,077 to \$13,721) |
| Random customers, random email (one draw; plan baseline 1) | 5,011 / 4,989 | 750 (582 to 918) | 33 (-9 to 75) | \$7,630 (\$1,954 to \$14,017) |

Increments over emailing nobody with 95% intervals, estimated by inverse propensity weighting with the design probability and a paired bootstrap (1,000 resamples). The recency rule sends the A/B winner (Mens).

**Against random targeting.** Random targeting of the same number of customers with the Mens email has a known expectation, the number of emails times the average effect: 766 visits, 68 conversions and \$7,698 of spend. The table gives each policy minus that expectation, with both recomputed on the same bootstrap resamples; a star marks an interval that excludes zero.

| Policy minus the random expectation | Visits | Conversions | Spend |
|---|---|---|---|
| Uplift policy (X-learner) | +41 (-163 to 232) | +61* (9 to 108) | +\$9,107* (\$323 to \$17,973) |
| Response model | +188 (-25 to 409) | +31 (-21 to 82) | +\$1,675 (-\$6,309 to \$9,608) |
| Recency rule | +113 (-72 to 301) | +19 (-27 to 68) | +\$1,985 (-\$5,008 to \$9,249) |
| Random customers, A/B winner (one draw; plan baseline 2) | +56 (-113 to 218) | -41* (-79 to -5) | -\$560 (-\$6,779 to \$5,535) |
| Random customers, random email (one draw; plan baseline 1) | -16 (-191 to 155) | -35 (-76 to 5) | -\$68 (-\$6,244 to \$6,565) |

- **Decision rule.** By the decision rule (analysis plan D10), use the uplift model: it is the only ranking that qualifies. A ranking qualifies when its paired difference against random targeting excludes zero on spend or on visits; here that holds for the uplift model (on spend). The rule was written after the uncorrected results had been seen and before these numbers were computed.
- **No ranking is separated from another** on visits, conversions or spend. In visits: the uplift policy minus the response model -147 (-369 to 63), the uplift policy minus the recency rule -72 (-303 to 168) and the response model minus the recency rule +75 (-165 to 327).
- **Budget curve.** Extending each ranking to every budget in steps of 1,000, the recency rule is at or above the uplift policy at 47 of 63 budgets on visits and at 24 of 63 on spend. These are point estimates.
- **Do-not-email list.** The 10,000 customers with the lowest predicted uplift, given their better email, gain +7.0 pp in visit rate (5.2 to 9.0 pp) and +\$1.11 in spend (\$0.45 to \$1.84). The model does not find customers who are better left alone.

**Does the selection depend on the arm?** A policy built from pre-treatment variables and out-of-fold scores must pick customers from the three arms in the proportions of the whole population; if it does not, its value above is biased. Counts of the selected customers by the arm they were randomised to:

| Selected group | Mens arm | Womens arm | Control | Expected (Mens / Womens / Control) | p | Passes |
|---|---|---|---|---|---|---|
| Uplift policy, targeted | 3,389 | 3,236 | 3,375 | 3,329 / 3,342 / 3,329 | 0.08 | yes |
| Response model, targeted | 3,283 | 3,365 | 3,352 | 3,329 / 3,342 / 3,329 | 0.62 | yes |
| Mens-by-default policy (exploratory), targeted | 3,367 | 3,344 | 3,289 | 3,329 / 3,342 / 3,329 | 0.63 | yes |
| Do-not-email list | 3,353 | 3,335 | 3,312 | 3,329 / 3,342 / 3,329 | 0.87 | yes |
| Recency rule, targeted (no model) | 3,309 | 3,360 | 3,331 | 3,329 / 3,342 / 3,329 | 0.89 | yes |
| Random draw (no model) | 3,304 | 3,394 | 3,302 | 3,329 / 3,342 / 3,329 | 0.54 | yes |

Every group passes the sample-ratio test (alarm threshold 0.001). An earlier version of the scoring failed it; that error and its correction are deviation D11.
<!-- /RESULTS:policy -->

![Incremental visits and spend against the number of customers emailed for the uplift policy, the response model, the recency rule and random targeting; the four curves stay close together and cross repeatedly](reports/figures/policy_budget_curve.png)

### Random targeting: the expectation and one draw

<!-- RESULTS:random_targeting -->
| 10,000 random customers | Exact expectation, Mens email | One draw, Mens email (baseline 2) | That draw minus the expectation | Same draw, random email (baseline 1) |
|---|---|---|---|---|
| Incremental visits | 766 (704 to 837) | 822 (645 to 990) | +56 (-113 to 218) | 750 (582 to 918) |
| Incremental conversions | 68 (51 to 86) | 27 (-12 to 66) | -41* (-79 to -5) | 33 (-9 to 75) |
| Incremental spend | \$7,698 (\$4,916 to \$10,489) | \$7,138 (\$1,077 to \$13,721) | -\$560 (-\$6,779 to \$5,535) | \$7,630 (\$1,954 to \$14,017) |

The expectation is the budget times the average effect and needs no draw; its interval comes from the bootstrap of that effect. The draw is one random set of customers (analysis plan §11, deviations D9 and D13). Its difference from the expectation is the luck of that one draw; a star marks an outcome on which the drawn customers landed measurably off. That is why the rankings are compared with the expectation and not with the draw (deviation D12).
<!-- /RESULTS:random_targeting -->

## 6. How it is verified

Every statistical function has a test with a known answer. The simulation-based ones matter
most, because they can fail in ways a code review would not catch.

<!-- RESULTS:verification -->
| Check | What is tested | What it rules out |
|---|---|---|
| Interval coverage | The 95% intervals for rates, spend, CUPED, the Lin regression and the Qini bootstrap are computed on repeated simulated experiments with a known effect and must cover it about 95% of the time, within Monte Carlo error. | Intervals that are too narrow on zero-inflated, heavy-tailed spend. |
| Calibration | Credible intervals cover parameters drawn from the prior at their nominal rate; the two-part spend model recovers known parameters; out-of-fold uplift scores are checked by decile against observed effects. | A posterior or a model whose stated uncertainty or level is wrong. |
| Published boundary constants | The simulated five-look boundaries (2.416 and 2.040) match the published 2.413 and 2.04; with one look both reduce to the normal quantile; every valid rule holds its false positive rate. | A sequential simulation that is wrong in a way that still looks plausible. |
| Oracle rankings | On synthetic customers with a known individual effect, the true-effect ranking scores a normalised Qini of 1, random rankings average 0, and each learner ranks customers in line with the planted effect. | An uplift metric or learner that rewards likely responders, not persuadable ones. |
| Known answers | Every estimator is compared with scipy or statsmodels, a closed form or planted data: for example 20 null segments give at least one false discovery in 63.5% of runs against 64.2% in theory, and inverse propensity weighting recovers the known value of a fixed policy. | Formula and indexing errors. |
| Leakage guard | The feature builder refuses any outcome or arm column by name, and before any fit a feature whose values copy or recode the outcome or the treatment is rejected; every score is out of fold. A noisy function of an outcome cannot be detected from values. | Results inflated by information from the outcome. |
| Selection independent of the arm | On pure-noise features a rule that picks the highest scores selects customers from the three arms in the design proportions, and the earlier fold-averaged scoring fails the same sample-ratio test; the committed policies must pass it. | A policy that selects on the realised arm, which biases its estimated value. |
<!-- /RESULTS:verification -->

The slow simulations run on demand (`pytest -m slow`); everything else, including an
end-to-end run of the pipeline on a synthetic fixture, runs in CI on every push. The methods,
their formulas, assumptions and tests are written up in
[docs/methodology.md](docs/methodology.md).

## 7. Limitations

- **Old data.** The experiment ran in 2008. Email habits, inbox filtering and the retailer's
  customers have changed; the effect sizes should not be carried over to a campaign today.
- **A two-week window.** Purchases pulled forward from later weeks count as gains, and later
  effects, good or bad, are not seen.
- **No cost and no unsubscribe data.** Spend is revenue, not margin, and the cost of an email
  is set to zero. With no unsubscribes or complaints recorded, "email everyone" looks better
  here than it would with the full picture, and the search for customers who should not be
  emailed has no negative outcome to find.
- **A single experiment.** One campaign, one season, one creative per email. "Mens beats
  Womens" is a statement about these two emails, not about men's and women's merchandise.
- **Exploratory segments.** The segment analysis was pre-specified as exploratory. Its
  findings are hypotheses for a follow-up test, not decisions.
- **A post hoc model choice.** The learner behind the targeting policy was chosen after the
  model comparison was seen (deviation D8), with no held-out sample, so its results are
  optimistic by an unknown amount.
- **Noisy policy values.** Inverse propensity weighting uses only the customers whose random
  arm matches the policy, about a third of them, so the intervals are wide, above all for
  spend, where almost every customer contributes a zero. The intervals also treat the model
  scores as fixed and leave out the variability of fitting the models.
- **A stylised peeking study.** The data have no arrival dates, so the simulation assumes
  customers arrive evenly and outcomes are known at once. It shows the size of the problem
  and the cost of each fix, not what happened in this experiment.

## 8. Reproduce

```bash
git clone https://github.com/tommasopongiluppi/experimentation-uplift.git
cd experimentation-uplift
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Download the data (checksum verified), run every stage, write reports/results.json,
# reports/tables/ and reports/figures/. Unchanged stages are skipped on later runs.
python -m abkit.pipeline run-all

# Optional: repeat the cross-fitting over fold seeds (slow).
python -m abkit.pipeline run-all --with-robustness

# Regenerate the numbers in this README from reports/results.json.
python -m abkit.report update-readme

# Tests: the fast suite, then the slow simulations.
pytest
pytest -m slow

# Optional: re-run the walkthrough notebook in place.
pip install -e ".[notebook]"
jupyter execute --inplace notebooks/walkthrough.ipynb
```

Python 3.11 or later. Seeds are fixed in `src/abkit/config.py`, so a re-run reproduces the
tables. `python -m abkit.pipeline run-all --fast --data tests/fixtures/hillstrom_fixture.csv
--out /tmp/abkit-demo` runs the whole pipeline on a small synthetic file without a download.
A short tour of the results is in [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb).

## 9. Repository structure

```text
src/abkit/            the package
  sql/                DuckDB metric layer: per-arm sufficient statistics, balance, segments
  data.py, validate.py        download with checksum, validation, cleaning
  eda.py, balance.py          sample ratio mismatch, balance, pooled description
  frequentist.py, readout.py  tests, intervals, multiplicity, the readout tables
  cuped.py                    CUPED and the Lin regression adjustment
  bayes.py                    Beta-Binomial rates, Bayesian bootstrap and two-part spend model
  power.py                    minimum detectable effects, sample sizes, power curves
  sequential.py               peeking simulation and the valid sequential rules
  segments.py                 exploratory segment effects, interactions, false discovery rate
  uplift/                     features, cross-fitting, learners, evaluation, policies
  pipeline.py                 command line, stages and their cache
  report.py, readme.py        results.json, tables and the generated blocks of this file
  results_schema.py           the schema results.json is checked against
docs/                 analysis plan (frozen, with deviations), methodology, data card, schema
reports/              results.json, tables, figures, validation and description reports
notebooks/            walkthrough.ipynb
tests/                unit, known-answer and simulation tests; a synthetic fixture
.github/workflows/    CI on every push; slow tests on manual trigger
```

## 10. References

- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and
  powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*,
  57(1), 289-300.
- Deng, A., Xu, Y., Kohavi, R. and Walker, T. (2013). Improving the sensitivity of online
  controlled experiments by utilizing pre-experiment data. *Proceedings of WSDM 2013*.
- Hillstrom, K. (2008). The MineThatData e-mail analytics and data mining challenge.
  *MineThatData* blog.
- Jennison, C. and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
  Clinical Trials*. Chapman and Hall/CRC.
- Johari, R., Koomen, P., Pekelis, L. and Walsh, D. (2017). Peeking at A/B tests: why it
  matters, and what to do about it. *Proceedings of KDD 2017*.
- Kohavi, R., Tang, D. and Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A
  Practical Guide to A/B Testing*. Cambridge University Press.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J. and Yu, B. (2019). Metalearners for estimating
  heterogeneous treatment effects using machine learning. *Proceedings of the National
  Academy of Sciences*, 116(10), 4156-4165.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data: reexamining
  Freedman's critique. *Annals of Applied Statistics*, 7(1), 295-318.
- O'Brien, P. C. and Fleming, T. R. (1979). A multiple testing procedure for clinical trials.
  *Biometrics*, 35(3), 549-556.
- Pocock, S. J. (1977). Group sequential methods in the design and analysis of clinical
  trials. *Biometrika*, 64(2), 191-199.
- Radcliffe, N. J. (2007). Using control groups to target on predicted lift: building and
  assessing uplift models. *Direct Marketing Analytics Journal*, 1, 14-21.
- Rubin, D. B. (1981). The Bayesian bootstrap. *Annals of Statistics*, 9(1), 130-134.

Released under the MIT licence ([LICENSE](LICENSE)).
