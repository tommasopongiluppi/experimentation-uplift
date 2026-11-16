# Structure of `reports/results.json`

Generated from `src/abkit/results_schema.py` by `python -m abkit.results_schema`;
edit the schema there, not this file. The pipeline validates `results.json` against
it at the end of every run (`python -m abkit.pipeline report`): every required
section and key must be present with the listed type, and every `path` must name an
existing file under `reports/`. Extra keys inside a section are allowed; unknown
top-level sections are not.

The file is a JSON object with one top-level key per pipeline stage, sorted by name.
Each stage replaces its own section and leaves the others untouched, so the file
accumulates across stages. A `float` accepts an integer. `NaN` and infinity are
written as `null`. Every number quoted in the README, docs and app is read from this
file or from the CSV tables it names.

## Sections

| Section | Written by | Required |
|---|---|---|
| `pipeline` | `report` | yes |
| `validity` | `checks` | yes |
| `eda` | `checks` | yes |
| `frequentist` | `frequentist` | yes |
| `cuped` | `cuped` | yes |
| `power` | `power` | yes |
| `bayes_rates` | `bayes` | yes |
| `bayes_spend` | `bayes` | yes |
| `sequential` | `sequential` | yes |
| `segments` | `segments` | yes |
| `uplift_models` | `uplift` | yes |
| `policy` | `policy` | yes |
| `seed_robustness` | `robustness` | no (only with `--with-robustness`) |

## `pipeline`

| Key | Type | Meaning |
|---|---|---|
| `mode` | `str` | ``full`` or ``fast`` (the sizes below say which was used). |
| `data` | `str` | ``hillstrom`` for the real download, else the name of the CSV given. |
| `with_robustness` | `bool` | Whether the seed-robustness stage was part of the run. |
| `stages` | `list[str]` | Stages in the run, in dependency order. |
| `sizes` | `dict[str, int]` | Simulation, bootstrap and seed counts used by the stages (full or fast values). |
| `abkit_version` | `str` | Package version that wrote the file. |

## `validity`

| Key | Type | Meaning |
|---|---|---|
| `srm` | `dict` | Sample ratio mismatch test of the arm counts. |
| `srm.counts` | `dict[str, int]` | Customers per arm. |
| `srm.expected` | `dict[str, float]` | Expected customers per arm under equal allocation. |
| `srm.statistic` | `float` | Chi-square statistic. |
| `srm.dof` | `int` | Degrees of freedom. |
| `srm.p_value` | `float` | p-value of the test. |
| `srm.alpha` | `float` | Level below which the readout is blocked. |
| `srm.passed` | `bool` | Whether the check passed. |
| `balance` | `dict` | Covariate balance of each email arm against control. |
| `balance.threshold` | `float` | |SMD| above which a covariate level is flagged. |
| `balance.max_abs_smd` | `float` | Largest absolute standardised mean difference. |
| `balance.max_abs_smd_variable` | `str` | Variable with the largest |SMD|. |
| `balance.max_abs_smd_level` | `str` | Level of that variable (empty for numeric). |
| `balance.n_flagged` | `int` | Number of levels above the threshold. |
| `balance.categorical_chi2` | `list[dict]` | Chi-square tests of each categorical variable. |

## `eda`

| Key | Type | Meaning |
|---|---|---|
| `pooled_rates` | `dict[str, float]` | Pooled outcome means and shares over all arms. |
| `history_correlations` | `dict[str, float]` | Pearson correlation of history with each outcome, pooled. |

## `frequentist`

| Key | Type | Meaning |
|---|---|---|
| `alpha` | `float` | Two-sided level. |
| `families` | `dict` | Multiple-testing family -> adjustment method and test count. |
| `spend_winsorisation_cap` | `float` | Cap applied in the winsorised sensitivity rows. |
| `bootstrap` | `dict` | Bootstrap settings. |
| `bootstrap.n_resamples` | `int` | Number of bootstrap resamples. |
| `table` | `path` | Path of the CSV behind this section, relative to the reports directory. |
| `figure` | `path` | Path of the PNG figure, relative to the reports directory. |
| `rows` | `list[dict]` | One record per row of the table named by ``table``, with the same columns. |

## `cuped`

| Key | Type | Meaning |
|---|---|---|
| `covariate` | `str` | Pre-experiment covariate used by CUPED. |
| `theta_pooling` | `str` | How the CUPED coefficient is pooled. |
| `lin_covariates` | `list[str]` | Design columns of the Lin regression adjustment. |
| `lin_standard_errors` | `str` | Standard-error type of the Lin estimator. |
| `raw_estimator` | `str` | The unadjusted estimator the adjusted ones are compared to. |
| `table` | `path` | Path of the CSV behind this section, relative to the reports directory. |
| `figure` | `path` | Path of the PNG figure, relative to the reports directory. |
| `variance_reduction_by_metric` | `dict` | Metric -> min and max of rho and the variance reductions. |
| `rows` | `list[dict]` | One record per row of the table named by ``table``, with the same columns. |

## `power`

| Key | Type | Meaning |
|---|---|---|
| `n_per_arm` | `int` | Smallest arm size, used for the MDE at the actual sample. |
| `arm_counts` | `dict[str, int]` | Customers per arm. |
| `power` | `float` | Target power. |
| `alpha` | `float` | Two-sided level. |
| `alpha_holm` | `float` | Level per test under Holm for the primary family. |
| `n_primary_tests` | `int` | Tests in the primary family. |
| `relative_lifts` | `list[float]` | Relative lifts the sample-size table covers. |
| `baseline_arm` | `str` | Arm whose mean and variance define the baseline. |
| `cuped_covariate` | `str` | Covariate for the CUPED variants. |
| `metrics` | `dict` | Metric -> baseline mean and variance. |
| `table` | `path` | Path of the CSV behind this section, relative to the reports directory. |
| `sample_size_table` | `path` | CSV of required sample sizes per relative lift. |
| `curve_table` | `path` | CSV of power against relative lift. |
| `figure` | `path` | Path of the PNG figure, relative to the reports directory. |
| `mde_rows` | `list[dict]` | One record per row of the MDE table. |
| `sample_size_rows` | `list[dict]` | One record per row of the sample-size table. |

## `bayes_rates`

| Key | Type | Meaning |
|---|---|---|
| `model` | `str` | Model description. |
| `n_draws` | `int` | Posterior draws per arm and row. |
| `seed` | `int` | Seed of the posterior draws. |
| `level` | `float` | Credible level. |
| `priors` | `list[dict]` | Prior label and strength for each prior tried. |
| `shared_prior_mean` | `str` | Where the shared priors are centred. |
| `pooled_rates` | `dict[str, float]` | Pooled rate per metric used as the prior mean. |
| `decision_rule` | `dict` | Thresholds of the ship / do-not-ship rule. |
| `table` | `path` | Path of the CSV behind this section, relative to the reports directory. |
| `figure` | `path` | Path of the PNG figure, relative to the reports directory. |
| `rows` | `list[dict]` | One record per row of the table named by ``table``, with the same columns. |

## `bayes_spend`

| Key | Type | Meaning |
|---|---|---|
| `primary` | `str` | The primary spend model. |
| `sensitivity` | `str` | The sensitivity analysis. |
| `cross_check` | `str` | The parametric cross-check. |
| `n_draws` | `int` | Posterior draws per arm and model. |
| `ppc_replicates` | `int` | Replicate datasets in the posterior predictive check. |
| `seed` | `int` | Seed of the posterior draws. |
| `level` | `float` | Credible level. |
| `revenue_scale` | `int` | Emails over which incremental revenue is also expressed. |
| `two_part_priors` | `dict` | Priors of the two-part cross-check. |
| `spend_cap` | `float` | Largest spend value in the data (top-coding). |
| `spend_floor` | `float` | Smallest positive spend value. |
| `decision_rule` | `dict` | Thresholds of the ship / do-not-ship rule. |
| `recommendation` | `dict` | The email the Bayesian rule recommends. |
| `recommendation.qualifying_emails` | `list[str]` | Emails that pass the rule against control. |
| `recommendation.recommended_email` | `str|null` | The recommended email, null if none qualifies. |
| `recommendation.basis` | `str` | Why it was chosen. |
| `recommendation.p_mens_better_than_womens` | `float|null` | Posterior P(mens spend > womens spend). |
| `table` | `path` | Path of the CSV behind this section, relative to the reports directory. |
| `ppc_table` | `path` | CSV of the posterior predictive check. |
| `histogram_table` | `path` | CSV of the binned posterior densities. |
| `figure` | `path` | Path of the PNG figure, relative to the reports directory. |
| `rows` | `list[dict]` | One record per row of the table named by ``table``, with the same columns. |
| `arms` | `list[dict]` | Per-arm posterior summaries. |
| `ppc` | `list[dict]` | One record per row of the posterior predictive check table. |

## `sequential`

| Key | Type | Meaning |
|---|---|---|
| `days` | `int` | Length of the simulated experiment in days. |
| `n_per_arm` | `int` | Customers per arm at the end. |
| `customers_per_arm_per_day` | `int` | Daily arrivals per arm. |
| `n_runs` | `int` | Simulated experiments per scenario and metric. |
| `n_runs_looks` | `int` | Simulated experiments per point of the looks table. |
| `n_paths` | `int` | Brownian paths used for the boundary constants. |
| `max_looks` | `int` | Largest number of looks in the looks table. |
| `alpha` | `float` | Two-sided level. |
| `p_better_threshold` | `float` | Posterior threshold of the Bayesian peeking rule. |
| `quadrature_nodes` | `int` | Nodes of the numerical integration. |
| `baseline_arm` | `str` | Arm supplying the baseline rates. |
| `baselines` | `dict` | Metric -> baseline mean and variance. |
| `spend_positive_values` | `int` | Distinct positive spend values resampled. |
| `effects` | `dict` | Scenario -> simulated visit effect. |
| `tau` | `dict` | mSPRT mixing variance per metric. |
| `tau_multipliers` | `dict` | Sensitivity multipliers of tau. |
| `constants` | `dict` | Boundary constants of each rule. |
| `published_k5_jennison_turnbull` | `dict` | Published constants for the cross-check. |
| `seed` | `int` | Simulation seed. |
| `table` | `path` | Path of the CSV behind this section, relative to the reports directory. |
| `looks_table` | `path` | CSV of false positive rate by number of looks. |
| `by_day_table` | `path` | CSV of cumulative rejection by day. |
| `boundaries_table` | `path` | CSV of the rejection boundaries. |
| `figures` | `dict` | Figure name -> PNG path. |
| `small_sample_check` | `list[dict]` | FPR at the first look against the full sample. |
| `rows` | `list[dict]` | One record per row of the table named by ``table``, with the same columns. |

## `segments`

| Key | Type | Meaning |
|---|---|---|
| `label` | `str` | Always ``exploratory``. |
| `plan_section` | `str` | Analysis-plan section. |
| `deviation` | `str` | Deviation from the plan, if any. |
| `alpha` | `float` | Two-sided level. |
| `segment_variables` | `dict` | Segment variable -> its levels. |
| `n_levels` | `int` | Total number of segment levels. |
| `families` | `dict` | Family -> adjustment, test counts and significant counts before and after it. |
| `families.<name>.adjustment` | `str` | Multiple-testing adjustment of the family (``bh``). |
| `families.<name>.q` | `float` | False discovery rate of the adjustment. |
| `families.<name>.alpha` | `float` | Level of the uncorrected count. |
| `families.<name>.n_level_tests` | `int` | Per-level tests in the family. |
| `families.<name>.n_interaction_tests` | `int` | Interaction tests in the family. |
| `families.<name>.n_tests` | `int` | All tests in the family. |
| `families.<name>.n_significant_raw` | `int` | Tests with raw p <= alpha, before any correction. |
| `families.<name>.n_significant_fdr` | `int` | Tests with Benjamini-Hochberg adjusted p <= q. |
| `challenge_question` | `list[dict]` | Effect of each email among past buyers of its category. |
| `fdr_simulation` | `dict` | Settings and rows of the false-discovery simulation. |
| `table` | `path` | Path of the CSV behind this section, relative to the reports directory. |
| `interactions_table` | `path` | CSV of the interaction tests. |
| `figure` | `path` | Path of the PNG figure, relative to the reports directory. |
| `rows` | `list[dict]` | One record per row of the table named by ``table``, with the same columns. |
| `interaction_rows` | `list[dict]` | One record per row of the interactions table. |

## `uplift_models`

| Key | Type | Meaning |
|---|---|---|
| `label` | `str` | Section label. |
| `plan_section` | `str` | Analysis-plan section. |
| `deviation` | `str` | Deviations from the plan. |
| `seeds` | `dict[str, int]` | Seeds of the cross-fitting and the bootstrap. |
| `n_folds` | `int` | Cross-fitting folds. |
| `rankings` | `list[str]` | All rankings evaluated. |
| `uplift_learners` | `list[str]` | The uplift meta-learners. |
| `baselines` | `list[str]` | The baseline rankings. |
| `outcomes` | `list[str]` | Outcomes modelled. |
| `hyperparameter_grid` | `list[dict]` | Grid searched on the inner split. |
| `fixed_params` | `dict` | Base-model parameters held fixed. |
| `selection_metric` | `str` | Metric that names the best ranking. |
| `uplift_at_k` | `list[float]` | Shares at which uplift at k is reported. |
| `bootstrap` | `dict` | Bootstrap settings. |
| `bootstrap.n_resamples` | `int` | Number of bootstrap resamples. |
| `limitation` | `str` | What the bootstrap intervals leave out. |
| `table` | `path` | Path of the CSV behind this section, relative to the reports directory. |
| `comparison_table` | `path` | CSV of paired comparisons against the best ranking. |
| `scores_file` | `str` | Name of the out-of-fold scores parquet under data/processed. |
| `summary` | `dict` | Pair and outcome -> best ranking and what it is separated from. |
| `rows` | `list[dict]` | One record per row of the table named by ``table``, with the same columns. |
| `comparison_rows` | `list[dict]` | One record per row of the comparison table. |

## `policy`

| Key | Type | Meaning |
|---|---|---|
| `label` | `str` | Section label. |
| `plan_section` | `str` | Analysis-plan section. |
| `deviation` | `str` | Deviations from the plan. |
| `learner` | `str` | Learner whose scores drive the uplift policy. |
| `learner_outcome` | `str` | Outcome the scores were trained on. |
| `learner_reason` | `str` | Why that learner. |
| `carry_over_note` | `str` | Caveat on the conversion and spend values. |
| `ab_winner` | `dict` | The A/B winner email and the rule that chose it. |
| `ab_winner.email` | `str` | Winning email. |
| `ab_winner.basis` | `str` | Rule applied. |
| `budget` | `int` | Email budget (scaled with the row count on a fixture). |
| `cost_per_email` | `float` | Cost per email used for the net value. |
| `assignment_probability` | `float` | Known propensity per arm. |
| `n_customers` | `int` | Customers in the data. |
| `seeds` | `dict[str, int]` | Seeds of the random policy and the bootstrap. |
| `bootstrap` | `dict` | Bootstrap settings. |
| `bootstrap.n_resamples` | `int` | Number of bootstrap resamples. |
| `mens_constant_uplift` | `float` | Average mens effect used by the exploratory policy. |
| `policies` | `dict` | Policy name -> label. |
| `exploratory` | `list[str]` | Policies that are exploratory. |
| `headline` | `dict` | Outcome -> policy -> incremental total, interval and share. |
| `uplift_versus` | `dict` | Outcome -> policy -> paired difference against uplift. |
| `uplift_beats_recency_rule` | `dict[str, bool]` | Outcome -> whether the paired difference excludes zero in favour. |
| `uplift_beats_response_model` | `dict[str, bool]` | Same, against the response model. |
| `uplift_beats_random` | `dict[str, bool]` | Same, against random targeting. |
| `calibration` | `list[dict]` | Level-calibration check of each email's scores. |
| `suppressed_group` | `list[dict]` | Realised uplift of the do-not-email list. |
| `profiles` | `list[dict]` | Pre-treatment profile of the targeted and suppressed. |
| `table` | `path` | Path of the CSV behind this section, relative to the reports directory. |
| `comparison_table` | `path` | CSV of the paired policy comparisons. |
| `budget_curve_table` | `path` | CSV of the budget curve. |
| `profiles_table` | `path` | CSV of the profiles. |
| `calibration_table` | `path` | CSV of the calibration check. |
| `suppressed_table` | `path` | CSV of the suppressed group. |
| `figure` | `path` | Path of the PNG figure, relative to the reports directory. |
| `rows` | `list[dict]` | One record per row of the table named by ``table``, with the same columns. |
| `comparison_rows` | `list[dict]` | One record per row of the comparison table. |

## `seed_robustness`

| Key | Type | Meaning |
|---|---|---|
| `label` | `str` | Section label. |
| `plan_section` | `str` | Analysis-plan section. |
| `outcome` | `str` | Outcome re-fitted (visit). |
| `learners` | `list[str]` | Rankings recorded per seed. |
| `seed_0` | `str` | Where the committed seed's rows come from. |
| `other_seeds` | `str` | How the other seeds are generated. |
| `budget` | `int` | Email budget of the policy metric. |
| `ab_winner` | `str` | A/B winner email. |
| `policy_metric` | `str` | Name of the policy metric. |
| `note` | `str` | What the spread adds to the bootstrap intervals. |
| `n_seeds` | `int` | Seeds including the committed one. |
| `spread` | `list[dict]` | Median, min and max of each metric across seeds. |
| `x_beats_response_model_womens` | `dict` | In how many seeds the X-learner beats the response model on the womens pair. |
| `table` | `path` | Path of the CSV behind this section, relative to the reports directory. |
| `rows` | `list[dict]` | One record per row of the table named by ``table``, with the same columns. |
