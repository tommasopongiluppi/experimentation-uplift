"""Schema of ``reports/results.json``: the keys each stage writes, their types and meaning.

The schema is the single source of truth for two things: the pipeline validates
``results.json`` against it at the end of every run, and ``docs/results_schema.md`` is
rendered from it (``python -m abkit.results_schema``), so the document cannot drift from the
check. Every number in the README and docs is read from the file this schema describes.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from abkit import config

logger = logging.getLogger(__name__)

DOC_PATH: Path = config.ROOT / "docs" / "results_schema.md"
#: Optional sections: present only when their stage has run.
OPTIONAL_SECTIONS: frozenset[str] = frozenset({"seed_robustness"})


class ResultsSchemaError(ValueError):
    """Raised when ``results.json`` does not match the schema."""


@dataclass(frozen=True)
class Field:
    """One key of a section.

    Attributes
    ----------
    type : str
        One of ``int``, ``float`` (an int is accepted), ``str``, ``bool``, ``dict``, ``list``,
        ``list[dict]``, ``list[str]``, ``list[float]``, ``list[int]``, ``dict[str, float]``,
        ``dict[str, bool]``, ``dict[str, int]`` or ``path`` (a string naming a file under the
        reports directory, which must exist). A ``|null`` suffix also allows ``None``.
    description : str
        What the value means, in plain language.
    fields : dict of str to Field, optional
        For a ``dict`` type: the required keys inside it.
    required : bool
        Whether the key must be present.
    values : dict of str to Field, optional
        For a ``dict`` type whose keys are free names (a family, a metric): the required keys
        inside every one of its values, each of which must itself be a dict.
    """

    type: str
    description: str
    fields: dict[str, "Field"] = field(default_factory=dict)
    required: bool = True
    values: dict[str, "Field"] = field(default_factory=dict)


def _f(type_: str, description: str, **fields: "Field") -> Field:
    return Field(type_, description, fields)


_ROWS = "One record per row of the table named by ``table``, with the same columns."
_TABLE = "Path of the CSV behind this section, relative to the reports directory."
_FIGURE = "Path of the PNG figure, relative to the reports directory."
_BOOTSTRAP = _f(
    "dict",
    "Bootstrap settings.",
    n_resamples=_f("int", "Number of bootstrap resamples."),
)

SCHEMA: dict[str, dict[str, Field]] = {
    "pipeline": {
        "mode": _f("str", "``full`` or ``fast`` (the sizes below say which was used)."),
        "data": _f("str", "``hillstrom`` for the real download, else the name of the CSV given."),
        "with_robustness": _f("bool", "Whether the seed-robustness stage was part of the run."),
        "stages": _f("list[str]", "Stages in the run, in dependency order."),
        "sizes": _f(
            "dict[str, int]",
            "Simulation, bootstrap and seed counts used by the stages (full or fast values).",
        ),
        "abkit_version": _f("str", "Package version that wrote the file."),
    },
    "validity": {
        "srm": _f(
            "dict",
            "Sample ratio mismatch test of the arm counts.",
            counts=_f("dict[str, int]", "Customers per arm."),
            expected=_f("dict[str, float]", "Expected customers per arm under equal allocation."),
            statistic=_f("float", "Chi-square statistic."),
            dof=_f("int", "Degrees of freedom."),
            p_value=_f("float", "p-value of the test."),
            alpha=_f("float", "Level below which the readout is blocked."),
            passed=_f("bool", "Whether the check passed."),
        ),
        "balance": _f(
            "dict",
            "Covariate balance of each email arm against control.",
            threshold=_f("float", "|SMD| above which a covariate level is flagged."),
            max_abs_smd=_f("float", "Largest absolute standardised mean difference."),
            max_abs_smd_variable=_f("str", "Variable with the largest |SMD|."),
            max_abs_smd_level=_f("str", "Level of that variable (empty for numeric)."),
            n_flagged=_f("int", "Number of levels above the threshold."),
            categorical_chi2=_f("list[dict]", "Chi-square tests of each categorical variable."),
        ),
    },
    "eda": {
        "pooled_rates": _f("dict[str, float]", "Pooled outcome means and shares over all arms."),
        "history_correlations": _f(
            "dict[str, float]", "Pearson correlation of history with each outcome, pooled."
        ),
    },
    "frequentist": {
        "alpha": _f("float", "Two-sided level."),
        "families": _f("dict", "Multiple-testing family -> adjustment method and test count."),
        "spend_winsorisation_cap": _f("float", "Cap applied in the winsorised sensitivity rows."),
        "bootstrap": _BOOTSTRAP,
        "table": _f("path", _TABLE),
        "figure": _f("path", _FIGURE),
        "rows": _f("list[dict]", _ROWS),
    },
    "cuped": {
        "covariate": _f("str", "Pre-experiment covariate used by CUPED."),
        "theta_pooling": _f("str", "How the CUPED coefficient is pooled."),
        "lin_covariates": _f("list[str]", "Design columns of the Lin regression adjustment."),
        "lin_standard_errors": _f("str", "Standard-error type of the Lin estimator."),
        "raw_estimator": _f("str", "The unadjusted estimator the adjusted ones are compared to."),
        "table": _f("path", _TABLE),
        "figure": _f("path", _FIGURE),
        "variance_reduction_by_metric": _f(
            "dict", "Metric -> min and max of rho and the variance reductions."
        ),
        "rows": _f("list[dict]", _ROWS),
    },
    "power": {
        "n_per_arm": _f("int", "Smallest arm size, used for the MDE at the actual sample."),
        "arm_counts": _f("dict[str, int]", "Customers per arm."),
        "power": _f("float", "Target power."),
        "alpha": _f("float", "Two-sided level."),
        "alpha_holm": _f("float", "Level per test under Holm for the primary family."),
        "n_primary_tests": _f("int", "Tests in the primary family."),
        "relative_lifts": _f("list[float]", "Relative lifts the sample-size table covers."),
        "baseline_arm": _f("str", "Arm whose mean and variance define the baseline."),
        "cuped_covariate": _f("str", "Covariate for the CUPED variants."),
        "metrics": _f("dict", "Metric -> baseline mean and variance."),
        "table": _f("path", _TABLE),
        "sample_size_table": _f("path", "CSV of required sample sizes per relative lift."),
        "curve_table": _f("path", "CSV of power against relative lift."),
        "figure": _f("path", _FIGURE),
        "mde_rows": _f(
            "list[dict]", "One record per row of the planning MDE table (control variance)."
        ),
        "mde_variance_note": _f("str", "Which variance each MDE table uses."),
        "mde_per_arm_rows": _f(
            "list[dict]", "MDE of each email against control with each arm's realised variance."
        ),
        "per_arm_table": _f("path", "CSV of the per-arm MDE table."),
        "sample_size_rows": _f("list[dict]", "One record per row of the sample-size table."),
    },
    "bayes_rates": {
        "model": _f("str", "Model description."),
        "n_draws": _f("int", "Posterior draws per arm and row."),
        "seed": _f("int", "Seed of the posterior draws."),
        "level": _f("float", "Credible level."),
        "priors": _f("list[dict]", "Prior label and strength for each prior tried."),
        "shared_prior_mean": _f("str", "Where the shared priors are centred."),
        "pooled_rates": _f("dict[str, float]", "Pooled rate per metric used as the prior mean."),
        "decision_rule": _f("dict", "Thresholds of the ship / do-not-ship rule."),
        "table": _f("path", _TABLE),
        "figure": _f("path", _FIGURE),
        "rows": _f("list[dict]", _ROWS),
    },
    "bayes_spend": {
        "primary": _f("str", "The primary spend model."),
        "sensitivity": _f("str", "The sensitivity analysis."),
        "cross_check": _f("str", "The parametric cross-check."),
        "n_draws": _f("int", "Posterior draws per arm and model."),
        "ppc_replicates": _f("int", "Replicate datasets in the posterior predictive check."),
        "seed": _f("int", "Seed of the posterior draws."),
        "level": _f("float", "Credible level."),
        "revenue_scale": _f("int", "Emails over which incremental revenue is also expressed."),
        "two_part_priors": _f("dict", "Priors of the two-part cross-check."),
        "spend_cap": _f("float", "Largest spend value in the data (top-coding)."),
        "spend_floor": _f("float", "Smallest positive spend value."),
        "decision_rule": _f("dict", "Thresholds of the ship / do-not-ship rule."),
        "recommendation": _f(
            "dict",
            "The email the Bayesian rule recommends.",
            qualifying_emails=_f("list[str]", "Emails that pass the rule against control."),
            recommended_email=_f("str|null", "The recommended email, null if none qualifies."),
            basis=_f("str", "Why it was chosen."),
            p_mens_better_than_womens=_f("float|null", "Posterior P(mens spend > womens spend)."),
        ),
        "table": _f("path", _TABLE),
        "ppc_table": _f("path", "CSV of the posterior predictive check."),
        "histogram_table": _f("path", "CSV of the binned posterior densities."),
        "figure": _f("path", _FIGURE),
        "rows": _f("list[dict]", _ROWS),
        "arms": _f("list[dict]", "Per-arm posterior summaries."),
        "ppc": _f("list[dict]", "One record per row of the posterior predictive check table."),
    },
    "sequential": {
        "days": _f("int", "Length of the simulated experiment in days."),
        "n_per_arm": _f("int", "Customers per arm at the end."),
        "customers_per_arm_per_day": _f("int", "Daily arrivals per arm."),
        "n_runs": _f("int", "Simulated experiments per scenario and metric."),
        "n_runs_looks": _f("int", "Simulated experiments per point of the looks table."),
        "n_paths": _f("int", "Brownian paths used for the boundary constants."),
        "max_looks": _f("int", "Largest number of looks in the looks table."),
        "alpha": _f("float", "Two-sided level."),
        "p_better_threshold": _f("float", "Posterior threshold of the Bayesian peeking rule."),
        "quadrature_nodes": _f("int", "Nodes of the numerical integration."),
        "baseline_arm": _f("str", "Arm supplying the baseline rates."),
        "baselines": _f("dict", "Metric -> baseline mean and variance."),
        "spend_positive_values": _f("int", "Distinct positive spend values resampled."),
        "effects": _f("dict", "Scenario -> simulated visit effect."),
        "tau": _f("dict", "mSPRT mixing variance per metric."),
        "tau_multipliers": _f("dict", "Sensitivity multipliers of tau."),
        "constants": _f("dict", "Boundary constants of each rule."),
        "published_k5_jennison_turnbull": _f("dict", "Published constants for the cross-check."),
        "seed": _f("int", "Simulation seed."),
        "table": _f("path", _TABLE),
        "looks_table": _f("path", "CSV of false positive rate by number of looks."),
        "by_day_table": _f("path", "CSV of cumulative rejection by day."),
        "boundaries_table": _f("path", "CSV of the rejection boundaries."),
        "figures": _f("dict", "Figure name -> PNG path."),
        "small_sample_check": _f("list[dict]", "FPR at the first look against the full sample."),
        "rows": _f("list[dict]", _ROWS),
    },
    "segments": {
        "label": _f("str", "Always ``exploratory``."),
        "plan_section": _f("str", "Analysis-plan section."),
        "deviation": _f("str", "Deviation from the plan, if any."),
        "alpha": _f("float", "Two-sided level."),
        "segment_variables": _f("dict", "Segment variable -> its levels."),
        "n_levels": _f("int", "Total number of segment levels."),
        "families": Field(
            "dict",
            "Family -> adjustment, test counts and significant counts before and after it.",
            values={
                "adjustment": _f("str", "Multiple-testing adjustment of the family (``bh``)."),
                "q": _f("float", "False discovery rate of the adjustment."),
                "alpha": _f("float", "Level of the uncorrected count."),
                "n_level_tests": _f("int", "Per-level tests in the family."),
                "n_interaction_tests": _f("int", "Interaction tests in the family."),
                "n_tests": _f("int", "All tests in the family."),
                "n_significant_raw": _f("int", "Tests with raw p <= alpha, before any correction."),
                "n_significant_fdr": _f("int", "Tests with Benjamini-Hochberg adjusted p <= q."),
            },
        ),
        "challenge_question": _f(
            "list[dict]", "Effect of each email among past buyers of its category."
        ),
        "fdr_simulation": _f("dict", "Settings and rows of the false-discovery simulation."),
        "table": _f("path", _TABLE),
        "interactions_table": _f("path", "CSV of the interaction tests."),
        "figure": _f("path", _FIGURE),
        "rows": _f("list[dict]", _ROWS),
        "interaction_rows": _f("list[dict]", "One record per row of the interactions table."),
    },
    "uplift_models": {
        "label": _f("str", "Section label."),
        "plan_section": _f("str", "Analysis-plan section."),
        "deviation": _f("str", "Deviations from the plan."),
        "seeds": _f("dict[str, int]", "Seeds of the cross-fitting and the bootstrap."),
        "n_folds": _f("int", "Cross-fitting folds."),
        "rankings": _f("list[str]", "All rankings evaluated."),
        "uplift_learners": _f("list[str]", "The uplift meta-learners."),
        "baselines": _f("list[str]", "The baseline rankings."),
        "outcomes": _f("list[str]", "Outcomes modelled."),
        "hyperparameter_grid": _f("list[dict]", "Grid searched on the inner split."),
        "fixed_params": _f("dict", "Base-model parameters held fixed."),
        "selection_metric": _f("str", "Metric that names the best ranking."),
        "uplift_at_k": _f("list[float]", "Shares at which uplift at k is reported."),
        "bootstrap": _BOOTSTRAP,
        "limitation": _f("str", "What the bootstrap intervals leave out."),
        "table": _f("path", _TABLE),
        "comparison_table": _f("path", "CSV of paired comparisons against the best ranking."),
        "scores_file": _f("str", "Name of the out-of-fold scores parquet under data/processed."),
        "summary": _f("dict", "Pair and outcome -> best ranking and what it is separated from."),
        "rows": _f("list[dict]", _ROWS),
        "comparison_rows": _f("list[dict]", "One record per row of the comparison table."),
    },
    "policy": {
        "label": _f("str", "Section label."),
        "plan_section": _f("str", "Analysis-plan section."),
        "deviation": _f("str", "Deviations from the plan."),
        "learner": _f("str", "Learner whose scores drive the uplift policy."),
        "learner_outcome": _f("str", "Outcome the scores were trained on."),
        "learner_reason": _f("str", "Why that learner."),
        "carry_over_note": _f("str", "Caveat on the conversion and spend values."),
        "ab_winner": _f(
            "dict",
            "The A/B winner email and the rule that chose it.",
            email=_f("str", "Winning email."),
            basis=_f("str", "Rule applied."),
        ),
        "budget": _f("int", "Email budget (scaled with the row count on a fixture)."),
        "cost_per_email": _f("float", "Cost per email used for the net value."),
        "assignment_probability": _f("float", "Known propensity per arm."),
        "n_customers": _f("int", "Customers in the data."),
        "seeds": _f("dict[str, int]", "Seeds of the random policy and the bootstrap."),
        "bootstrap": _BOOTSTRAP,
        "mens_constant_uplift": _f("float", "Average mens effect used by the exploratory policy."),
        "policies": _f("dict", "Policy name -> label."),
        "exploratory": _f("list[str]", "Policies that are exploratory."),
        "headline": _f("dict", "Outcome -> policy -> incremental total, interval and share."),
        "uplift_versus": _f("dict", "Outcome -> policy -> paired difference against uplift."),
        "uplift_beats_recency_rule": _f(
            "dict[str, bool]", "Outcome -> whether the paired difference excludes zero in favour."
        ),
        "uplift_beats_response_model": _f("dict[str, bool]", "Same, against the response model."),
        "uplift_beats_random": _f("dict[str, bool]", "Same, against the random draw."),
        "versus_random_expected": _f(
            "list[dict]",
            "Each budgeted policy against the exact expectation of random targeting, paired.",
        ),
        "recommendation": _f(
            "dict",
            "The decision rule of analysis plan D10 applied to the comparisons.",
            rule=_f("str", "The rule in words."),
            qualifying=_f("dict", "Ranking -> outcomes on which it beats the expectation."),
            recommended=_f("str|null", "Recommended ranking, or null if none qualifies."),
            basis=_f("str", "Why."),
        ),
        "arm_counts": _f(
            "list[dict]", "Realised-arm counts of each selected group with a sample-ratio test."
        ),
        "vs_random_table": _f("path", "CSV of the comparison with the random expectation."),
        "arm_counts_table": _f("path", "CSV of the arm counts."),
        "calibration": _f("list[dict]", "Level-calibration check of each email's scores."),
        "suppressed_group": _f("list[dict]", "Realised uplift of the do-not-email list."),
        "profiles": _f("list[dict]", "Pre-treatment profile of the targeted and suppressed."),
        "table": _f("path", _TABLE),
        "comparison_table": _f("path", "CSV of the paired policy comparisons."),
        "budget_curve_table": _f("path", "CSV of the budget curve."),
        "profiles_table": _f("path", "CSV of the profiles."),
        "calibration_table": _f("path", "CSV of the calibration check."),
        "suppressed_table": _f("path", "CSV of the suppressed group."),
        "figure": _f("path", _FIGURE),
        "rows": _f("list[dict]", _ROWS),
        "comparison_rows": _f("list[dict]", "One record per row of the comparison table."),
    },
    "seed_robustness": {
        "label": _f("str", "Section label."),
        "plan_section": _f("str", "Analysis-plan section."),
        "outcome": _f("str", "Outcome re-fitted (visit)."),
        "learners": _f("list[str]", "Rankings recorded per seed."),
        "seed_0": _f("str", "Where the committed seed's rows come from."),
        "other_seeds": _f("str", "How the other seeds are generated."),
        "budget": _f("int", "Email budget of the policy metric."),
        "ab_winner": _f("str", "A/B winner email."),
        "policy_metric": _f("str", "Name of the policy metric."),
        "note": _f("str", "What the spread adds to the bootstrap intervals."),
        "n_seeds": _f("int", "Seeds including the committed one."),
        "spread": _f("list[dict]", "Median, min and max of each metric across seeds."),
        "beats_random_expectation": _f(
            "dict",
            "Policy -> outcome -> in how many seeds its interval against the random "
            "expectation lies above zero (robustness reporting; not an input of the rule).",
        ),
        "x_beats_response_model_womens": _f(
            "dict", "In how many seeds the X-learner beats the response model on the womens pair."
        ),
        "table": _f("path", _TABLE),
        "rows": _f("list[dict]", _ROWS),
    },
}


def _check_type(value: Any, type_: str) -> bool:
    """Return whether ``value`` has the schema type ``type_``."""
    if type_.endswith("|null"):
        return value is None or _check_type(value, type_[: -len("|null")])
    if type_ == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_ == "float":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if type_ in ("str", "path"):
        return isinstance(value, str)
    if type_ == "bool":
        return isinstance(value, bool)
    if type_ == "dict":
        return isinstance(value, dict)
    if type_ == "list":
        return isinstance(value, list)
    if type_.startswith("list["):
        inner = type_[5:-1]
        return isinstance(value, list) and all(_check_type(v, inner) for v in value)
    if type_.startswith("dict[str, "):
        inner = type_[10:-1]
        return isinstance(value, dict) and all(
            isinstance(k, str) and _check_type(v, inner) for k, v in value.items()
        )
    raise ValueError(f"Unknown schema type {type_!r}")


def _validate_fields(
    data: dict[str, Any], fields: dict[str, Field], prefix: str, reports_dir: Path | None
) -> list[str]:
    problems = []
    for name, spec in fields.items():
        key = f"{prefix}.{name}"
        if name not in data:
            if spec.required:
                problems.append(f"{key}: missing")
            continue
        value = data[name]
        if not _check_type(value, spec.type):
            problems.append(f"{key}: expected {spec.type}, got {type(value).__name__}")
            continue
        if spec.type == "path" and reports_dir is not None and not (reports_dir / value).exists():
            problems.append(f"{key}: file {value!r} not found under {reports_dir}")
        if spec.fields and isinstance(value, dict):
            problems.extend(_validate_fields(value, spec.fields, key, reports_dir))
        if spec.values and isinstance(value, dict):
            for name_, item in value.items():
                if isinstance(item, dict):
                    problems.extend(
                        _validate_fields(item, spec.values, f"{key}.{name_}", reports_dir)
                    )
                else:
                    problems.append(f"{key}.{name_}: expected dict, got {type(item).__name__}")
    return problems


def validate_results(
    results: dict[str, Any],
    reports_dir: Path | None = None,
    required_sections: frozenset[str] | None = None,
) -> list[str]:
    """Check a loaded ``results.json`` against the schema.

    Parameters
    ----------
    results : dict
        Parsed results.json.
    reports_dir : Path, optional
        When given, every ``path`` field must name an existing file under it.
    required_sections : frozenset of str, optional
        Sections that must be present; defaults to every non-optional section. Optional
        sections are validated when present.

    Returns
    -------
    list of str
        Problems found, empty when the file matches the schema.
    """
    if required_sections is None:
        required_sections = frozenset(SCHEMA) - OPTIONAL_SECTIONS
    problems = []
    for section in sorted(required_sections):
        if section not in results:
            problems.append(f"{section}: section missing")
    for section, data in results.items():
        if section not in SCHEMA:
            problems.append(f"{section}: unknown section")
        elif not isinstance(data, dict):
            problems.append(f"{section}: expected dict, got {type(data).__name__}")
        else:
            problems.extend(_validate_fields(data, SCHEMA[section], section, reports_dir))
    return problems


def check_results(
    path: Path = config.RESULTS_JSON, required_sections: frozenset[str] | None = None
) -> dict[str, Any]:
    """Load ``results.json``, validate it and return it.

    Parameters
    ----------
    path : Path
        Location of results.json; referenced tables and figures are resolved from its parent.
    required_sections : frozenset of str, optional
        See :func:`validate_results`.

    Returns
    -------
    dict
        The parsed file.

    Raises
    ------
    ResultsSchemaError
        Listing every problem found.
    """
    results = json.loads(path.read_text())
    problems = validate_results(results, path.parent, required_sections)
    if problems:
        raise ResultsSchemaError(
            f"{path} does not match docs/results_schema.md ({len(problems)} problems):\n  "
            + "\n  ".join(problems)
        )
    logger.info("%s matches the schema (%d sections).", path, len(results))
    return results


def _render_fields(fields: dict[str, Field], prefix: str = "") -> list[str]:
    lines = []
    for name, spec in fields.items():
        req = "" if spec.required else " (optional)"
        lines.append(f"| `{prefix}{name}` | `{spec.type}`{req} | {spec.description} |")
        if spec.fields:
            lines.extend(_render_fields(spec.fields, f"{prefix}{name}."))
        if spec.values:
            lines.extend(_render_fields(spec.values, f"{prefix}{name}.<name>."))
    return lines


def render_markdown() -> str:
    """Render the schema as the Markdown document ``docs/results_schema.md``."""
    out = [
        "# Structure of `reports/results.json`",
        "",
        "Generated from `src/abkit/results_schema.py` by `python -m abkit.results_schema`;",
        "edit the schema there, not this file. The pipeline validates `results.json` against",
        "it at the end of every run (`python -m abkit.pipeline report`): every required",
        "section and key must be present with the listed type, and every `path` must name an",
        "existing file under `reports/`. Extra keys inside a section are allowed; unknown",
        "top-level sections are not.",
        "",
        "The file is a JSON object with one top-level key per pipeline stage, sorted by name.",
        "Each stage replaces its own section and leaves the others untouched, so the file",
        "accumulates across stages. A `float` accepts an integer. `NaN` and infinity are",
        "written as `null`. Every number quoted in the README and docs is read from this",
        "file or from the CSV tables it names.",
        "",
        "## Sections",
        "",
        "| Section | Written by | Required |",
        "|---|---|---|",
    ]
    writers = {
        "pipeline": "report",
        "validity": "checks",
        "eda": "checks",
        "frequentist": "frequentist",
        "cuped": "cuped",
        "power": "power",
        "bayes_rates": "bayes",
        "bayes_spend": "bayes",
        "sequential": "sequential",
        "segments": "segments",
        "uplift_models": "uplift",
        "policy": "policy",
        "seed_robustness": "robustness",
    }
    for section in SCHEMA:
        required = "no (only with `--with-robustness`)" if section in OPTIONAL_SECTIONS else "yes"
        out.append(f"| `{section}` | `{writers[section]}` | {required} |")
    for section, fields in SCHEMA.items():
        out += ["", f"## `{section}`", "", "| Key | Type | Meaning |", "|---|---|---|"]
        out += _render_fields(fields)
    return "\n".join(out) + "\n"


def write_doc(path: Path = DOC_PATH) -> Path:
    """Write :func:`render_markdown` to ``path`` and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown())
    return path


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    logging.basicConfig(level=logging.INFO)
    sys.stdout.write(f"wrote {write_doc()}\n")
