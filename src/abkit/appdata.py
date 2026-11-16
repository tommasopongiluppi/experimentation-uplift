"""Data access, formatting and the live peeking simulation behind the Streamlit app.

The app is a reader of the committed artifacts: everything it shows comes from
``reports/results.json`` and the CSV tables that file names, or from a simulation it runs on
the spot with :mod:`abkit.sequential`. Nothing here opens the raw data or the DuckDB file, so
the app can be deployed from the repository alone.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from abkit import config, plots, sequential
from abkit.power import mde

REPO_URL: str = "https://github.com/tommasopongiluppi/experimentation-uplift"

ARM_LABELS: dict[str, str] = {"mens": "Mens email", "womens": "Womens email", "control": "No email"}
COMPARISON_LABELS: dict[str, str] = {
    "mens_vs_control": "Mens email vs no email",
    "womens_vs_control": "Womens email vs no email",
    "mens_vs_womens": "Mens email vs Womens email",
}
METRIC_LABELS: dict[str, str] = {
    "spend": "Spend per customer",
    "conversion": "Conversion rate",
    "visit": "Visit rate",
    "spend_winsorised": "Spend per customer, winsorised (sensitivity)",
}
RATE_METRICS: tuple[str, ...] = ("conversion", "visit")
ADJUSTMENT_LABELS: dict[str, str] = {
    "holm": "Holm",
    "bh": "Benjamini-Hochberg",
    "none": "no adjustment (family of one)",
}
POLICY_LABELS: dict[str, str] = {
    "uplift": "Uplift policy",
    "response_model": "Response model",
    "recency_rule": "Recency rule",
    "random": "Random targeting",
}
POLICY_COLORS: dict[str, str] = {k: v[0] for k, v in plots.POLICY_STYLES.items()}
PEEKING_METHODS: dict[str, str] = {
    "naive": f"Naive peeking (stop at the first p < {config.ALPHA})",
    "fixed": "Fixed horizon (test once at the end)",
    "bayes": (
        "Bayesian peeking (stop when P(better) leaves "
        f"{100 * (1 - config.BAYES_P_BETTER):.0f}%-{100 * config.BAYES_P_BETTER:.0f}%)"
    ),
    "pocock": "Pocock boundary",
    "obf": "O'Brien-Fleming boundary",
    "msprt": "mSPRT (always-valid p-value)",
}

#: Caps of the live simulation, sized so one run answers within about two seconds.
LIVE_RUNS: int = 10_000
#: The Bayesian rule integrates a posterior for every run and day, so its runs are capped by
#: the number of such evaluations.
LIVE_BAYES_EVALUATIONS: int = 24_000
LIVE_PATHS: int = 50_000
LIVE_MAX_DAYS: int = sequential.MAX_LOOKS
LIVE_MAX_USERS_PER_DAY: int = 50_000


# ---------------------------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------------------------


def reports_dir() -> Path:
    """Return the reports directory holding ``results.json``.

    The package's own ``reports/`` is used when it is there (a checkout or an editable
    install); otherwise ``reports/`` under the working directory, which is where a hosted app
    started from the repository root finds it.

    Returns
    -------
    Path
        The directory.

    Raises
    ------
    FileNotFoundError
        If neither location holds ``results.json``.
    """
    candidates = (config.REPORTS, Path.cwd() / "reports")
    for candidate in candidates:
        if (candidate / "results.json").is_file():
            return candidate
    raise FileNotFoundError("results.json not found in " + " or ".join(str(c) for c in candidates))


@st.cache_data(show_spinner=False)
def load_results() -> dict[str, Any]:
    """Load ``reports/results.json`` (cached for the life of the app process).

    Returns
    -------
    dict
        The parsed file, one key per pipeline stage.
    """
    return json.loads((reports_dir() / "results.json").read_text())


@st.cache_data(show_spinner=False)
def load_table(relative_path: str) -> pd.DataFrame:
    """Load one CSV table named by ``results.json`` (cached).

    Parameters
    ----------
    relative_path : str
        Path under the reports directory, as stored in results.json
        (for example ``tables/policy_budget_curve.csv``).

    Returns
    -------
    pandas.DataFrame
        The table.
    """
    return pd.read_csv(reports_dir() / relative_path)


def find_row(rows: list[dict[str, Any]], **where: Any) -> dict[str, Any]:
    """Return the single record of ``rows`` matching every ``key=value`` condition.

    Parameters
    ----------
    rows : list of dict
        Records from a ``rows`` list of results.json.
    **where
        Conditions every field must satisfy.

    Returns
    -------
    dict
        The matching record.

    Raises
    ------
    LookupError
        If no record or more than one matches, so a page can never show an ambiguous number.
    """
    found = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    if len(found) != 1:
        raise LookupError(f"Expected one row for {where}, found {len(found)}.")
    return found[0]


def doc_url(path: str, anchor: str = "") -> str:
    """Link to a file of the repository on its default branch.

    Parameters
    ----------
    path : str
        Path from the repository root, for example ``docs/methodology.md``.
    anchor : str
        Heading anchor without the ``#``.

    Returns
    -------
    str
        The URL.
    """
    return f"{REPO_URL}/blob/main/{path}" + (f"#{anchor}" if anchor else "")


# ---------------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------------


def format_p(p: float) -> str:
    """Format a p-value: three decimals, scientific notation below 0.001."""
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def format_probability(p: float) -> str:
    """Format a probability as a percentage, never showing 0% or 100%.

    A probability estimated from a finite number of draws is not a certainty, so values that
    would round to the ends are shown as bounds.
    """
    if p >= 0.9995:
        return "> 99.9%"
    if p <= 0.0005:
        return "< 0.1%"
    return f"{100 * p:.1f}%"


def format_effect(value: float, metric: str, signed: bool = True) -> str:
    """Format an absolute effect: dollars for spend, percentage points for rates.

    Parameters
    ----------
    value : float
        Difference between two arms (dollars per customer, or a difference in proportions).
    metric : str
        Metric name; ``conversion`` and ``visit`` are rates.
    signed : bool
        Whether to show a leading plus sign on positive values.

    Returns
    -------
    str
        For example ``+$0.77`` or ``+7.66 pp``.
    """
    sign = "+" if signed and value > 0 else ("-" if value < 0 else "")
    if metric in RATE_METRICS:
        return f"{sign}{abs(100 * value):.2f} pp"
    return f"{sign}${abs(value):,.2f}"


def format_level(value: float, metric: str) -> str:
    """Format an arm's level: dollars for spend, a percentage for rates."""
    return f"{100 * value:.2f}%" if metric in RATE_METRICS else f"${value:,.2f}"


def format_interval(low: float, high: float, metric: str) -> str:
    """Format a confidence or credible interval of an effect."""
    return f"{format_effect(low, metric)} to {format_effect(high, metric)}"


def format_small(value: float, metric: str) -> str:
    """Format a small non-negative quantity (an expected loss or its threshold) in metric units.

    Parameters
    ----------
    value : float
        Dollars per customer for spend, a difference in proportions for rates.
    metric : str
        Metric name.

    Returns
    -------
    str
        Two significant figures, for example ``$0.0065`` or ``0.0088 pp``.
    """
    if metric in RATE_METRICS:
        return f"{100 * value:.2g} pp"
    return f"${value:.2g}"


def display_scale(metric: str) -> tuple[float, str]:
    """Multiplier and axis unit that turn an effect into its display units.

    Returns
    -------
    tuple of (float, str)
        ``(100, "percentage points")`` for rates, ``(1, "dollars per customer")`` for spend.
    """
    if metric in RATE_METRICS:
        return 100.0, "percentage points"
    return 1.0, "dollars per customer"


def heading_anchor(heading: str) -> str:
    """Anchor GitHub gives a Markdown heading: lower case, punctuation dropped, hyphens for spaces.

    Parameters
    ----------
    heading : str
        Heading text without the leading ``#`` marks.

    Returns
    -------
    str
        The anchor, without the ``#``.
    """
    kept = "".join(c for c in heading.lower() if c.isalnum() or c in " -_")
    return kept.replace(" ", "-")


# ---------------------------------------------------------------------------------------------
# Page chrome and chart style
# ---------------------------------------------------------------------------------------------


def page_setup(title: str) -> None:
    """Configure a page: browser title, centred single-column layout, then the page title.

    Parameters
    ----------
    title : str
        Page title.
    """
    st.set_page_config(page_title=f"{title} | Hillstrom email experiment", layout="centered")
    st.title(title)


def chart_style(chart: Any) -> Any:
    """Apply the README figures' look to an Altair chart: same inks, grid and no frame.

    Parameters
    ----------
    chart : altair.Chart
        Any Altair chart.

    Returns
    -------
    altair.Chart
        The configured chart.
    """
    return (
        chart.configure(background=plots.SURFACE)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=plots.GRID,
            domainColor=plots.GRID,
            tickColor=plots.GRID,
            labelColor=plots.TEXT_SECONDARY,
            titleColor=plots.TEXT_SECONDARY,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=plots.TEXT,
            titleColor=plots.TEXT_SECONDARY,
            orient="bottom",
            labelFontSize=12,
            labelLimit=260,
        )
    )


# ---------------------------------------------------------------------------------------------
# Live peeking simulation
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PeekingResult:
    """Outcome of one live peeking simulation.

    Attributes
    ----------
    method : str
        Decision rule applied.
    n_runs : int
        Simulated experiments.
    n_days : int
        Looks, one per day.
    n_per_arm_per_day : int
        Customers added to each arm every day.
    rate_control, rate_treatment : float
        True rates of the two arms.
    reject_rate, reject_mcse : float
        Share of experiments declared significant and its Monte Carlo standard error. It is
        the false positive rate when the two rates are equal and the power otherwise.
    mean_stop_day, mean_stop_day_se : float
        Average day the experiment ends (the last day when it never rejects).
    stop_day_share : numpy.ndarray
        Share of experiments stopping with a rejection on each day, shape ``(n_days,)``.
    never_share : float
        Share that never rejects; with ``stop_day_share`` it sums to one.
    alpha : float
        Nominal two-sided level.
    """

    method: str
    n_runs: int
    n_days: int
    n_per_arm_per_day: int
    rate_control: float
    rate_treatment: float
    reject_rate: float
    reject_mcse: float
    mean_stop_day: float
    mean_stop_day_se: float
    stop_day_share: np.ndarray
    never_share: float
    alpha: float

    @property
    def is_null(self) -> bool:
        """Whether the two arms share one rate, so every rejection is a false positive."""
        return self.rate_treatment == self.rate_control


def simulate_peeking(
    baseline_rate: float,
    relative_lift: float,
    users_per_day: int,
    n_days: int,
    method: str,
    rng: np.random.Generator,
    n_runs: int = LIVE_RUNS,
    n_paths: int = LIVE_PATHS,
    alpha: float = config.ALPHA,
) -> PeekingResult:
    """Simulate an A/B test checked once a day under one stopping rule.

    A thin wrapper over :mod:`abkit.sequential`: the experiments come from
    :func:`~abkit.sequential.simulate_rate_paths`, the group-sequential constants from
    :func:`~abkit.sequential.calibrate_boundaries`, the rule from
    :func:`~abkit.sequential.rejections` and the summary from
    :func:`~abkit.sequential.summarise_rule`. The mSPRT mixing scale is the minimum
    detectable effect of the fixed-horizon design, as in the pipeline's simulation.

    Parameters
    ----------
    baseline_rate : float
        Rate of the control arm, strictly between 0 and 1.
    relative_lift : float
        True relative lift of the treatment arm (0 simulates an A/A test).
    users_per_day : int
        Customers entering the experiment each day, split evenly between the two arms.
    n_days : int
        Days the experiment runs; the rule looks once a day.
    method : str
        One of :data:`PEEKING_METHODS`.
    rng : numpy.random.Generator
        Random generator.
    n_runs : int
        Simulated experiments.
    n_paths : int
        Null paths used to calibrate the Pocock and O'Brien-Fleming constants.
    alpha : float
        Nominal two-sided level.

    Returns
    -------
    PeekingResult
        Rejection rate with its Monte Carlo standard error and the stopping-day distribution.
    """
    if method not in PEEKING_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(PEEKING_METHODS)}.")
    if not 0 < baseline_rate < 1:
        raise ValueError("The baseline rate must lie strictly between 0 and 1.")
    rate_treatment = baseline_rate * (1 + relative_lift)
    if not 0 < rate_treatment < 1:
        raise ValueError("The lift takes the treatment rate outside (0, 1).")
    if not 1 <= n_days <= LIVE_MAX_DAYS:
        raise ValueError(f"The number of days must lie between 1 and {LIVE_MAX_DAYS}.")
    per_day = users_per_day // 2
    if per_day < 1:
        raise ValueError("Need at least two users per day, one per arm.")
    n_per_arm = per_day * n_days
    # Separate child streams: calibrating the boundaries never shifts the simulated experiments.
    rng_paths, rng_boundaries = rng.spawn(2)
    paths = sequential.simulate_rate_paths(
        baseline_rate, rate_treatment, n_per_arm, n_days, n_runs, rng_paths
    )
    constants = sequential.calibrate_boundaries(n_days, n_paths, rng_boundaries, alpha=alpha)
    tau = mde(baseline_rate, baseline_rate * (1 - baseline_rate), n_per_arm, alpha=alpha).absolute
    reject = sequential.rejections(
        paths,
        constants,
        tau,
        alpha=alpha,
        tau_multipliers={"msprt": 1.0},
        methods=[method],
        bayes_per_run=True,
    )[method]
    summary = sequential.summarise_rule(reject)
    stop = sequential.first_rejection(reject)
    share = np.bincount(stop[stop >= 0], minlength=n_days) / n_runs
    return PeekingResult(
        method=method,
        n_runs=n_runs,
        n_days=n_days,
        n_per_arm_per_day=per_day,
        rate_control=baseline_rate,
        rate_treatment=rate_treatment,
        reject_rate=summary["reject_rate"],
        reject_mcse=summary["reject_mcse"],
        mean_stop_day=summary["mean_stop_day"],
        mean_stop_day_se=summary["mean_stop_day_se"],
        stop_day_share=share,
        never_share=float(np.mean(stop < 0)),
        alpha=alpha,
    )


def live_runs(method: str, n_days: int) -> int:
    """Return the number of experiments the live simulator runs for a method and length.

    Parameters
    ----------
    method : str
        Decision rule.
    n_days : int
        Days (looks).

    Returns
    -------
    int
        :data:`LIVE_RUNS`, or fewer for the Bayesian rule so that runs times days stays
        within :data:`LIVE_BAYES_EVALUATIONS`.
    """
    if method == "bayes":
        return min(LIVE_RUNS, LIVE_BAYES_EVALUATIONS // n_days)
    return LIVE_RUNS


@st.cache_data(show_spinner=False, max_entries=64)
def cached_peeking(
    baseline_rate: float, relative_lift: float, users_per_day: int, n_days: int, method: str
) -> PeekingResult:
    """Run :func:`simulate_peeking` from the configured simulation seed (cached per input).

    The generator is rebuilt from ``config.SEEDS["simulation"]`` on every call, so the same
    inputs always give the same answer.
    """
    rng = np.random.default_rng(config.SEEDS["simulation"])
    return simulate_peeking(
        baseline_rate,
        relative_lift,
        users_per_day,
        n_days,
        method,
        rng,
        n_runs=live_runs(method, n_days),
    )


def stop_day_table(result: PeekingResult) -> pd.DataFrame:
    """Stopping-day distribution of a simulation as a table for plotting.

    Parameters
    ----------
    result : PeekingResult
        Output of :func:`simulate_peeking`.

    Returns
    -------
    pandas.DataFrame
        Columns ``day`` (label), ``order``, ``share`` and ``outcome``; one row per day on
        which a run can stop with a rejection, plus a final row for runs that never reject.
    """
    days = np.arange(1, result.n_days + 1)
    table = pd.DataFrame(
        {
            "day": [f"{d}" for d in days],
            "order": days,
            "share": result.stop_day_share,
            "outcome": "stopped and declared a winner",
        }
    )
    never = pd.DataFrame(
        {
            "day": ["never"],
            "order": [result.n_days + 1],
            "share": [result.never_share],
            "outcome": ["ran to the end, no winner declared"],
        }
    )
    return pd.concat([table, never], ignore_index=True)


def monte_carlo_note(result: PeekingResult) -> str:
    """One line stating the size and Monte Carlo error of a live simulation."""
    half = 1.96 * result.reject_mcse
    return (
        f"{result.n_runs:,} simulated experiments, {result.n_per_arm_per_day:,} customers "
        f"per arm per day. Monte Carlo standard error {100 * result.reject_mcse:.2f} pp "
        f"(about ± {100 * half:.1f} pp)."
    )


def is_finite(value: Any) -> bool:
    """Whether a value read from results.json is a usable number (not null or NaN)."""
    return isinstance(value, int | float) and math.isfinite(value)
