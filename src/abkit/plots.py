"""Matplotlib figures written to reports/figures, in one shared style.

Figures use the object-oriented API (``matplotlib.figure.Figure``) so no GUI backend or global
pyplot state is involved. Every figure has a CSV table behind it in reports/tables, and every
title states the finding the figure shows, computed from that table at plot time (never typed
by hand), so the titles can only say what the data says.

Colour: the reference palette validated for colour-vision deficiency. Arms always use
:data:`abkit.config.ARM_COLORS` (categorical slots 1-3: blue, orange, aqua). Any figure with
more than three series adds neutral inks and uses line style and marker shape as a second
encoding, because no larger subset of the palette keeps every pair apart under every form of
colour blindness. Text is always in the ink colours, never in a series colour.
"""

import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats

from abkit import config

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE, ORANGE, AQUA = (config.ARM_COLORS[a] for a in ("mens", "womens", "control"))
YELLOW, VIOLET = "#eda100", "#6250d6"  # palette slots 4 and 7, a validated pair
SERIES = BLUE  # single-series colour
FIGSIZE = (8.0, 4.5)
DPI = 150
TITLE_SIZE = 13
#: Approximate title characters per inch of figure width at the title size.
_CHARS_PER_INCH = 9.0

STYLE: dict[str, object] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.titlecolor": TEXT,
    "axes.titlepad": 8,
    "axes.labelsize": 11,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.color": TEXT,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
}


@contextmanager
def styled() -> Iterator[None]:
    """Context manager applying the project style to figures created inside it."""
    with mpl.rc_context(STYLE):
        yield


def new_figure(figsize: tuple[float, float] = FIGSIZE) -> tuple[Figure, Axes]:
    """Create a styled figure with one axes (call inside :func:`styled`)."""
    fig = Figure(figsize=figsize, layout="constrained")
    return fig, fig.add_subplot()


def finish(fig: Figure, title: str, note: str | None = None) -> None:
    """Set the figure title (the finding) and an optional caption under the figure.

    Parameters
    ----------
    fig : Figure
        Figure with constrained layout.
    title : str
        The finding, wrapped to the figure width.
    note : str, optional
        Secondary-ink caption at the bottom of the figure (data source, caveat), placed by
        the layout engine below the plot and any legend, so a wrapped title never meets it.
    """
    width = int(fig.get_figwidth() * _CHARS_PER_INCH)
    height = fig.get_figheight()
    title_text = textwrap.fill(title, width)
    # Title and note are placed by hand from the top edge (in inches) and the layout engine
    # is told to keep the axes, panel titles and any outside legend below them.
    used = 0.1
    fig.suptitle(
        title_text,
        x=0.01,
        y=1 - used / height,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=TITLE_SIZE,
        color=TEXT,
    )
    used += (title_text.count("\n") + 1) * TITLE_SIZE * 1.3 / 72
    if note:
        note_text = textwrap.fill(note, int(width * 1.3))
        used += 0.04
        fig.text(
            0.01,
            1 - used / height,
            note_text,
            ha="left",
            va="top",
            fontsize=10,
            color=TEXT_SECONDARY,
        )
        used += (note_text.count("\n") + 1) * 10 * 1.3 / 72
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - (used + 0.12) / height))


def save(fig: Figure, path: Path) -> Path:
    """Save a figure as PNG at README-friendly resolution.

    Parameters
    ----------
    fig : Figure
        Figure to save.
    path : Path
        Output path; parent directories are created.

    Returns
    -------
    Path
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    return path


def _join(items: list[str]) -> str:
    """Join items in prose: ``a``, ``a and b`` or ``a, b and c``."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _dollars(v: float) -> str:
    return f"${v:,.0f}"


# ---------------------------------------------------------------------------------------------
# EDA and validity
# ---------------------------------------------------------------------------------------------


def _histogram_summary(hist: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return (low, high, median, mean) of a binned distribution, using bin midpoints."""
    left, right, n = (hist[c].to_numpy(float) for c in ("bin_left", "bin_right", "n"))
    mid = np.sqrt(left * right) if (left > 0).all() else (left + right) / 2
    total = n.sum()
    cumulative = np.cumsum(n)
    median = float(mid[np.searchsorted(cumulative, total / 2)])
    mean = float((mid * n).sum() / total)
    return float(left.min()), float(right.max()), median, mean


def plot_log_histogram(
    hist: pd.DataFrame,
    subject: str,
    xlabel: str,
    path: Path,
    note: str | None = None,
    ticks: tuple[float, ...] = (30, 50, 100, 200, 500, 1000, 2000, 5000),
) -> Path:
    """Histogram with log-spaced bins on a logarithmic x-axis.

    The title states the shape found in the bins: the range, the median bin and whether the
    distribution is right-skewed (bin-weighted mean well above the median).

    Parameters
    ----------
    hist : pandas.DataFrame
        Columns ``bin_left``, ``bin_right``, ``n``.
    subject : str
        What is binned, for example ``"Spend among converters"``.
    xlabel : str
        X-axis label.
    path : Path
        Output PNG path.
    note : str, optional
        Caption under the title.
    ticks : tuple of float
        Candidate dollar tick positions; those inside the data range are shown.

    Returns
    -------
    Path
        The path written.
    """
    lo, hi, median, mean = _histogram_summary(hist)
    shape = "is right-skewed" if mean > 1.1 * median else "is roughly symmetric on a log scale"
    title = (
        f"{subject} {shape}: half of the values are below {_dollars(median)}, "
        f"the range runs {_dollars(lo)} to {_dollars(hi)}"
    )
    with styled():
        fig, ax = new_figure()
        ax.bar(
            hist["bin_left"],
            hist["n"],
            width=hist["bin_right"] - hist["bin_left"],
            align="edge",
            color=SERIES,
            edgecolor=SURFACE,
            linewidth=1.0,
        )
        ax.set_xscale("log")
        ax.set_xticks([t for t in ticks if lo <= t <= hi * 1.05])
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="x", visible=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Customers")
        finish(fig, title, note)
        return save(fig, path)


def plot_recency(recency: pd.DataFrame, path: Path) -> Path:
    """Bar chart of customers by months since last purchase.

    Parameters
    ----------
    recency : pandas.DataFrame
        Columns ``recency`` and ``n``.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        The path written.
    """
    months = recency["recency"].to_numpy()
    n = recency["n"].to_numpy(float)
    half = int(np.ceil(months.max() / 2))
    share = n[months <= half].sum() / n.sum()
    title = (
        f"{share:.0%} of customers last bought within {half} months; "
        f"recency runs {months.min()} to {months.max()} months"
    )
    with styled():
        fig, ax = new_figure()
        ax.bar(months, n, width=0.7, color=SERIES)
        ax.set_xticks(months)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Recency (months since last purchase)")
        ax.set_ylabel("Customers")
        finish(fig, title, "All arms pooled.")
        return save(fig, path)


def plot_arm_counts(srm: pd.DataFrame, path: Path) -> Path:
    """Horizontal bars of customers per arm with the expected count marked.

    Parameters
    ----------
    srm : pandas.DataFrame
        Output of :func:`abkit.eda.srm_table`: columns ``arm``, ``observed``, ``expected``,
        ``p_value``, ``alpha`` and ``passed``.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        The path written.
    """
    p, alpha, passed = (srm[c].iloc[0] for c in ("p_value", "alpha", "passed"))
    title = (
        f"Sample ratio check {'passed' if passed else 'FAILED'}: arm sizes "
        f"{'match' if passed else 'do not match'} an equal split (chi-square p = {p:.2f}; "
        f"the readout is blocked below {alpha:g})"
    )
    with styled():
        fig, ax = new_figure((8.0, 3.4))
        y = np.arange(len(srm))
        colors = [config.ARM_COLORS[a] for a in srm["arm"]]
        ax.barh(y, srm["observed"], height=0.55, color=colors)
        expected = float(srm["expected"].iloc[0])
        ax.axvline(expected, color=TEXT_SECONDARY, linestyle="--", linewidth=1.2)
        for yi, n in zip(y, srm["observed"], strict=True):
            ax.text(n, yi, f"  {n:,}", va="center", fontsize=10, color=TEXT)
        ax.text(
            expected,
            len(srm) - 0.45,
            f"expected {expected:,.0f}",
            ha="center",
            fontsize=10,
            color=TEXT_SECONDARY,
        )
        ax.set_yticks(y, list(srm["arm"]))
        ax.set_xlim(0, srm["observed"].max() * 1.15)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Customers")
        finish(fig, title)
        return save(fig, path)


def plot_balance(
    balance: pd.DataFrame, path: Path, threshold: float = config.SMD_THRESHOLD
) -> Path:
    """Dot plot of standardised mean differences, each email vs control.

    Parameters
    ----------
    balance : pandas.DataFrame
        Balance table from :func:`abkit.balance.balance_table`.
    path : Path
        Output PNG path.
    threshold : float
        |SMD| flag threshold, drawn as dashed lines.

    Returns
    -------
    Path
        The path written.
    """
    worst = float(balance["max_abs_smd"].max())
    n_flagged = int(balance["flagged"].sum())
    if n_flagged == 0:
        title = (
            f"Pre-treatment covariates are balanced across arms: the largest |SMD| is "
            f"{worst:.3f}, below the {threshold:g} flag"
        )
    else:
        title = (
            f"{n_flagged} covariate level{'s' if n_flagged > 1 else ''} exceed the "
            f"{threshold:g} |SMD| flag (largest {worst:.3f})"
        )
    labels = [
        f"{v}: {lvl}" if lvl else v
        for v, lvl in zip(balance["variable"], balance["level"], strict=True)
    ]
    y = np.arange(len(balance))[::-1]
    with styled():
        fig, ax = new_figure((8.0, 0.32 * len(balance) + 1.8))
        for arm, offset, marker in (("mens", 0.12, "o"), ("womens", -0.12, "s")):
            ax.scatter(
                balance[f"smd_{arm}_vs_control"],
                y + offset,
                s=36,
                marker=marker,
                color=config.ARM_COLORS[arm],
                edgecolor=SURFACE,
                linewidth=1.0,
                label=f"{arm} vs control",
                zorder=3,
            )
        for x in (-threshold, threshold):
            ax.axvline(x, color=TEXT_SECONDARY, linestyle="--", linewidth=1.0)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
        limit = max(1.5 * threshold, 1.2 * worst)
        ax.set_xlim(-limit, limit)
        ax.set_yticks(y, labels)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"Standardised mean difference (dashed: ±{threshold:g})")
        ax.legend(loc="lower right")
        finish(fig, title)
        return save(fig, path)


# ---------------------------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------------------------

FOREST_METRIC_LABELS: dict[str, str] = {
    "spend": "Spend per customer",
    "spend_winsorised": "Spend, winsorised (sensitivity)",
    "conversion": "Conversion rate",
    "visit": "Visit rate",
}
SHORT_METRIC_LABELS: dict[str, str] = {
    "spend": "spend",
    "spend_winsorised": "winsorised spend",
    "conversion": "conversion",
    "visit": "visits",
}


def _lift_finding(readout: pd.DataFrame) -> str:
    """Which metrics each email lifts significantly against control, from the table."""
    metrics = list(dict.fromkeys(readout["metric"]))
    lifted: dict[str, list[str]] = {}
    for treatment in config.TREATMENTS:
        rows = readout[(readout["treatment"] == treatment) & (readout["control"] == config.CONTROL)]
        rows = rows[rows["significant"] & (rows["diff"] > 0)]
        lifted[treatment] = [m for m in metrics if m in set(rows["metric"])]

    def what(names: list[str]) -> str:
        if len(names) == len(metrics):
            return "every metric"
        return _join([SHORT_METRIC_LABELS.get(m, m) for m in names]) if names else "no metric"

    if len(set(map(tuple, lifted.values()))) == 1:
        return f"both emails lift {what(next(iter(lifted.values())))}"
    return "; ".join(f"the {t} email lifts {what(names)}" for t, names in lifted.items())


def plot_forest(readout: pd.DataFrame, path: Path) -> Path:
    """Forest plot of relative lifts with their 95% intervals, grouped by metric.

    Parameters
    ----------
    readout : pandas.DataFrame
        Frequentist readout with columns ``metric``, ``treatment``, ``control``, ``diff``,
        ``significant``, ``rel_lift``, ``rel_ci_low`` and ``rel_ci_high``.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        The path written.
    """
    title = f"Against control, {_lift_finding(readout)} (adjusted within families)"
    metrics = list(dict.fromkeys(readout["metric"]))
    comparisons = list(dict.fromkeys(zip(readout["treatment"], readout["control"], strict=True)))
    colours = {
        (t, c): (config.ARM_COLORS[t] if c == config.CONTROL else TEXT_SECONDARY)
        for t, c in comparisons
    }
    markers = {(t, c): ("o" if c == config.CONTROL else "D") for t, c in comparisons}
    gap = 0.8  # vertical distance between comparisons within a metric group
    step = gap * len(comparisons) + 1.0
    with styled():
        fig, ax = new_figure((8.0, 0.42 * step * len(metrics) + 1.8))
        ticks, labels = [], []
        for g, metric in enumerate(metrics):
            base = -g * step
            ticks.append(base - gap * (len(comparisons) - 1) / 2)
            labels.append(FOREST_METRIC_LABELS.get(metric, metric))
            part = readout[readout["metric"] == metric]
            for k, (t, c) in enumerate(comparisons):
                row = part[(part["treatment"] == t) & (part["control"] == c)]
                if row.empty:
                    continue
                r = row.iloc[0]
                y = base - k * gap
                est = 100 * r["rel_lift"]
                lo, hi = 100 * r["rel_ci_low"], 100 * r["rel_ci_high"]
                ax.plot([lo, hi], [y, y], color=colours[(t, c)], linewidth=2.0, zorder=2)
                ax.scatter(
                    [est],
                    [y],
                    s=40,
                    marker=markers[(t, c)],
                    color=colours[(t, c)],
                    edgecolor=SURFACE,
                    linewidth=1.0,
                    zorder=3,
                    label=f"{t} vs {c}" if g == 0 else None,
                )
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.9)
        ax.set_yticks(ticks, labels)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
        ax.set_xlabel("Relative lift (first arm vs second), 95% interval")
        ax.legend(loc="lower right")
        finish(fig, title)
        return save(fig, path)


# Estimator colours for the adjusted-interval figure. Raw is the neutral reference; CUPED and
# Lin take palette slots 7 (violet) and 4 (yellow), a validated pair that never reuses an
# arm's colour. Marker shapes are the second encoding.
ESTIMATOR_STYLES: dict[str, tuple[str, str, str]] = {
    "raw": ("Raw (Welch)", TEXT_SECONDARY, "o"),
    "cuped": ("CUPED (history)", VIOLET, "s"),
    "lin": ("Lin regression (all pre-treatment)", YELLOW, "D"),
}


def plot_adjusted_intervals(table: pd.DataFrame, path: Path) -> Path:
    """Small multiples of raw, CUPED and Lin-adjusted 95% intervals, one panel per metric.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`abkit.readout.build_cuped_readout`: columns ``metric``,
        ``treatment``, ``control``, ``{raw,cuped,lin}_{diff,ci_low,ci_high}`` and
        ``{cuped,lin}_variance_reduction``.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        The path written.
    """
    best = max(
        float(table["cuped_variance_reduction"].max()), float(table["lin_variance_reduction"].max())
    )
    if best < 0.05:
        title = (
            f"Covariate adjustment barely helps here: CUPED and the Lin regression remove at "
            f"most {best:.1%} of the variance, so their intervals sit on the raw ones"
        )
    else:
        title = (
            f"Covariate adjustment narrows the intervals: up to {best:.0%} of the variance is "
            "removed by CUPED or the Lin regression"
        )
    metrics = list(dict.fromkeys(table["metric"]))
    comparisons = list(dict.fromkeys(zip(table["treatment"], table["control"], strict=True)))
    n_cols = 2
    n_rows = -(-len(metrics) // n_cols)
    gap, step = 0.25, 1.2
    with styled():
        fig = Figure(figsize=(10.0, 2.6 * n_rows + 1.4), layout="constrained")
        axes = fig.subplots(n_rows, n_cols, squeeze=False).ravel()
        for ax, metric in zip(axes, metrics, strict=False):
            part = table[table["metric"] == metric]
            is_rate = metric in ("visit", "conversion")
            scale = 100.0 if is_rate else 1.0
            ticks, labels = [], []
            for k, (t, c) in enumerate(comparisons):
                r = part[(part["treatment"] == t) & (part["control"] == c)].iloc[0]
                base = -k * step
                ticks.append(base)
                labels.append(f"{t} vs {c}")
                for j, (est, (name, colour, marker)) in enumerate(ESTIMATOR_STYLES.items()):
                    y = base + gap * (1 - j)
                    lo, hi = scale * r[f"{est}_ci_low"], scale * r[f"{est}_ci_high"]
                    ax.plot([lo, hi], [y, y], color=colour, linewidth=2.0, zorder=2)
                    ax.scatter(
                        [scale * r[f"{est}_diff"]],
                        [y],
                        s=40,
                        marker=marker,
                        color=colour,
                        edgecolor=SURFACE,
                        linewidth=1.0,
                        zorder=3,
                        label=name if (k == 0 and ax is axes[0]) else None,
                    )
            ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.9)
            ax.set_yticks(ticks, labels)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="y", visible=False)
            if is_rate:
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:+.1f} pp"))
            else:
                ax.xaxis.set_major_formatter(
                    mpl.ticker.FuncFormatter(
                        lambda v, _: f"{'+' if v > 0 else '-' if v < 0 else ''}${abs(v):.2f}"
                    )
                )
            ax.set_title(FOREST_METRIC_LABELS.get(metric, metric))
        for ax in axes[len(metrics) :]:
            ax.set_visible(False)
        fig.legend(loc="outside lower center", ncol=len(ESTIMATOR_STYLES))
        finish(fig, title, "Difference vs the comparison arm with 95% intervals.")
        return save(fig, path)


# Power-curve lines: colour encodes the level (slot 1 blue for alpha as pre-registered, grey
# for the Holm level), line style encodes CUPED (solid without, dashed with).
POWER_LEVEL_COLORS: dict[str, str] = {"unadjusted": SERIES, "holm": TEXT_SECONDARY}
POWER_METRIC_LABELS: dict[str, str] = {
    "spend": "Spend per customer",
    "conversion": "Conversion rate",
    "visit": "Visit rate",
}


def plot_power_curve(curve: pd.DataFrame, mdes: pd.DataFrame, path: Path) -> Path:
    """Power against relative lift at the actual sample size, one panel per metric.

    Parameters
    ----------
    curve : pandas.DataFrame
        Output of :func:`abkit.power.power_curve_table`.
    mdes : pandas.DataFrame
        Output of :func:`abkit.power.mde_table`; the unadjusted, non-CUPED MDE of each metric
        is marked on its panel and quoted in the title.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        The path written.
    """
    metrics = list(dict.fromkeys(curve["metric"]))
    levels = list(dict.fromkeys(zip(curve["alpha_label"], curve["alpha"], strict=True)))
    target = float(mdes["power"].iloc[0])
    n = int(mdes["n_per_arm"].iloc[0])
    reduction = float(mdes["variance_reduction"].max())
    reference = mdes[(mdes["alpha_label"] == "unadjusted") & ~mdes["cuped"]].set_index("metric")
    mde_text = _join(
        [
            f"{100 * float(reference.loc[m, 'mde_relative']):.0f}% on "
            f"{SHORT_METRIC_LABELS.get(m, m)}"
            for m in metrics
        ]
    )
    title = (
        f"With {n:,} customers per arm the smallest relative lift detectable at "
        f"{target:.0%} power is {mde_text}"
    )
    with styled():
        fig = Figure(figsize=(10.0, 4.0), layout="constrained")
        axes = fig.subplots(1, len(metrics), squeeze=False).ravel()
        for ax, metric in zip(axes, metrics, strict=True):
            part = curve[curve["metric"] == metric]
            for label, alpha in levels:
                for with_cuped in (False, True):
                    line = part[(part["alpha_label"] == label) & (part["cuped"] == with_cuped)]
                    ax.plot(
                        100 * line["relative_lift"],
                        line["power"],
                        color=POWER_LEVEL_COLORS[label],
                        linestyle="--" if with_cuped else "-",
                        label=(
                            f"alpha = {alpha:g}{' (Holm)' if label == 'holm' else ''}"
                            f"{', with CUPED' if with_cuped else ''}"
                        )
                        if ax is axes[0]
                        else None,
                    )
            ax.axhline(target, color=TEXT_SECONDARY, linestyle=":", linewidth=1.0)
            x_mde = 100 * float(reference.loc[metric, "mde_relative"])
            ax.axvline(x_mde, color=SERIES, linestyle=":", linewidth=1.0)
            ax.text(x_mde, 0.03, f" MDE {x_mde:.0f}%", fontsize=9, color=TEXT_SECONDARY)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(0, 100 * float(part["relative_lift"].max()))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.set_title(POWER_METRIC_LABELS.get(metric, metric))
            ax.set_xlabel("True relative lift vs control")
        axes[0].set_ylabel(f"Power (dotted: {target:.0%} target)")
        fig.legend(loc="outside lower center", ncol=2)
        finish(
            fig,
            title,
            f"CUPED on history removes at most {reduction:.1%} of the variance, so the dashed "
            "curves sit on the solid ones.",
        )
        return save(fig, path)


def _prior_label(prior: str, strength: float) -> str:
    return "Flat\nBeta(1, 1)" if prior == "flat" else f"Shared\nn₀ = {strength:,.0f}"


def plot_p_better_by_prior(
    table: pd.DataFrame, path: Path, threshold: float = config.BAYES_P_BETTER
) -> Path:
    """P(first arm better) against prior strength, one panel per rate metric.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`abkit.bayes.build_bayes_rates`: columns ``metric``, ``prior``,
        ``prior_strength``, ``treatment``, ``control`` and ``p_better``.
    path : Path
        Output PNG path.
    threshold : float
        The decision rule's P(better) line, drawn dotted.

    Returns
    -------
    Path
        The path written.
    """
    metrics = list(dict.fromkeys(table["metric"]))
    versus_control = table[table["control"] == config.CONTROL]
    weakest = versus_control.groupby(["treatment", "metric"])["p_better"].min()
    below = [
        f"{t} on {SHORT_METRIC_LABELS.get(m, m)}" for (t, m), p in weakest.items() if p < threshold
    ]
    if not below:
        title = (
            f"Each email beats control on {_join([SHORT_METRIC_LABELS.get(m, m) for m in metrics])}"
            f" with P(better) at or above {threshold:.0%} under every prior, so the prior does "
            "not drive the answer"
        )
    else:
        title = (
            f"P(better than control) falls below {threshold:.0%} for {_join(below)} under at "
            "least one prior; the other comparisons hold under every prior"
        )
    strength = table.groupby("prior", sort=False)["prior_strength"].first()
    priors = [(str(p), float(strength[p])) for p in dict.fromkeys(table["prior"])]
    comparisons = list(dict.fromkeys(zip(table["treatment"], table["control"], strict=True)))
    colours = {
        (t, c): (config.ARM_COLORS[t] if c == config.CONTROL else TEXT_SECONDARY)
        for t, c in comparisons
    }
    # Series often coincide (several probabilities are 1 to many decimals), so each comparison
    # is dodged slightly along the categorical axis and gets its own marker shape.
    x = np.arange(len(priors))
    markers = ("o", "s", "D")
    dodge = 0.1 * (np.arange(len(comparisons)) - (len(comparisons) - 1) / 2)
    with styled():
        fig = Figure(figsize=(10.0, 4.8), layout="constrained")
        axes = fig.subplots(1, len(metrics), squeeze=False, sharey=True).ravel()
        for ax, metric in zip(axes, metrics, strict=True):
            part = table[table["metric"] == metric].set_index(["prior", "treatment", "control"])
            for k, (t, c) in enumerate(comparisons):
                y = [part.loc[(p, t, c), "p_better"] for p, _ in priors]
                ax.plot(
                    x + dodge[k],
                    y,
                    marker=markers[k % len(markers)],
                    markersize=6,
                    markeredgecolor=SURFACE,
                    color=colours[(t, c)],
                    label=f"{t} vs {c}" if ax is axes[0] else None,
                    zorder=3,
                )
            ax.axhline(threshold, color=TEXT_SECONDARY, linestyle=":", linewidth=1.0)
            ax.axhline(0.5, color=TEXT_SECONDARY, linewidth=0.8)
            ax.text(
                x[-1] + 0.3,
                threshold - 0.02,
                f"{threshold:.0%} ship threshold",
                ha="right",
                va="top",
                fontsize=9,
                color=TEXT_SECONDARY,
            )
            ax.set_ylim(0, 1.04)
            ax.set_xlim(-0.4, len(priors) - 0.6)
            ax.set_xticks(x, [_prior_label(p, s) for p, s in priors])
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.grid(axis="x", visible=False)
            ax.set_title(FOREST_METRIC_LABELS.get(metric, metric))
            ax.set_xlabel("Prior (shared priors centred on the pooled rate)")
        axes[0].set_ylabel("P(first arm better)")
        fig.legend(loc="outside lower center", ncol=len(comparisons))
        finish(fig, title)
        return save(fig, path)


# Models in the spend-posterior figure: the primary analysis is a filled density, the
# parametric cross-check a dashed outline. Both take the comparison's colour (arm colour vs
# control, neutral for mens vs womens); line style is the second encoding.
SPEND_MODEL_LABELS: dict[str, str] = {
    "bayesian_bootstrap": "Bayesian bootstrap (primary)",
    "two_part_lognormal": "Two-part log-normal (cross-check)",
}


def plot_spend_posterior(
    hist: pd.DataFrame, table: pd.DataFrame, path: Path, threshold: float = config.BAYES_P_BETTER
) -> Path:
    """Posterior densities of the difference in spend per customer, one panel per comparison.

    Parameters
    ----------
    hist : pandas.DataFrame
        Binned densities with columns ``model``, ``comparison``, ``bin_left``, ``bin_right``
        and ``density`` (raw spend).
    table : pandas.DataFrame
        Output of :func:`abkit.bayes.build_bayes_spend` (first element); the raw-spend
        Bayesian bootstrap rows supply the credible interval and P(better) annotations.
    path : Path
        Output PNG path.
    threshold : float
        The decision rule's P(better), used to word the title.

    Returns
    -------
    Path
        The path written.
    """
    comparisons = list(dict.fromkeys(hist["comparison"]))
    models = list(dict.fromkeys(hist["model"]))
    primary = table[(table["model"] == "bayesian_bootstrap") & (table["metric"] == "spend")]
    primary = primary.set_index("comparison")
    p = {c: float(primary.loc[c, "p_better"]) for c in comparisons}
    strong = [t for t in config.TREATMENTS if p.get(f"{t}_vs_{config.CONTROL}", 0) >= threshold]
    probs = _join([f"{p[f'{t}_vs_{config.CONTROL}']:.3f} for {t}" for t in config.TREATMENTS])
    if len(strong) == len(config.TREATMENTS):
        lead = "Both emails raise spend per customer"
    elif strong:
        lead = f"Only the {strong[0]} email clearly raises spend per customer"
    else:
        lead = "Neither email clearly raises spend per customer"
    head = f"{config.TREATMENTS[0]}_vs_{config.TREATMENTS[1]}"
    tail = f"; P(mens > womens) = {p[head]:.3f}" if head in p else ""
    title = f"{lead}: P(better than control) is {probs}{tail}"
    with styled():
        fig = Figure(figsize=(8.0, 1.9 * len(comparisons) + 1.8), layout="constrained")
        axes = fig.subplots(len(comparisons), 1, squeeze=False, sharex=True).ravel()
        for ax, comparison in zip(axes, comparisons, strict=True):
            t, c = comparison.split("_vs_")
            colour = config.ARM_COLORS[t] if c == config.CONTROL else TEXT_SECONDARY
            for model in models:
                part = hist[(hist["model"] == model) & (hist["comparison"] == comparison)]
                x = np.append(part["bin_left"].to_numpy(), part["bin_right"].iloc[-1])
                y = np.append(part["density"].to_numpy(), part["density"].iloc[-1])
                label = SPEND_MODEL_LABELS.get(model, model) if ax is axes[0] else None
                if model == "bayesian_bootstrap":
                    ax.fill_between(x, y, step="post", color=colour, alpha=0.25, linewidth=0)
                    ax.step(x, y, where="post", color=colour, linewidth=1.6, label=label)
                else:
                    ax.step(
                        x, y, where="post", color=colour, linewidth=1.4, linestyle="--", label=label
                    )
            r = primary.loc[comparison]
            ax.plot(
                [r["ci_low"], r["ci_high"]],
                [0, 0],
                color=colour,
                linewidth=3.0,
                solid_capstyle="butt",
                zorder=3,
            )
            ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.9)
            ax.text(
                0.99,
                0.92,
                f"{t} vs {c}\nP(first arm better) = {r['p_better']:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
            )
            ax.set_yticks([])
            ax.grid(axis="y", visible=False)
            ax.set_ylabel("Density")
        axes[-1].xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: f"{'-' if v < 0 else '+'}${abs(v):.2f}")
        )
        axes[-1].set_xlabel("Difference in mean spend per customer (first arm minus second)")
        fig.legend(loc="outside lower center", ncol=2)
        finish(fig, title, "Bar on the axis: 95% credible interval of the primary model.")
        return save(fig, path)


# ---------------------------------------------------------------------------------------------
# Sequential testing
# ---------------------------------------------------------------------------------------------

SEQUENTIAL_METHOD_LABELS: dict[str, str] = {
    "fixed": "Fixed horizon (day 14)",
    "naive": "Naive peeking (z-test daily)",
    "bayes": "Bayesian peeking (P > 0.95)",
    "pocock": "Pocock",
    "obf": "O'Brien-Fleming",
    "msprt": "mSPRT (always-valid p)",
}
# Five rules plus the fixed-horizon reference: three palette slots, two neutral inks, and a
# distinct line style and marker for each so no pair relies on colour alone.
SEQUENTIAL_METHOD_STYLES: dict[str, tuple[str, str, str]] = {
    "fixed": (TEXT_SECONDARY, ":", ""),
    "naive": (ORANGE, "-", "o"),
    "bayes": (TEXT, "-.", "x"),
    "pocock": (BLUE, "-", "s"),
    "obf": (AQUA, "-", "D"),
    "msprt": (MUTED, "--", "^"),
}
VALID_SEQUENTIAL: tuple[str, ...] = ("pocock", "obf", "msprt")
SEQUENTIAL_METRIC_LABELS: dict[str, str] = {
    "visit": "Visit rate",
    "conversion": "Conversion rate",
    "spend": "Spend per customer",
}
SEQUENTIAL_SCENARIO_LABELS: dict[str, str] = {
    "aa": "A/A: no true effect",
    "ab_observed": "A/B: observed visit lift",
    "ab_mde": "A/B: visit lift at the MDE",
}


def _method_line(ax: Axes, x: pd.Series, y: pd.Series, method: str, label: bool = True) -> None:
    colour, style, marker = SEQUENTIAL_METHOD_STYLES[method]
    ax.plot(
        x,
        y,
        color=colour,
        linestyle=style,
        marker=marker or None,
        markersize=4,
        label=SEQUENTIAL_METHOD_LABELS[method] if label else None,
    )


def plot_sequential_fpr_vs_looks(looks: pd.DataFrame, path: Path, alpha: float = 0.05) -> Path:
    """A/A false positive rate against the number of looks, one panel per metric.

    Parameters
    ----------
    looks : pandas.DataFrame
        Looks table from :func:`abkit.sequential.build_sequential_tables`.
    path : Path
        Output PNG path.
    alpha : float
        Nominal level, drawn dotted.

    Returns
    -------
    Path
        The path written.
    """
    metrics = list(dict.fromkeys(looks["metric"]))
    methods = [m for m in SEQUENTIAL_METHOD_LABELS if m in set(looks["method"])]
    k = int(looks["n_looks"].max())
    at_k = looks[looks["n_looks"] == k].groupby("method")["reject_rate"].max()
    valid = [m for m in VALID_SEQUENTIAL if m in at_k.index]
    naive = f"{at_k['naive']:.0%}" if "naive" in at_k.index else "n/a"
    bayes = f" and Bayesian peeking to {at_k['bayes']:.0%}" if "bayes" in at_k.index else ""
    valid_text = (
        f"; {_join([SEQUENTIAL_METHOD_LABELS[m].split(' (')[0] for m in valid])} stay at or "
        f"below {at_k[valid].max():.1%}"
        if valid
        else ""
    )
    title = (
        f"Peeking {k} times inflates the A/A false positive rate from the nominal {alpha:.0%} "
        f"to {naive} with the naive z-test{bayes}{valid_text}"
    )
    n_runs = int(looks["n_runs"].iloc[0])
    with styled():
        fig = Figure(figsize=(10.0, 4.2), layout="constrained")
        axes = fig.subplots(1, len(metrics), squeeze=False, sharey=True).ravel()
        for ax, metric in zip(axes, metrics, strict=True):
            part = looks[looks["metric"] == metric]
            for method in methods:
                line = part[part["method"] == method].sort_values("n_looks")
                _method_line(ax, line["n_looks"], line["reject_rate"], method, ax is axes[0])
            ax.axhline(alpha, color=TEXT_SECONDARY, linestyle=":", linewidth=1.0)
            ax.set_title(SEQUENTIAL_METRIC_LABELS.get(metric, metric))
            ax.set_xlabel("Number of equally spaced looks")
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
            ax.set_ylim(0, None)
        axes[0].set_ylabel(f"False positive rate (dotted: {alpha:.0%})")
        fig.legend(loc="outside lower center", ncol=3)
        finish(fig, title, f"Simulated A/A tests, {n_runs:,} runs per point; worst metric quoted.")
        return save(fig, path)


def plot_sequential_cumulative(
    by_day: pd.DataFrame, path: Path, metric: str = "visit", alpha: float = 0.05
) -> Path:
    """Cumulative rejection rate by day for one metric, one panel per scenario.

    Parameters
    ----------
    by_day : pandas.DataFrame
        By-day table from :func:`abkit.sequential.build_sequential_tables`.
    path : Path
        Output PNG path.
    metric : str
        Metric to show (the headline one).
    alpha : float
        Nominal level, drawn dotted on the A/A panel.

    Returns
    -------
    Path
        The path written.
    """
    part = by_day[(by_day["metric"] == metric) & (by_day["method"] != "single_look")]
    scenarios = [s for s in SEQUENTIAL_SCENARIO_LABELS if s in set(part["scenario"])]
    methods = [m for m in SEQUENTIAL_METHOD_LABELS if m in set(part["method"])]
    last_day = int(part["day"].max())
    final = part[part["day"] == last_day].set_index(["scenario", "method"])["reject_rate"].to_dict()
    name = SEQUENTIAL_METRIC_LABELS.get(metric, metric).lower()
    pieces = []
    if ("aa", "naive") in final:
        pieces.append(
            f"with no true effect naive daily peeking has rejected {final[('aa', 'naive')]:.0%} "
            f"of runs by day {last_day}"
            + (
                f" against {final[('aa', 'obf')]:.0%} for O'Brien-Fleming"
                if ("aa", "obf") in final
                else ""
            )
        )
    if ("ab_observed", "obf") in final:
        pieces.append(
            f"with the observed lift O'Brien-Fleming has stopped "
            f"{final[('ab_observed', 'obf')]:.0%} of runs"
            + (
                f" and the fixed-horizon test {final[('ab_observed', 'fixed')]:.0%}"
                if ("ab_observed", "fixed") in final
                else ""
            )
        )
    title = f"Daily looks on the {name}: " + "; ".join(pieces) if pieces else name
    with styled():
        fig = Figure(figsize=(10.0, 4.2), layout="constrained")
        axes = fig.subplots(1, len(scenarios), squeeze=False, sharey=True).ravel()
        for ax, scenario in zip(axes, scenarios, strict=True):
            sub = part[part["scenario"] == scenario]
            for method in methods:
                line = sub[sub["method"] == method].sort_values("day")
                _method_line(ax, line["day"], line["reject_rate"], method, ax is axes[0])
            if scenario == "aa":
                ax.axhline(alpha, color=TEXT_SECONDARY, linestyle=":", linewidth=1.0)
            ax.set_title(SEQUENTIAL_SCENARIO_LABELS[scenario])
            ax.set_xlabel("Day")
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
            ax.set_ylim(0, 1.0)
        axes[0].set_ylabel("Share of runs stopped by the day")
        fig.legend(loc="outside lower center", ncol=3)
        finish(fig, title, "Simulated experiments; each rule looks once a day.")
        return save(fig, path)


def plot_sequential_boundaries(boundaries: pd.DataFrame, path: Path) -> Path:
    """|z| rejection boundaries of each rule at every daily look.

    Parameters
    ----------
    boundaries : pandas.DataFrame
        Boundaries table from :func:`abkit.sequential.build_sequential_tables`: columns
        ``look``, ``naive``, ``pocock``, ``obf`` and ``msprt_visit``.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        The path written.
    """
    first = boundaries.sort_values("look").iloc[0]
    n_looks = int(boundaries["n_looks"].iloc[0])
    title = (
        f"Valid rules ask for more evidence at early looks: on day 1 O'Brien-Fleming needs "
        f"|z| ≥ {first['obf']:.1f} and Pocock {first['pocock']:.2f} against {first['naive']:.2f} "
        f"for naive peeking (two-sided 5%, {n_looks} looks)"
    )
    with styled():
        fig, ax = new_figure((8.0, 4.8))
        look = boundaries["look"]
        for column, method, label in (
            ("naive", "naive", "Naive peeking: 1.96 at every look"),
            ("pocock", "pocock", "Pocock: constant boundary"),
            ("obf", "obf", "O'Brien-Fleming: c · sqrt(K / k)"),
            ("msprt_visit", "msprt", "mSPRT on visit (tau = MDE)"),
        ):
            colour, style, marker = SEQUENTIAL_METHOD_STYLES[method]
            ax.plot(
                look,
                boundaries[column],
                color=colour,
                linestyle=style,
                marker=marker,
                markersize=4,
                label=label,
            )
        ax.set_xlabel("Look (day)")
        ax.set_ylabel("|z| needed to stop")
        ax.set_xticks(look)
        ax.set_ylim(0, None)
        ax.legend(loc="upper right")
        finish(fig, title)
        return save(fig, path)


# ---------------------------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------------------------

SEGMENT_VARIABLE_LABELS: dict[str, str] = {
    "newbie": "New customer",
    "channel": "Channel",
    "zip_code": "Zip code",
    "mens": "Past men's buyer",
    "womens": "Past women's buyer",
    "recency": "Recency (months)",
    "history_segment": "History band",
}
SEGMENT_OUTCOME_LABELS: dict[str, tuple[str, float]] = {
    "spend": ("Spend per customer, $", 1.0),
    "visit": ("Visit rate, pp", 100.0),
    "conversion": ("Conversion rate, pp", 100.0),
}


def plot_segment_forest(
    effects: pd.DataFrame, path: Path, interactions: pd.DataFrame | None = None
) -> Path:
    """Forest plot of the effect of each email vs control within every segment level.

    One panel per outcome, segment levels grouped by variable on the vertical axis, one
    series per email. Filled markers are significant after Benjamini-Hochberg within the
    outcome's family; hollow markers are not. Every result is exploratory.

    Parameters
    ----------
    effects : pandas.DataFrame
        Output of :func:`abkit.segments.segment_effects` after adjustment: columns
        ``outcome``, ``variable``, ``level``, ``treatment``, ``diff``, ``ci_low``,
        ``ci_high`` and ``significant_fdr``.
    path : Path
        Output PNG path.
    interactions : pandas.DataFrame, optional
        Interaction tests with a ``significant_fdr`` column; when given, the title also says
        how many of them survive the adjustment.

    Returns
    -------
    Path
        The path written.
    """
    n_sig, n = int(effects["significant_fdr"].sum()), len(effects)
    title = f"Exploratory: {n_sig} of {n} within-segment effects survive BH"
    if interactions is not None:
        k, m = int(interactions["significant_fdr"].sum()), len(interactions)
        title += (
            f", but none of the {m} interaction tests does, so no segment is shown to "
            "respond differently"
            if k == 0
            else f", and {k} of the {m} interaction tests do"
        )
    outcomes = [o for o in SEGMENT_OUTCOME_LABELS if o in set(effects["outcome"])]
    treatments = list(dict.fromkeys(effects["treatment"]))
    groups = list(dict.fromkeys(zip(effects["variable"], effects["level"], strict=True)))
    # Vertical positions: one row per level, a blank row between variables.
    positions: dict[tuple[str, str], float] = {}
    y, previous = 0.0, None
    group_mid: dict[str, list[float]] = {}
    for variable, level in groups:
        if previous is not None and variable != previous:
            y -= 1.0
        positions[(variable, level)] = y
        group_mid.setdefault(variable, []).append(y)
        y -= 1.0
        previous = variable
    dodge = 0.18 * (np.arange(len(treatments)) - (len(treatments) - 1) / 2)
    markers = ("o", "s", "D")
    with styled():
        fig = Figure(figsize=(11.0, 0.3 * len(positions) + 3.0), layout="constrained")
        axes = fig.subplots(1, len(outcomes), squeeze=False, sharey=True).ravel()
        for ax, outcome in zip(axes, outcomes, strict=True):
            label, scale = SEGMENT_OUTCOME_LABELS[outcome]
            part = effects[effects["outcome"] == outcome]
            for k, treatment in enumerate(treatments):
                colour = config.ARM_COLORS[treatment]
                sub = part[part["treatment"] == treatment]
                for _, r in sub.iterrows():
                    yy = positions[(r["variable"], r["level"])] + dodge[k]
                    ax.plot(
                        [scale * r["ci_low"], scale * r["ci_high"]],
                        [yy, yy],
                        color=colour,
                        linewidth=1.6,
                        zorder=2,
                    )
                    ax.scatter(
                        [scale * r["diff"]],
                        [yy],
                        s=30,
                        marker=markers[k % len(markers)],
                        facecolor=colour if r["significant_fdr"] else SURFACE,
                        edgecolor=colour,
                        linewidth=1.2,
                        zorder=3,
                    )
                if ax is axes[0]:
                    ax.scatter(
                        [],
                        [],
                        s=30,
                        marker=markers[k],
                        color=colour,
                        label=f"{treatment} vs control",
                    )
            ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.9)
            ax.set_xlabel(label)
            ax.set_title(label.split(",")[0])
        for variable, ys in group_mid.items():
            axes[0].text(
                -0.02,
                max(ys) + 0.55,
                SEGMENT_VARIABLE_LABELS.get(variable, variable),
                transform=axes[0].get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=TEXT_SECONDARY,
            )
        axes[0].set_yticks(list(positions.values()), [level for _, level in positions])
        for ax in axes:
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="y", visible=False)
        axes[0].scatter([], [], s=30, facecolor=TEXT, edgecolor=TEXT, label="BH-significant")
        axes[0].scatter([], [], s=30, facecolor=SURFACE, edgecolor=TEXT, label="not significant")
        fig.legend(loc="outside lower center", ncol=4)
        finish(
            fig,
            title,
            "Effect of each email vs control within pre-specified segments, 95% intervals.",
        )
        return save(fig, path)


# ---------------------------------------------------------------------------------------------
# Uplift models and policies
# ---------------------------------------------------------------------------------------------

# Three learners on palette slots 1-3; the baselines are neutral inks with their own line
# styles (dashed response model, dash-dot recency rule, dotted random reference).
RANKING_STYLES: dict[str, tuple[str, str]] = {
    "s_learner": (BLUE, "-"),
    "t_learner": (ORANGE, "-"),
    "x_learner": (AQUA, "-"),
    "response_model": (TEXT_SECONDARY, "--"),
    "recency_rule": (TEXT, "-."),
    "random": (MUTED, ":"),
}
RANKING_LABELS: dict[str, str] = {
    "s_learner": "S-learner",
    "t_learner": "T-learner",
    "x_learner": "X-learner",
    "response_model": "response model",
    "recency_rule": "recency rule",
    "random": "random",
}


def _ranking_colour(name: str) -> str:
    return RANKING_STYLES.get(name, (SERIES, "-"))[0]


def _draw_calibration(ax: Axes, deciles: pd.DataFrame, colour: str = SERIES) -> None:
    pred, obs = deciles["predicted"].to_numpy(), deciles["observed"].to_numpy()
    err = np.vstack([obs - deciles["boot_low"], deciles["boot_high"] - obs])
    ax.errorbar(pred, obs, yerr=err, fmt="o", color=colour, ecolor=colour, capsize=3, lw=1.2)
    for d, xp, yo in zip(deciles["decile"], pred, obs, strict=True):
        ax.annotate(str(d), (xp, yo), xytext=(5, 4), textcoords="offset points", fontsize=8)
    lo = min(float(np.nanmin(deciles["boot_low"])), float(pred.min()))
    hi = max(float(np.nanmax(deciles["boot_high"])), float(pred.max()))
    ax.plot([lo, hi], [lo, hi], color=TEXT_SECONDARY, lw=1.0, ls="--", label="perfect calibration")
    ax.set_xlabel("Mean predicted uplift in decile")
    ax.set_ylabel("Observed uplift (treated minus control rate)")


def _rank_correlation(deciles: pd.DataFrame) -> float:
    """Spearman correlation between predicted and observed uplift across deciles."""
    if len(deciles) < 3:
        return float("nan")
    return float(stats.spearmanr(deciles["predicted"], deciles["observed"]).statistic)


def plot_calibration_grid(deciles: dict[str, pd.DataFrame], path: Path, title: str = "") -> Path:
    """One calibration panel per learner, sharing both axes.

    The title names the learners whose decile uplift tracks their predictions (Spearman
    rank correlation of at least 0.5 across deciles) and those whose does not.

    Parameters
    ----------
    deciles : dict of str to pandas.DataFrame
        Learner name to its ``deciles`` table of
        :class:`abkit.uplift.evaluation.BootstrapEvaluation`.
    path : Path
        Output PNG path.
    title : str
        Name of the pair and outcome.

    Returns
    -------
    Path
        The path written.
    """
    names = list(deciles)
    rho = {n: _rank_correlation(deciles[n]) for n in names}
    tracks = [RANKING_LABELS.get(n, n) for n in names if rho[n] >= 0.5]
    prefix = f"{title}: " if title else ""
    if len(tracks) == len(names):
        finding = "observed uplift by decile tracks the prediction for every learner"
    elif tracks:
        finding = f"observed uplift by decile tracks the prediction for the {_join(tracks)} only"
    else:
        finding = "observed uplift by decile does not track the prediction for any learner"
    with styled():
        fig = Figure(figsize=(5.0 * len(names), 5.2), layout="constrained")
        axes = fig.subplots(1, len(names), squeeze=False, sharex=True, sharey=True).ravel()
        for ax, name in zip(axes, names, strict=True):
            _draw_calibration(ax, deciles[name], _ranking_colour(name))
            ax.set_title(f"{RANKING_LABELS.get(name, name)} (rank correlation {rho[name]:.2f})")
        axes[0].legend(loc="upper left")
        finish(
            fig, f"{prefix}{finding}", "Bars: bootstrap intervals of the observed decile uplift."
        )
    return save(fig, path)


def plot_qini_comparison(
    curves: dict[str, pd.DataFrame],
    path: Path,
    title: str = "",
    comparison: pd.DataFrame | None = None,
) -> Path:
    """Qini curves of several rankings on one axis, with bootstrap bands for the uplift models.

    Parameters
    ----------
    curves : dict of str to pandas.DataFrame
        Ranking name to its ``curves`` table of
        :class:`abkit.uplift.evaluation.BootstrapEvaluation`. Names ending in ``_learner``
        get a shaded band; baselines are drawn dashed. The ``random`` entry is drawn as the
        straight line from the origin to its final value.
    path : Path
        Output PNG path.
    title : str
        Name of the pair and outcome.
    comparison : pandas.DataFrame, optional
        Output of :func:`abkit.uplift.models.compare_rankings` for this pair and outcome
        (columns ``best``, ``ranking``, ``meaningful``); when given, the title states which
        ranking is best and which rankings it is meaningfully ahead of.

    Returns
    -------
    Path
        The path written.
    """
    prefix = f"{title}: " if title else ""
    if comparison is not None and not comparison.empty:
        best = str(comparison["best"].iloc[0])
        ahead = [
            RANKING_LABELS.get(r, r) for r in comparison.loc[comparison["meaningful"], "ranking"]
        ]
        tied = [
            RANKING_LABELS.get(r, r)
            for r in comparison.loc[~comparison["meaningful"], "ranking"]
            if r != best
        ]
        finding = f"the {RANKING_LABELS.get(best, best)} ranks best"
        finding += f", meaningfully ahead of {_join(ahead)}" if ahead else ""
        finding += f", but not separated from {_join(tied)}" if tied else ""
        if not ahead:
            finding += "; no ranking is meaningfully ahead of any other"
    else:
        finding = "Qini curves by ranking"
    with styled():
        fig, ax = new_figure((8.0, 5.4))
        for name, table in curves.items():
            colour, style = RANKING_STYLES.get(name, (SERIES, "-"))
            label = RANKING_LABELS.get(name, name)
            x = table["share"].to_numpy()
            if name == "random":
                final = float(table["qini"].iloc[-1])
                ax.plot([0, 1], [0, final], color=colour, lw=1.2, ls=style, label=label)
                continue
            is_model = name.endswith("_learner")
            if is_model:
                ax.fill_between(
                    x, table["qini_low"], table["qini_high"], color=colour, alpha=0.12, lw=0
                )
            ax.plot(
                x, table["qini"], color=colour, lw=1.8 if is_model else 1.4, ls=style, label=label
            )
        ax.set_xlabel("Share of customers targeted")
        ax.set_ylabel("Incremental responses among the treated")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper left", ncol=2)
        finish(fig, f"{prefix}{finding}", "Qini curves; bands are paired bootstrap intervals.")
    return save(fig, path)


POLICY_STYLES: dict[str, tuple[str, str]] = {
    "uplift": (BLUE, "-"),
    "response_model": (ORANGE, "--"),
    "recency_rule": (AQUA, "-."),
    "random": (MUTED, ":"),
}
POLICY_LABELS: dict[str, str] = {
    "uplift": "uplift policy (X-learner)",
    "response_model": "response model",
    "random": "random targeting",
    "recency_rule": "recency rule",
}
POLICY_PROSE: dict[str, str] = {
    "response_model": "the response model",
    "random": "random targeting",
    "recency_rule": "the recency rule",
}
POLICY_PANELS: tuple[tuple[str, str, str], ...] = (
    ("incremental_visits", "Incremental visits", "visit"),
    ("incremental_spend", "Incremental spend, $", "spend"),
)


def _policy_finding(comparison: pd.DataFrame, budget: int, policies: list[str]) -> str:
    """Whether the uplift policy is separated, at the budget, from the policies drawn."""
    rows = comparison[
        (comparison["reference"] == "uplift")
        & comparison["policy"].isin([p for p in POLICY_PROSE if p in policies])
        & comparison["outcome"].isin([o for _, _, o in POLICY_PANELS])
    ]
    beats, loses, noise = [], [], []
    for _, r in rows.iterrows():
        item = f"{POLICY_PROSE[r['policy']]} on {SHORT_METRIC_LABELS[r['outcome']]}"
        if not r["meaningful"]:
            noise.append(item)
        elif r["reference_better"]:
            beats.append(item)
        else:
            loses.append(item)
    parts = []
    if beats:
        parts.append(f"beats {_join(beats)}")
    if loses:
        parts.append(f"trails {_join(loses)}")
    if noise:
        references = sorted({i.split(" on ")[0] for i in noise})
        parts.append(
            "is within noise of "
            + _join(references)
            + (" on every outcome shown" if not beats and not loses else " otherwise")
        )
    return f"At {budget:,} emails the uplift policy " + "; ".join(parts)


def plot_budget_curve(
    curve: pd.DataFrame,
    path: Path,
    budget: int = config.BUDGET,
    comparison: pd.DataFrame | None = None,
) -> Path:
    """Incremental visits and spend against the number of customers emailed, per policy.

    Parameters
    ----------
    curve : pandas.DataFrame
        Output of :func:`abkit.uplift.policy.budget_curve`: columns ``policy``,
        ``n_emailed``, ``incremental_visits`` and ``incremental_spend``.
    path : Path
        Output PNG path.
    budget : int
        Email budget drawn as a vertical line.
    comparison : pandas.DataFrame, optional
        Paired policy comparisons (columns ``reference``, ``policy``, ``outcome``,
        ``meaningful``, ``reference_better``); when given, the title says whether the uplift
        policy is separated from the reference policies at the budget, and never more.

    Returns
    -------
    Path
        The path written.
    """
    policies = list(dict.fromkeys(curve["policy"]))
    title = (
        _policy_finding(comparison, budget, policies)
        if comparison is not None
        else f"Incremental visits and spend against emails sent (budget {budget:,})"
    )
    with styled():
        fig = Figure(figsize=(11.0, 4.8), layout="constrained")
        axes = fig.subplots(1, 2).ravel()
        for ax, (col, label, _) in zip(axes, POLICY_PANELS, strict=True):
            for name, part in curve.groupby("policy", sort=False):
                colour, style = POLICY_STYLES.get(name, (SERIES, "-"))
                ax.plot(
                    part["n_emailed"],
                    part[col],
                    color=colour,
                    lw=1.8 if name == "uplift" else 1.4,
                    ls=style,
                    label=POLICY_LABELS.get(name, name),
                )
            ax.axvline(budget, color=TEXT_SECONDARY, lw=1.0, ls=":")
            ax.annotate(
                f"budget {budget:,}",
                (budget, 0),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
                color=TEXT_SECONDARY,
            )
            ax.set_title(f"{label} by emails sent")
            ax.set_xlabel("Customers emailed")
            ax.set_ylabel(label)
            ax.set_xlim(0, float(curve["n_emailed"].max()))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        axes[0].legend(loc="lower right")
        finish(
            fig,
            title,
            "Point estimates from the committed cross-fitting seed; paired bootstrap "
            "intervals are in tables/policy_comparison.csv.",
        )
    return save(fig, path)
