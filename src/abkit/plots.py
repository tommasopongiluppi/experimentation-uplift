"""Matplotlib figures written to reports/figures, in one shared style.

Figures use the object-oriented API (``matplotlib.figure.Figure``) so no GUI backend or global
pyplot state is involved. Colours: arms use ``config.ARM_COLORS`` (categorical slots 1-3 of the
reference palette, validated for colour-vision deficiency); single-series charts use slot 1.
Every figure has a CSV table behind it in reports/tables.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from abkit import config

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = config.ARM_COLORS["mens"]  # slot 1 blue, also the single-series colour
FIGSIZE = (8.0, 4.5)
DPI = 150

STYLE: dict[str, object] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
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


def plot_log_histogram(
    hist: pd.DataFrame,
    title: str,
    xlabel: str,
    path: Path,
    note: str | None = None,
    ticks: tuple[float, ...] = (30, 50, 100, 200, 500, 1000, 2000, 5000),
) -> Path:
    """Histogram with log-spaced bins on a logarithmic x-axis.

    Parameters
    ----------
    hist : pandas.DataFrame
        Columns ``bin_left``, ``bin_right``, ``n``.
    title, xlabel : str
        Chart title and x-axis label.
    path : Path
        Output PNG path.
    note : str, optional
        Short caption shown under the title.
    ticks : tuple of float
        Candidate dollar tick positions; those inside the data range are shown.

    Returns
    -------
    Path
        The path written.
    """
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
        lo, hi = float(hist["bin_left"].min()), float(hist["bin_right"].max())
        ax.set_xticks([t for t in ticks if lo <= t <= hi * 1.05])
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="x", visible=False)
        ax.set_title(title, pad=26 if note else None)
        if note:
            ax.text(
                0,
                1.02,
                note,
                transform=ax.transAxes,
                va="bottom",
                fontsize=10,
                color=TEXT_SECONDARY,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Customers")
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
    with styled():
        fig, ax = new_figure()
        ax.bar(recency["recency"], recency["n"], width=0.7, color=SERIES)
        ax.set_xticks(recency["recency"])
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="x", visible=False)
        ax.set_title("Customers by months since last purchase (all arms pooled)")
        ax.set_xlabel("Recency (months)")
        ax.set_ylabel("Customers")
        return save(fig, path)


def plot_arm_counts(srm: pd.DataFrame, path: Path) -> Path:
    """Horizontal bars of customers per arm with the expected count marked.

    Parameters
    ----------
    srm : pandas.DataFrame
        Columns ``arm``, ``observed``, ``expected``.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        The path written.
    """
    with styled():
        fig, ax = new_figure((8.0, 3.2))
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
        ax.set_title("Customers per arm vs an equal split")
        ax.set_xlabel("Customers")
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
    labels = [
        f"{v}: {lvl}" if lvl else v
        for v, lvl in zip(balance["variable"], balance["level"], strict=True)
    ]
    y = np.arange(len(balance))[::-1]
    with styled():
        fig, ax = new_figure((8.0, 0.32 * len(balance) + 1.6))
        for arm, offset in (("mens", 0.12), ("womens", -0.12)):
            ax.scatter(
                balance[f"smd_{arm}_vs_control"],
                y + offset,
                s=36,
                color=config.ARM_COLORS[arm],
                edgecolor=SURFACE,
                linewidth=1.0,
                label=f"{arm} vs control",
                zorder=3,
            )
        for x in (-threshold, threshold):
            ax.axvline(x, color=TEXT_SECONDARY, linestyle="--", linewidth=1.0)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
        limit = max(1.5 * threshold, 1.2 * float(balance["max_abs_smd"].max()))
        ax.set_xlim(-limit, limit)
        ax.set_yticks(y, labels)
        ax.grid(axis="y", visible=False)
        ax.set_title("Covariate balance: standardised mean differences")
        ax.set_xlabel(f"Standardised mean difference (dashed: ±{threshold})")
        ax.legend(loc="lower right")
        return save(fig, path)


FOREST_METRIC_LABELS: dict[str, str] = {
    "spend": "Spend per customer",
    "spend_winsorised": "Spend, winsorised (sensitivity)",
    "conversion": "Conversion rate",
    "visit": "Visit rate",
}


def plot_forest(readout: pd.DataFrame, path: Path) -> Path:
    """Forest plot of relative lifts with their 95% intervals, grouped by metric.

    Parameters
    ----------
    readout : pandas.DataFrame
        Frequentist readout with columns ``metric``, ``treatment``, ``control``, ``rel_lift``,
        ``rel_ci_low`` and ``rel_ci_high``.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        The path written.
    """
    metrics = list(dict.fromkeys(readout["metric"]))
    comparisons = list(dict.fromkeys(zip(readout["treatment"], readout["control"], strict=True)))
    colours = {
        (t, c): (config.ARM_COLORS[t] if c == config.CONTROL else TEXT_SECONDARY)
        for t, c in comparisons
    }
    gap = 0.8  # vertical distance between comparisons within a metric group
    step = gap * len(comparisons) + 1.0
    with styled():
        fig, ax = new_figure((8.0, 0.42 * step * len(metrics) + 1.4))
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
        ax.set_title("Relative lift by metric and comparison")
        ax.legend(loc="lower right")
        return save(fig, path)


# Estimator colours for the adjusted-interval figure. Raw is the neutral reference; CUPED and
# Lin take categorical slots 7 (violet) and 4 (yellow), so they never reuse an arm's colour.
# Marker shapes are a second encoding, and the table behind the figure is the text view.
ESTIMATOR_STYLES: dict[str, tuple[str, str, str]] = {
    "raw": ("Raw (Welch)", TEXT_SECONDARY, "o"),
    "cuped": ("CUPED (history)", "#6250d6", "s"),
    "lin": ("Lin regression (all pre-treatment)", "#eda100", "D"),
}


def plot_adjusted_intervals(table: pd.DataFrame, path: Path) -> Path:
    """Small multiples of raw, CUPED and Lin-adjusted 95% intervals, one panel per metric.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`abkit.readout.build_cuped_readout`: columns ``metric``,
        ``treatment``, ``control`` and ``{raw,cuped,lin}_{diff,ci_low,ci_high}``.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        The path written.
    """
    metrics = list(dict.fromkeys(table["metric"]))
    comparisons = list(dict.fromkeys(zip(table["treatment"], table["control"], strict=True)))
    n_cols = 2
    n_rows = -(-len(metrics) // n_cols)
    gap, step = 0.25, 1.2
    with styled():
        fig = Figure(figsize=(10.0, 2.6 * n_rows + 1.0), layout="constrained")
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
            ax.set_title(FOREST_METRIC_LABELS.get(metric, metric), fontsize=11)
        for ax in axes[len(metrics) :]:
            ax.set_visible(False)
        fig.suptitle(
            "Difference vs comparison arm: raw, CUPED and regression-adjusted 95% intervals",
            x=0.01,
            ha="left",
            fontweight="bold",
            fontsize=13,
        )
        fig.legend(loc="outside lower center", ncol=len(ESTIMATOR_STYLES))
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
        is marked on its panel.
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
    with styled():
        fig = Figure(figsize=(10.0, 3.6), layout="constrained")
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
                        linewidth=1.8,
                        label=(
                            f"alpha = {alpha:g}{' (Holm)' if label == 'holm' else ''}"
                            f"{', with CUPED' if with_cuped else ''}"
                        )
                        if ax is axes[0]
                        else None,
                    )
            ax.axhline(target, color=TEXT_SECONDARY, linestyle=":", linewidth=1.0)
            ref = mdes[(mdes["metric"] == metric) & (mdes["alpha_label"] == "unadjusted")]
            ref = ref[~ref["cuped"]].iloc[0]
            x_mde = 100 * float(ref["mde_relative"])
            ax.axvline(x_mde, color=SERIES, linestyle=":", linewidth=1.0)
            ax.text(
                x_mde,
                0.03,
                f" MDE {x_mde:.0f}%",
                fontsize=9,
                color=TEXT_SECONDARY,
                va="bottom",
            )
            ax.set_ylim(0, 1.0)
            ax.set_xlim(0, 100 * float(part["relative_lift"].max()))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.set_title(POWER_METRIC_LABELS.get(metric, metric), fontsize=11)
            ax.set_xlabel("True relative lift vs control")
        axes[0].set_ylabel(f"Power (dotted: {target:.0%} target)")
        n = int(mdes["n_per_arm"].iloc[0])
        reduction = float(mdes["variance_reduction"].max())
        fig.suptitle(
            f"Power by true relative lift at {n:,} customers per arm",
            x=0.01,
            ha="left",
            fontweight="bold",
            fontsize=13,
        )
        # The note is a figure-level artist, so reserve a band under the title for it.
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.9))
        fig.text(
            0.01,
            0.9,
            f"CUPED on history removes at most {reduction:.1%} of the variance, "
            "so the dashed curves sit on the solid ones.",
            fontsize=10,
            color=TEXT_SECONDARY,
            va="bottom",
        )
        fig.legend(loc="outside lower center", ncol=2)
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
    priors = list(dict.fromkeys(zip(table["prior"], table["prior_strength"], strict=True)))
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
        fig = Figure(figsize=(10.0, 4.4), layout="constrained")
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
                    linewidth=1.8,
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
            ax.set_title(FOREST_METRIC_LABELS.get(metric, metric), fontsize=11)
            ax.set_xlabel("Prior (shared priors centred on the pooled rate)")
        axes[0].set_ylabel("P(first arm better)")
        fig.suptitle(
            "Posterior probability that the first arm is better, by prior strength",
            x=0.01,
            ha="left",
            fontweight="bold",
            fontsize=13,
        )
        fig.legend(loc="outside lower center", ncol=len(comparisons))
        return save(fig, path)


# Models in the spend-posterior figure: the primary analysis is a filled density, the
# parametric cross-check a dashed outline. Both take the comparison's colour (arm colour vs
# control, neutral for mens vs womens); line style is the second encoding.
SPEND_MODEL_LABELS: dict[str, str] = {
    "bayesian_bootstrap": "Bayesian bootstrap (primary)",
    "two_part_lognormal": "Two-part log-normal (cross-check)",
}


def plot_spend_posterior(hist: pd.DataFrame, table: pd.DataFrame, path: Path) -> Path:
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

    Returns
    -------
    Path
        The path written.
    """
    comparisons = list(dict.fromkeys(hist["comparison"]))
    models = list(dict.fromkeys(hist["model"]))
    primary = table[(table["model"] == "bayesian_bootstrap") & (table["metric"] == "spend")]
    primary = primary.set_index("comparison")
    with styled():
        fig = Figure(figsize=(8.0, 1.9 * len(comparisons) + 1.6), layout="constrained")
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
        fig.suptitle(
            "Posterior of the difference in spend per customer (bar: 95% credible interval)",
            x=0.01,
            ha="left",
            fontweight="bold",
            fontsize=13,
        )
        fig.legend(loc="outside lower center", ncol=2)
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
SEQUENTIAL_METHOD_COLORS: dict[str, str] = {
    "fixed": TEXT_SECONDARY,
    "naive": config.ARM_COLORS["womens"],
    "bayes": "#c23a8c",
    "pocock": config.ARM_COLORS["mens"],
    "obf": "#6250d6",
    "msprt": config.ARM_COLORS["control"],
}
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
    ax.plot(
        x,
        y,
        color=SEQUENTIAL_METHOD_COLORS[method],
        linewidth=1.8,
        marker="o",
        markersize=3,
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
    with styled():
        fig = Figure(figsize=(10.0, 3.8), layout="constrained")
        axes = fig.subplots(1, len(metrics), squeeze=False, sharey=True).ravel()
        for ax, metric in zip(axes, metrics, strict=True):
            part = looks[looks["metric"] == metric]
            for method in methods:
                line = part[part["method"] == method].sort_values("n_looks")
                _method_line(ax, line["n_looks"], line["reject_rate"], method, ax is axes[0])
            ax.axhline(alpha, color=TEXT_SECONDARY, linestyle=":", linewidth=1.0)
            ax.set_title(SEQUENTIAL_METRIC_LABELS.get(metric, metric), fontsize=11)
            ax.set_xlabel("Number of equally spaced looks")
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
            ax.set_ylim(0, None)
        axes[0].set_ylabel(f"False positive rate (dotted: {alpha:.0%})")
        n_runs = int(looks["n_runs"].iloc[0])
        fig.suptitle(
            f"Simulated A/A tests: false positive rate by number of looks "
            f"({n_runs:,} runs per point)",
            x=0.01,
            ha="left",
            fontweight="bold",
            fontsize=13,
        )
        fig.legend(loc="outside lower center", ncol=3)
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
    with styled():
        fig = Figure(figsize=(10.0, 3.8), layout="constrained")
        axes = fig.subplots(1, len(scenarios), squeeze=False, sharey=True).ravel()
        for ax, scenario in zip(axes, scenarios, strict=True):
            sub = part[part["scenario"] == scenario]
            for method in methods:
                line = sub[sub["method"] == method].sort_values("day")
                _method_line(ax, line["day"], line["reject_rate"], method, ax is axes[0])
            if scenario == "aa":
                ax.axhline(alpha, color=TEXT_SECONDARY, linestyle=":", linewidth=1.0)
            ax.set_title(SEQUENTIAL_SCENARIO_LABELS[scenario], fontsize=11)
            ax.set_xlabel("Day")
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
            ax.set_ylim(0, 1.0)
        axes[0].set_ylabel("Share of runs stopped by the day")
        fig.suptitle(
            f"Simulated daily looks on the {SEQUENTIAL_METRIC_LABELS[metric].lower()}: "
            "cumulative rejection by day",
            x=0.01,
            ha="left",
            fontweight="bold",
            fontsize=13,
        )
        fig.legend(loc="outside lower center", ncol=3)
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
    with styled():
        fig, ax = new_figure()
        look = boundaries["look"]
        for column, method, label in (
            ("naive", "naive", "Naive peeking: 1.96 at every look"),
            ("pocock", "pocock", "Pocock: constant boundary"),
            ("obf", "obf", "O'Brien-Fleming: c · sqrt(K / k)"),
            ("msprt_visit", "msprt", "mSPRT on visit (tau = MDE)"),
        ):
            ax.plot(
                look,
                boundaries[column],
                color=SEQUENTIAL_METHOD_COLORS[method],
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=label,
            )
        ax.set_xlabel("Look (day)")
        ax.set_ylabel("|z| needed to stop")
        ax.set_xticks(look)
        ax.set_ylim(0, None)
        n_looks = int(boundaries["n_looks"].iloc[0])
        ax.set_title(f"Two-sided 5% boundaries for {n_looks} equally spaced looks")
        ax.legend(loc="upper right")
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


def plot_segment_forest(effects: pd.DataFrame, path: Path) -> Path:
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

    Returns
    -------
    Path
        The path written.
    """
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
    with styled():
        fig = Figure(figsize=(11.0, 0.3 * len(positions) + 2.6), layout="constrained")
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
                        facecolor=colour if r["significant_fdr"] else SURFACE,
                        edgecolor=colour,
                        linewidth=1.2,
                        zorder=3,
                    )
                if ax is axes[0]:
                    ax.scatter([], [], s=30, color=colour, label=f"{treatment} vs control")
            ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.9)
            ax.set_xlabel(label)
            ax.set_title(label.split(",")[0], fontsize=11)
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
        fig.suptitle(
            "Exploratory: effect of each email vs control within pre-specified segments "
            "(95% intervals)",
            x=0.01,
            ha="left",
            fontweight="bold",
            fontsize=13,
        )
        fig.legend(loc="outside lower center", ncol=4)
        return save(fig, path)


def plot_uplift_curves(curves: pd.DataFrame, path: Path, title: str = "") -> Path:
    """Uplift and Qini curves with bootstrap bands against the random-targeting line.

    Parameters
    ----------
    curves : pandas.DataFrame
        ``curves`` table of :class:`abkit.uplift.evaluation.BootstrapEvaluation`: columns
        ``share``, ``uplift``, ``uplift_low``, ``uplift_high``, ``qini``, ``qini_low``,
        ``qini_high``.
    path : Path
        Output PNG path.
    title : str
        Prefix for the panel titles, for example the pair being evaluated.

    Returns
    -------
    Path
        The path written.
    """
    panels = (("uplift", "Uplift curve"), ("qini", "Qini curve"))
    with styled():
        fig = Figure(figsize=(11.0, 4.5), layout="constrained")
        axes = fig.subplots(1, 2).ravel()
        for ax, (col, label) in zip(axes, panels, strict=True):
            x = curves["share"].to_numpy()
            ax.fill_between(
                x, curves[f"{col}_low"], curves[f"{col}_high"], color=SERIES, alpha=0.18, lw=0
            )
            ax.plot(x, curves[col], color=SERIES, lw=1.8, label="model ranking")
            final = float(curves[col].iloc[-1])
            ax.plot([0, 1], [0, final], color=TEXT_SECONDARY, lw=1.0, ls="--", label="random")
            ax.set_title(f"{title}{label}" if not title else f"{title}: {label}")
            ax.set_xlabel("share of customers targeted")
            ax.set_ylabel("incremental responses")
            ax.set_xlim(0, 1)
        axes[0].legend(loc="upper left")
    return save(fig, path)


RANKING_COLORS: dict[str, str] = {
    "s_learner": "#8e6bd6",
    "t_learner": config.ARM_COLORS["mens"],
    "x_learner": config.ARM_COLORS["womens"],
    "response_model": config.ARM_COLORS["control"],
    "recency_rule": "#b8860b",
    "random": TEXT_SECONDARY,
}
RANKING_LABELS: dict[str, str] = {
    "s_learner": "S-learner",
    "t_learner": "T-learner",
    "x_learner": "X-learner",
    "response_model": "response model",
    "recency_rule": "recency rule",
    "random": "random",
}


def _draw_calibration(ax: Axes, deciles: pd.DataFrame, colour: str = SERIES) -> None:
    pred, obs = deciles["predicted"].to_numpy(), deciles["observed"].to_numpy()
    err = np.vstack([obs - deciles["boot_low"], deciles["boot_high"] - obs])
    ax.errorbar(pred, obs, yerr=err, fmt="o", color=colour, ecolor=colour, capsize=3, lw=1.2)
    for d, xp, yo in zip(deciles["decile"], pred, obs, strict=True):
        ax.annotate(str(d), (xp, yo), xytext=(5, 4), textcoords="offset points", fontsize=8)
    lo = min(float(np.nanmin(deciles["boot_low"])), float(pred.min()))
    hi = max(float(np.nanmax(deciles["boot_high"])), float(pred.max()))
    ax.plot([lo, hi], [lo, hi], color=TEXT_SECONDARY, lw=1.0, ls="--", label="perfect calibration")
    ax.set_xlabel("mean predicted uplift in decile")
    ax.set_ylabel("observed uplift (treated minus control rate)")


def plot_uplift_calibration(deciles: pd.DataFrame, path: Path, title: str = "") -> Path:
    """Predicted vs observed uplift by decile of the predicted score, with bootstrap bars.

    Parameters
    ----------
    deciles : pandas.DataFrame
        ``deciles`` table of :class:`abkit.uplift.evaluation.BootstrapEvaluation`: columns
        ``decile``, ``predicted``, ``observed``, ``boot_low``, ``boot_high``.
    path : Path
        Output PNG path.
    title : str
        Title prefix.

    Returns
    -------
    Path
        The path written.
    """
    with styled():
        fig, ax = new_figure((6.0, 5.0))
        _draw_calibration(ax, deciles)
        ax.set_title(f"{title}: calibration by decile" if title else "Calibration by decile")
        ax.legend(loc="upper left")
    return save(fig, path)


def plot_calibration_grid(deciles: dict[str, pd.DataFrame], path: Path, title: str = "") -> Path:
    """One calibration panel per learner, sharing both axes.

    Parameters
    ----------
    deciles : dict of str to pandas.DataFrame
        Learner name to its ``deciles`` table (see :func:`plot_uplift_calibration`).
    path : Path
        Output PNG path.
    title : str
        Figure title prefix, for example the pair and outcome.

    Returns
    -------
    Path
        The path written.
    """
    names = list(deciles)
    with styled():
        fig = Figure(figsize=(5.0 * len(names), 4.8), layout="constrained")
        axes = fig.subplots(1, len(names), squeeze=False, sharex=True, sharey=True).ravel()
        for ax, name in zip(axes, names, strict=True):
            _draw_calibration(ax, deciles[name], RANKING_COLORS.get(name, SERIES))
            ax.set_title(RANKING_LABELS.get(name, name))
        axes[0].legend(loc="upper left")
        fig.suptitle(f"{title}: calibration by decile" if title else "Calibration by decile")
    return save(fig, path)


def plot_qini_comparison(curves: dict[str, pd.DataFrame], path: Path, title: str = "") -> Path:
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
        Title prefix.

    Returns
    -------
    Path
        The path written.
    """
    with styled():
        fig, ax = new_figure((8.0, 5.0))
        for name, table in curves.items():
            colour, label = RANKING_COLORS.get(name, SERIES), RANKING_LABELS.get(name, name)
            x = table["share"].to_numpy()
            if name == "random":
                final = float(table["qini"].iloc[-1])
                ax.plot([0, 1], [0, final], color=colour, lw=1.0, ls=":", label=label)
                continue
            is_model = name.endswith("_learner")
            if is_model:
                ax.fill_between(
                    x, table["qini_low"], table["qini_high"], color=colour, alpha=0.12, lw=0
                )
            ax.plot(
                x,
                table["qini"],
                color=colour,
                lw=1.8 if is_model else 1.2,
                ls="-" if is_model else "--",
                label=label,
            )
        ax.set_title(f"{title}: Qini curves" if title else "Qini curves")
        ax.set_xlabel("share of customers targeted")
        ax.set_ylabel("incremental responses among the treated")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper left", ncol=2)
    return save(fig, path)


POLICY_COLORS: dict[str, str] = {
    "uplift": config.ARM_COLORS["mens"],
    "response_model": config.ARM_COLORS["control"],
    "random": TEXT_SECONDARY,
    "recency_rule": "#b8860b",
}
POLICY_LABELS: dict[str, str] = {
    "uplift": "uplift policy (X-learner)",
    "response_model": "response model",
    "random": "random targeting",
    "recency_rule": "recency rule",
}


def plot_budget_curve(curve: pd.DataFrame, path: Path, budget: int = config.BUDGET) -> Path:
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

    Returns
    -------
    Path
        The path written.
    """
    panels = (
        ("incremental_visits", "incremental visits"),
        ("incremental_spend", "incremental spend"),
    )
    with styled():
        fig = Figure(figsize=(11.0, 4.5), layout="constrained")
        axes = fig.subplots(1, 2).ravel()
        for ax, (col, label) in zip(axes, panels, strict=True):
            for name, part in curve.groupby("policy", sort=False):
                ax.plot(
                    part["n_emailed"],
                    part[col],
                    color=POLICY_COLORS.get(name, SERIES),
                    lw=1.8 if name == "uplift" else 1.2,
                    ls="-" if name == "uplift" else "--",
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
            ax.set_title(f"{label} vs emails sent")
            ax.set_xlabel("customers emailed")
            ax.set_ylabel(label)
            ax.set_xlim(0, float(curve["n_emailed"].max()))
        axes[0].legend(loc="lower right")
    return save(fig, path)
