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
