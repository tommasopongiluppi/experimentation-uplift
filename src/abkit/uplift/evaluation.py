"""Uplift evaluation: uplift and Qini curves, their areas, uplift at k, calibration, bootstrap.

All functions take three aligned arrays: a predicted uplift ``score`` per customer, the 0/1
``treated`` indicator and a 0/1 outcome ``y``. Customers are ranked by descending score and
the curves are evaluated at every distinct score value (tied customers enter together), so
the result does not depend on row order. Shares are fractions of all customers in the pair.

Curves, with cumulative counts over the top share of customers ``n``, ``n_t`` treated with
``y_t`` responders and ``n_c`` controls with ``y_c`` responders:

- **Uplift curve:** ``(y_t / n_t - y_c / n_c) * n``, the incremental responses expected if
  everyone in the top share were treated.
- **Qini curve** (Radcliffe 2007): ``y_t - y_c * n_t / n_c``, the incremental responses among
  the treated customers in the top share.

A rate whose group is still empty counts as zero, as in scikit-uplift.

Areas use the trapezoid rule over the share axis, so they are in incremental responses
averaged over targeting depths. Each area is reported three ways: ``model``, the ``gain``
over random targeting (``model - random``, where the random curve is the straight line from
the origin to the curve's final point) and the ``normalised`` value
``(model - random) / (perfect - random)``, which is 0 for random targeting and 1 for the
perfect ranking. The perfect ranking sorts by the observed outcomes exactly as scikit-uplift's
``perfect_qini_curve`` (treated responders, then everyone else, then control responders) and
``perfect_uplift_curve`` do, so the normalised values are comparable with its
``qini_auc_score`` and ``uplift_auc_score``.

The bootstrap resamples customers with replacement and treats the scores as fixed, so its
intervals leave out model-fitting variability (analysis plan §11).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from abkit import config
from abkit.frequentist import _quantile_mcse, diff_in_proportions

N_RESAMPLES: int = 1_000
N_BINS: int = 10
UPLIFT_AT_K: tuple[float, ...] = (0.1, 0.2, 0.3)
SHARE_GRID: np.ndarray = np.round(np.linspace(0.01, 1.0, 100), 2)


# --- ranking ---------------------------------------------------------------------------------


def _check(score: np.ndarray, treated: np.ndarray, y: np.ndarray) -> None:
    if not (score.shape == treated.shape == y.shape) or score.ndim != 1:
        raise ValueError("score, treated and y must be 1-d arrays of the same length.")
    if score.size == 0:
        raise ValueError("Need at least one customer.")
    if not np.isfinite(score).all():
        raise ValueError("score must be finite.")
    for name, v in (("treated", treated), ("y", y)):
        if not np.isin(v, (0, 1)).all():
            raise ValueError(f"{name} must be 0/1.")


def _perfect_qini_score(treated: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Treated responders first, then non-responders, then control responders last."""
    return (y * treated - y * (1 - treated)).astype(float)


def _perfect_uplift_score(treated: np.ndarray, y: np.ndarray) -> np.ndarray:
    """scikit-uplift's ideal ordering for the uplift curve."""
    control_responders = int(((y == 1) & (treated == 0)).sum())
    treated_non_responders = int(((y == 0) & (treated == 1)).sum())
    summand = y if control_responders > treated_non_responders else treated
    return (2 * (y == treated) + summand).astype(float)


class _Ranking:
    """Customers sorted by one score, with the end index of every distinct-score group."""

    def __init__(self, score: np.ndarray, treated: np.ndarray, y: np.ndarray) -> None:
        self.order = np.argsort(-score, kind="stable")
        s = score[self.order]
        self.treated = treated[self.order].astype(float)
        self.y = y[self.order].astype(float)
        self.ends = np.r_[np.flatnonzero(np.diff(s) != 0), s.size - 1]

    def cumulative(self, weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Cumulative (weighted) counts at each distinct-score threshold, starting from 0."""
        w = np.ones(self.y.size) if weights is None else weights[self.order]
        wt = w * self.treated
        n = np.cumsum(w)[self.ends]
        n_t = np.cumsum(wt)[self.ends]
        y_t = np.cumsum(wt * self.y)[self.ends]
        y_c = np.cumsum((w - wt) * self.y)[self.ends]
        keep = np.diff(np.r_[0.0, n]) > 0  # drop groups that the resample left empty
        zero = np.zeros(1)
        n, n_t, y_t, y_c = (np.r_[zero, a[keep]] for a in (n, n_t, y_t, y_c))
        return {"n": n, "n_t": n_t, "n_c": n - n_t, "y_t": y_t, "y_c": y_c, "share": n / n[-1]}


def _rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def _qini_values(c: dict[str, np.ndarray]) -> np.ndarray:
    return c["y_t"] - c["y_c"] * _rate(c["n_t"], c["n_c"])


def _uplift_values(c: dict[str, np.ndarray]) -> np.ndarray:
    return (_rate(c["y_t"], c["n_t"]) - _rate(c["y_c"], c["n_c"])) * c["n"]


def _area(share: np.ndarray, values: np.ndarray) -> float:
    return float(np.trapezoid(values, share))


# --- curves ----------------------------------------------------------------------------------


def targeting_curves(
    score: np.ndarray, treated: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> pd.DataFrame:
    """Uplift and Qini curves as a function of the share of customers targeted.

    Parameters
    ----------
    score : numpy.ndarray
        Predicted uplift per customer; higher means targeted earlier.
    treated : numpy.ndarray
        0/1 treatment indicator.
    y : numpy.ndarray
        0/1 outcome.
    weights : numpy.ndarray, optional
        Non-negative resampling weights per customer (used by the bootstrap).

    Returns
    -------
    pandas.DataFrame
        One row per distinct score value plus the origin, columns ``share``, ``n``,
        ``n_treated``, ``n_control``, ``uplift`` and ``qini``, in incremental responses.
    """
    score, treated, y = (np.asarray(a) for a in (score, treated, y))
    _check(score, treated, y)
    c = _Ranking(score, treated, y).cumulative(weights)
    return pd.DataFrame(
        {
            "share": c["share"],
            "n": c["n"],
            "n_treated": c["n_t"],
            "n_control": c["n_c"],
            "uplift": _uplift_values(c),
            "qini": _qini_values(c),
        }
    )


def uplift_curve(score: np.ndarray, treated: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Return the uplift curve alone: ``share`` and ``uplift`` (see :func:`targeting_curves`)."""
    return targeting_curves(score, treated, y)[["share", "uplift"]]


def qini_curve(score: np.ndarray, treated: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Return the Qini curve alone: ``share`` and ``qini`` (see :func:`targeting_curves`)."""
    return targeting_curves(score, treated, y)[["share", "qini"]]


# --- areas -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CurveArea:
    """Area under a targeting curve, against the random and perfect references.

    Attributes
    ----------
    curve : str
        "uplift" or "qini".
    model, random, perfect : float
        Trapezoid areas over the share axis, in incremental responses averaged over
        targeting depths.
    gain : float
        ``model - random``: extra incremental responses over random targeting.
    normalised : float
        ``(model - random) / (perfect - random)``; 0 for random, 1 for the perfect ranking.
        NaN when the perfect ranking cannot beat random (no signal in the outcome).
    """

    curve: str
    model: float
    random: float
    perfect: float
    gain: float
    normalised: float


def _curve_area(
    curve: str,
    score: np.ndarray,
    treated: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> CurveArea:
    values_of = _qini_values if curve == "qini" else _uplift_values
    perfect_score = _perfect_qini_score if curve == "qini" else _perfect_uplift_score
    c_model = _Ranking(score, treated, y).cumulative(weights)
    c_perfect = _Ranking(perfect_score(treated, y), treated, y).cumulative(weights)
    v_model, v_perfect = values_of(c_model), values_of(c_perfect)
    model = _area(c_model["share"], v_model)
    random = 0.5 * float(v_model[-1])  # straight line from (0, 0) to (1, final value)
    perfect = _area(c_perfect["share"], v_perfect)
    denominator = perfect - random
    normalised = (model - random) / denominator if denominator > 0 else float("nan")
    return CurveArea(curve, model, random, perfect, model - random, normalised)


def auuc(score: np.ndarray, treated: np.ndarray, y: np.ndarray) -> CurveArea:
    """Area under the uplift curve with its random and perfect references.

    Parameters
    ----------
    score, treated, y : numpy.ndarray
        As in :func:`targeting_curves`.

    Returns
    -------
    CurveArea
        ``normalised`` matches scikit-uplift's ``uplift_auc_score``.
    """
    score, treated, y = (np.asarray(a) for a in (score, treated, y))
    _check(score, treated, y)
    return _curve_area("uplift", score, treated, y)


def qini_coefficient(score: np.ndarray, treated: np.ndarray, y: np.ndarray) -> CurveArea:
    """Area under the Qini curve with its random and perfect references.

    Parameters
    ----------
    score, treated, y : numpy.ndarray
        As in :func:`targeting_curves`.

    Returns
    -------
    CurveArea
        ``normalised`` matches scikit-uplift's ``qini_auc_score`` (``negative_effect=True``).
    """
    score, treated, y = (np.asarray(a) for a in (score, treated, y))
    _check(score, treated, y)
    return _curve_area("qini", score, treated, y)


# --- uplift at k and calibration -------------------------------------------------------------


def uplift_at_k(
    score: np.ndarray,
    treated: np.ndarray,
    y: np.ndarray,
    k: float,
    weights: np.ndarray | None = None,
) -> float:
    """Observed treated-minus-control response rate among the top ``k`` share of customers.

    The top group is the first ``floor(k * n)`` customers by descending score; ties are
    broken by row order, so prefer scores without ties.

    Parameters
    ----------
    score, treated, y : numpy.ndarray
        As in :func:`targeting_curves`.
    k : float
        Share of customers targeted, in (0, 1].
    weights : numpy.ndarray, optional
        Non-negative resampling weights per customer (used by the bootstrap).

    Returns
    -------
    float
        Rate difference; NaN if either arm is absent from the top group.
    """
    score, treated, y = (np.asarray(a) for a in (score, treated, y))
    _check(score, treated, y)
    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1], got {k}.")
    order = np.argsort(-score, kind="stable")
    w = (np.ones(score.size) if weights is None else weights)[order]
    top = np.cumsum(w) <= np.floor(k * w.sum())
    w_top, yt = w[top], y[order][top]
    wt = w_top * treated[order][top]
    wc = w_top - wt
    if wt.sum() == 0 or wc.sum() == 0:
        return float("nan")
    return float((wt * yt).sum() / wt.sum() - (wc * yt).sum() / wc.sum())


def assign_bins(score: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Equal-count bins of the predicted score, 1 = highest scores, ``n_bins`` = lowest.

    Ranks by descending score with ties broken by row order, so bins have equal counts
    (up to one customer).

    Parameters
    ----------
    score : numpy.ndarray
        Predicted uplift per customer.
    n_bins : int
        Number of bins (10 for deciles).

    Returns
    -------
    numpy.ndarray
        Bin label per customer in ``1..n_bins``.
    """
    score = np.asarray(score)
    rank = np.empty(score.size, dtype=int)
    rank[np.argsort(-score, kind="stable")] = np.arange(score.size)
    return rank * n_bins // score.size + 1


def uplift_by_decile(
    score: np.ndarray,
    treated: np.ndarray,
    y: np.ndarray,
    n_bins: int = N_BINS,
    alpha: float = config.ALPHA,
) -> pd.DataFrame:
    """Predicted vs observed uplift within equal-count bins of the predicted score.

    Parameters
    ----------
    score, treated, y : numpy.ndarray
        As in :func:`targeting_curves`.
    n_bins : int
        Number of bins; 10 for deciles.
    alpha : float
        Newcombe interval level for the observed difference.

    Returns
    -------
    pandas.DataFrame
        One row per bin: ``decile`` (1 = highest predicted uplift), ``n``, ``n_treated``,
        ``n_control``, ``predicted`` (mean score), ``rate_treated``, ``rate_control``,
        ``observed`` (rate difference), ``ci_low``, ``ci_high``.
    """
    score, treated, y = (np.asarray(a) for a in (score, treated, y))
    _check(score, treated, y)
    bins = assign_bins(score, n_bins)
    rows = []
    for b in range(1, n_bins + 1):
        m = bins == b
        t, c = m & (treated == 1), m & (treated == 0)
        res = diff_in_proportions(y[t].sum(), t.sum(), y[c].sum(), c.sum(), alpha=alpha)
        rows.append(
            {
                "decile": b,
                "n": int(m.sum()),
                "n_treated": int(t.sum()),
                "n_control": int(c.sum()),
                "predicted": float(score[m].mean()),
                "rate_treated": float(res.mean_t),
                "rate_control": float(res.mean_c),
                "observed": float(res.diff),
                "ci_low": float(res.ci_low),
                "ci_high": float(res.ci_high),
            }
        )
    return pd.DataFrame(rows)


# --- bootstrap -------------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapEvaluation:
    """Point estimates with percentile bootstrap intervals for the evaluation metrics.

    Attributes
    ----------
    scalars : pandas.DataFrame
        One row per metric: ``metric``, ``estimate``, ``ci_low``, ``ci_high``, ``mcse_low``,
        ``mcse_high`` (Monte Carlo SE of the interval ends) and ``n_resamples``. Metrics are
        ``auuc_normalised``, ``auuc_gain``, ``qini_normalised``, ``qini_gain`` and
        ``uplift_at_<k>`` for each k.
    curves : pandas.DataFrame
        Both curves on a fixed share grid: ``share``, ``uplift``, ``uplift_low``,
        ``uplift_high``, ``qini``, ``qini_low``, ``qini_high``.
    deciles : pandas.DataFrame
        :func:`uplift_by_decile` with ``boot_low`` and ``boot_high`` added.
    """

    scalars: pd.DataFrame
    curves: pd.DataFrame
    deciles: pd.DataFrame


def _interval(
    reps: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Percentile interval ends and their MCSE along axis 0 of ``reps``."""
    srt = np.sort(reps, axis=0)
    lo, hi = np.quantile(srt, alpha / 2, axis=0), np.quantile(srt, 1 - alpha / 2, axis=0)
    cols = srt.reshape(srt.shape[0], -1).T
    mcse_lo = np.array([_quantile_mcse(c, alpha / 2) for c in cols]).reshape(lo.shape)
    mcse_hi = np.array([_quantile_mcse(c, 1 - alpha / 2) for c in cols]).reshape(hi.shape)
    return lo, hi, mcse_lo, mcse_hi


def bootstrap_evaluation(
    score: np.ndarray,
    treated: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = N_RESAMPLES,
    ks: tuple[float, ...] = UPLIFT_AT_K,
    n_bins: int = N_BINS,
    share_grid: np.ndarray = SHARE_GRID,
    alpha: float = config.ALPHA,
) -> BootstrapEvaluation:
    """Percentile bootstrap intervals for the curves, their areas, uplift at k and deciles.

    Each resample draws customers with replacement (multinomial counts used as weights) and
    recomputes every metric with the scores held fixed. Bin membership is fixed from the
    original scores.

    Parameters
    ----------
    score, treated, y : numpy.ndarray
        As in :func:`targeting_curves`.
    rng : numpy.random.Generator
        Source of randomness.
    n_resamples : int
        Bootstrap resamples.
    ks : tuple of float
        Shares for uplift at k.
    n_bins : int
        Bins for the calibration table.
    share_grid : numpy.ndarray
        Shares at which the curves are interpolated for the interval bands.
    alpha : float
        1 - confidence level.

    Returns
    -------
    BootstrapEvaluation
    """
    score, treated, y = (np.asarray(a) for a in (score, treated, y))
    _check(score, treated, y)
    n = score.size
    bins = assign_bins(score, n_bins)
    ranking = _Ranking(score, treated, y)
    perfect = {
        "qini": _Ranking(_perfect_qini_score(treated, y), treated, y),
        "uplift": _Ranking(_perfect_uplift_score(treated, y), treated, y),
    }
    values_of = {"qini": _qini_values, "uplift": _uplift_values}
    metric_names = [
        "auuc_normalised",
        "auuc_gain",
        "qini_normalised",
        "qini_gain",
        *(f"uplift_at_{round(100 * k)}" for k in ks),
    ]

    def evaluate(weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = ranking.cumulative(weights)
        curve_vals, scalars = [], []
        for name in ("uplift", "qini"):
            v = values_of[name](c)
            curve_vals.append(np.interp(share_grid, c["share"], v))
            cp = perfect[name].cumulative(weights)
            model, random = _area(c["share"], v), 0.5 * float(v[-1])
            perf = _area(cp["share"], values_of[name](cp))
            den = perf - random
            scalars += [(model - random) / den if den > 0 else np.nan, model - random]
        scalars += [uplift_at_k(score, treated, y, k, weights) for k in ks]
        w = np.ones(n) if weights is None else weights
        wt, wc = w * treated, w * (1 - treated)
        rate_t = _rate(np.bincount(bins, wt * y, n_bins + 1), np.bincount(bins, wt, n_bins + 1))
        rate_c = _rate(np.bincount(bins, wc * y, n_bins + 1), np.bincount(bins, wc, n_bins + 1))
        return np.array(scalars), np.array(curve_vals), (rate_t - rate_c)[1:]

    est_s, est_c, _ = evaluate(None)
    rep_s = np.empty((n_resamples, est_s.size))
    rep_c = np.empty((n_resamples, *est_c.shape))
    rep_d = np.empty((n_resamples, n_bins))
    probs = np.full(n, 1.0 / n)
    for b in range(n_resamples):
        rep_s[b], rep_c[b], rep_d[b] = evaluate(rng.multinomial(n, probs).astype(float))

    lo_s, hi_s, mlo_s, mhi_s = _interval(rep_s, alpha)
    scalars = pd.DataFrame(
        {
            "metric": metric_names,
            "estimate": est_s,
            "ci_low": lo_s,
            "ci_high": hi_s,
            "mcse_low": mlo_s,
            "mcse_high": mhi_s,
            "n_resamples": n_resamples,
        }
    )
    lo_c, hi_c, _, _ = _interval(rep_c, alpha)
    curves = pd.DataFrame(
        {
            "share": share_grid,
            "uplift": est_c[0],
            "uplift_low": lo_c[0],
            "uplift_high": hi_c[0],
            "qini": est_c[1],
            "qini_low": lo_c[1],
            "qini_high": hi_c[1],
        }
    )
    lo_d, hi_d, _, _ = _interval(rep_d, alpha)
    deciles = uplift_by_decile(score, treated, y, n_bins, alpha).assign(
        boot_low=lo_d, boot_high=hi_d
    )
    return BootstrapEvaluation(scalars, curves, deciles)
