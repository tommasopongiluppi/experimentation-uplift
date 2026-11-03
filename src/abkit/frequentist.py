"""Frequentist tests: SRM check, z-tests, Welch t-tests, CIs and multiplicity corrections."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import stats

from abkit import config


@dataclass(frozen=True)
class SRMResult:
    """Result of a sample ratio mismatch test.

    Attributes
    ----------
    counts, expected : dict
        Observed and expected customers per arm.
    statistic : float
        Pearson chi-square statistic.
    dof : int
        Degrees of freedom (number of arms minus one).
    p_value : float
        Upper-tail chi-square p-value.
    alpha : float
        Threshold below which the check fails.
    passed : bool
        True when ``p_value >= alpha``.
    """

    counts: dict[str, int]
    expected: dict[str, float]
    statistic: float
    dof: int
    p_value: float
    alpha: float
    passed: bool


def srm_test(
    counts: Mapping[str, int],
    expected_shares: Mapping[str, float] | None = None,
    alpha: float = config.SRM_ALPHA,
) -> SRMResult:
    """Chi-square goodness-of-fit test of arm counts against the planned allocation.

    The statistic is sum over arms of (observed - expected)^2 / expected, compared with a
    chi-square distribution with (number of arms - 1) degrees of freedom.

    Parameters
    ----------
    counts : mapping of str to int
        Observed customers per arm.
    expected_shares : mapping of str to float, optional
        Planned share per arm; defaults to equal shares. Must cover the same arms and sum to 1.
    alpha : float
        Failure threshold; defaults to ``config.SRM_ALPHA`` (0.001, analysis plan §5).

    Returns
    -------
    SRMResult
        Counts, expected counts, statistic, p-value and pass/fail.

    Raises
    ------
    ValueError
        If fewer than two arms are given, a count is negative, or the shares are invalid.
    """
    arms = list(counts)
    if len(arms) < 2:
        raise ValueError("Need at least two arms.")
    observed = np.array([counts[a] for a in arms], dtype=float)
    if (observed < 0).any():
        raise ValueError("Counts must be non-negative.")
    if expected_shares is None:
        shares = np.full(len(arms), 1.0 / len(arms))
    else:
        if set(expected_shares) != set(arms):
            raise ValueError("expected_shares must have the same arms as counts.")
        shares = np.array([expected_shares[a] for a in arms], dtype=float)
        if (shares <= 0).any() or not np.isclose(shares.sum(), 1.0):
            raise ValueError("expected_shares must be positive and sum to 1.")
    expected = observed.sum() * shares
    statistic = float(((observed - expected) ** 2 / expected).sum())
    dof = len(arms) - 1
    p_value = float(stats.chi2.sf(statistic, dof))
    return SRMResult(
        counts={a: int(n) for a, n in zip(arms, observed, strict=True)},
        expected={a: float(e) for a, e in zip(arms, expected, strict=True)},
        statistic=statistic,
        dof=dof,
        p_value=p_value,
        alpha=alpha,
        passed=p_value >= alpha,
    )
