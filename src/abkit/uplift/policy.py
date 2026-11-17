"""Budget-constrained targeting policy: who gets which email, and what each policy is worth.

A policy assigns every one of the 64,000 customers an arm: ``mens``, ``womens`` or
``control`` (no email). Its value is estimated from the experiment by inverse propensity
weighting (IPW) with the known design probability ``p = 1/3`` per arm::

    V(pi) = (1 / n) sum_i Y_i 1{A_i = pi(x_i)} / p

and by the self-normalised version, which divides by the realised weight instead of ``n``::

    V_sn(pi) = sum_i Y_i 1{A_i = pi(x_i)} / p  /  sum_i 1{A_i = pi(x_i)} / p

Every policy is a fixed function of the pre-treatment variables and of out-of-fold uplift
scores, so ``1{A_i = pi(x_i)}`` is independent of ``Y_i`` given ``x_i`` and both estimators
are consistent. Values are reported as increments over emailing nobody, in total over the
population, per email sent, and as a share of the incremental effect of emailing everyone
with the A/B winner. Intervals come from a customer-level bootstrap with the same resamples
for every policy, so differences between policies are paired.

Policies (``POLICIES``): email nobody; everyone gets the mens email; everyone gets the womens
email; a random 10,000 get the A/B winner (the first 10,000 of one random permutation of the
customers, which also orders the random line of the budget curve); the recency rule (most
recent purchasers) sends the A/B winner to 10,000; the response model emails the 10,000 most
likely responders with the email under which they are more likely to visit; the uplift
policy gives each customer the email with the higher predicted uplift and emails the 10,000
with the highest such uplift (if positive); the exploratory mens-by-default policy holds the
mens uplift at its average and uses the womens model only where it predicts more; the
suppression policy emails everyone with their better email except the 10,000 lowest
predicted uplifts.

The uplift scores are the X-learner's visit scores for both emails. For the mens email no
ranking beat random in the uplift-model evaluation, so picking a "best" mens learner by its
Qini coefficient would select on noise; the same learner is used for both pairs. The scores
were trained on visits, so the conversion and spend values measure how well a visit-based
targeting carries over to those outcomes.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abkit import config, plots
from abkit.frequentist import _quantile_mcse, diff_in_proportions
from abkit.report import _json_value, json_records, update_results, write_table
from abkit.uplift.learners import recency_rule_score
from abkit.uplift.models import SCORES_NAME, load_customers

logger = logging.getLogger(__name__)

LABEL: str = "policy"
POLICY_LEARNER: str = "x_learner"
POLICY_OUTCOME: str = "visit"
LEARNER_REASON: str = (
    "X-learner visit scores for both emails: best and meaningfully above the baselines for the "
    "womens email; for the mens email no ranking beat random, so a 'best' mens learner chosen by "
    "its Qini coefficient would be selected on noise."
)
POLICIES: tuple[str, ...] = (
    "nobody",
    "all_mens",
    "all_womens",
    "random",
    "recency_rule",
    "response_model",
    "uplift",
    "uplift_mens_default",
    "suppression",
)
POLICY_LABELS: dict[str, str] = {
    "nobody": "email nobody",
    "all_mens": "everyone gets the mens email",
    "all_womens": "everyone gets the womens email",
    "random": "random 10,000 get the A/B winner",
    "recency_rule": "recency rule: 10,000 most recent get the A/B winner",
    "response_model": "response model: 10,000 most likely responders",
    "uplift": "uplift policy: top 10,000 by predicted uplift, better email each",
    "uplift_mens_default": "mens by default, womens where its predicted uplift is higher",
    "suppression": "everyone except the 10,000 lowest predicted uplifts",
}
EXPLORATORY: tuple[str, ...] = ("uplift_mens_default",)
BUDGETED: tuple[str, ...] = (
    "random",
    "recency_rule",
    "response_model",
    "uplift",
    "uplift_mens_default",
)
#: Policies every other policy is compared with by a paired bootstrap difference. Random
#: targeting is a reference so that each ranking can be judged against no ranking at all.
REFERENCES: tuple[str, ...] = ("uplift", "recency_rule", "random")
BUDGET_STEP: int = 1_000
N_RESAMPLES: int = 1_000
OUTCOMES: tuple[str, ...] = config.OUTCOME_COLS
PROFILE_SHARES: dict[str, tuple[str, ...]] = {
    "zip_code": tuple(config.CATEGORY_MAPS["zip_code"].values()),
    "channel": tuple(config.CATEGORY_MAPS["channel"].values()),
}
TABLE_NAME: str = "policy"
COMPARISON_TABLE_NAME: str = "policy_comparison"
CURVE_TABLE_NAME: str = "policy_budget_curve"
PROFILE_TABLE_NAME: str = "policy_profiles"
CALIBRATION_TABLE_NAME: str = "policy_calibration"
SUPPRESSED_TABLE_NAME: str = "policy_suppressed_uplift"
FIGURE_NAME: str = "policy_budget_curve.png"
DEVIATION: str = (
    "D8: mens-by-default policy (exploratory) and the level-calibration check added after the "
    "uplift results were seen; the same X-learner is used for both emails"
)


# --- inputs ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyInputs:
    """Everything a policy needs, aligned by row over all customers.

    Attributes
    ----------
    customer_id : numpy.ndarray
        Identifiers.
    arm : numpy.ndarray
        Realised arm per customer ("mens", "womens", "control").
    y : pandas.DataFrame
        Outcome columns ``visit``, ``conversion``, ``spend``.
    tau : pandas.DataFrame
        Predicted uplift per email: columns ``mens`` and ``womens``.
    response : pandas.DataFrame
        Predicted response probability under each email: columns ``mens`` and ``womens``.
    pre : pandas.DataFrame
        Pre-treatment columns (for the recency rule and the profiles).
    """

    customer_id: np.ndarray
    arm: np.ndarray
    y: pd.DataFrame
    tau: pd.DataFrame
    response: pd.DataFrame
    pre: pd.DataFrame

    @property
    def n(self) -> int:
        """Number of customers."""
        return int(self.customer_id.size)


def load_policy_inputs(
    db_path: Path = config.DUCKDB_PATH,
    scores_path: Path = config.DATA_PROCESSED / SCORES_NAME,
    learner: str = POLICY_LEARNER,
    outcome: str = POLICY_OUTCOME,
) -> PolicyInputs:
    """Join the cleaned customers with the out-of-fold scores of both pairs.

    Parameters
    ----------
    db_path : Path
        DuckDB database with the ``customers`` table.
    scores_path : Path
        Parquet written by the uplift-model stage, with every customer scored in each pair.
    learner : str
        Score column used as the predicted uplift.
    outcome : str
        Outcome the scores were trained on.

    Returns
    -------
    PolicyInputs

    Raises
    ------
    ValueError
        If any customer lacks a score for either email.
    """
    customers = load_customers(db_path)
    scores = pd.read_parquet(scores_path)
    scores = scores[scores["outcome"] == outcome]
    ids = customers["customer_id"].to_numpy()
    tau, response = {}, {}
    for email in config.TREATMENTS:
        part = scores[scores["arm"] == email].set_index("customer_id")
        missing = np.setdiff1d(ids, part.index.to_numpy())
        if missing.size:
            raise ValueError(
                f"{missing.size} customers have no {email} score (first {missing[:3]})."
            )
        tau[email] = part[learner].reindex(ids).to_numpy()
        response[email] = part["response_model"].reindex(ids).to_numpy()
    return PolicyInputs(
        customer_id=ids,
        arm=customers[config.ARM_COL].to_numpy().astype(str),
        y=customers[list(OUTCOMES)].reset_index(drop=True).astype(float),
        tau=pd.DataFrame(tau),
        response=pd.DataFrame(response),
        pre=customers[list(config.PRE_TREATMENT_COLS)].reset_index(drop=True),
    )


def ab_winner(results_path: Path = config.RESULTS_JSON) -> tuple[str, str]:
    """Return the A/B winner by the analysis plan's rule, read from ``results.json``.

    The email that beats control on spend in the primary frequentist family; if both do, the
    one with the larger estimated difference; if neither does, the Bayesian spend readout's
    recommended email. When no email meets either rule (possible on small or synthetic data,
    never on the real experiment), the email ahead in the Bayesian head-to-head on spend is
    used so the targeting policies still run, and the basis says that no winner was declared.

    Parameters
    ----------
    results_path : Path
        ``results.json`` holding the ``frequentist`` and ``bayes_spend`` sections.

    Returns
    -------
    tuple of str
        ``(email, basis)``.
    """
    results = json.loads(results_path.read_text())
    rows = [
        r
        for r in results["frequentist"]["rows"]
        if r["metric"] == "spend"
        and r["family"] == "primary"
        and r["treatment"] in config.TREATMENTS
    ]
    winners = [r for r in rows if r["significant"]]
    if len(winners) == 2:
        best = max(winners, key=lambda r: r["diff"])
        return best[
            "treatment"
        ], "both emails beat control on spend; larger estimated difference wins"
    if len(winners) == 1:
        return winners[0]["treatment"], "the only email that beats control on spend"
    recommendation = results["bayes_spend"]["recommendation"]
    email = recommendation["recommended_email"]
    if email is not None:
        return email, "neither email beats control on spend; higher posterior mean spend wins"
    p_mens = recommendation["p_mens_better_than_womens"]
    email = "mens" if p_mens is None or p_mens >= 0.5 else "womens"
    return email, (
        "no email meets either rule against control; no winner declared, the email ahead "
        "in the head-to-head posterior on spend is used"
    )


# --- policy value ------------------------------------------------------------------------------


def policy_value(
    assignment: np.ndarray,
    arm: np.ndarray,
    y: np.ndarray,
    p: float = config.ARM_PROBABILITY,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """IPW and self-normalised IPW value of a policy, per outcome column.

    Parameters
    ----------
    assignment : numpy.ndarray
        Arm the policy assigns to each customer.
    arm : numpy.ndarray
        Arm each customer was actually randomised to.
    y : numpy.ndarray
        Outcomes, shape ``(n,)`` or ``(n, m)``.
    p : float
        Known assignment probability of every arm.
    weights : numpy.ndarray, optional
        Non-negative resampling weights (bootstrap).

    Returns
    -------
    tuple of numpy.ndarray
        ``(ipw, self_normalised)``, each of shape ``(m,)``.
    """
    assignment, arm = np.asarray(assignment), np.asarray(arm)
    y = np.asarray(y, dtype=float)
    y = y[:, None] if y.ndim == 1 else y
    w = np.ones(arm.size) if weights is None else np.asarray(weights, dtype=float)
    match = (assignment == arm).astype(float) / p
    num = (w * match) @ y
    return num / w.sum(), num / (w @ match)


def top_k(score: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the ``k`` highest scores (ties broken by row order)."""
    mask = np.zeros(score.size, dtype=bool)
    if k > 0:
        mask[np.argsort(-np.asarray(score), kind="stable")[:k]] = True
    return mask


def mens_constant(inputs: PolicyInputs, outcome: str = POLICY_OUTCOME) -> float:
    """Return the mens email's average effect on ``outcome`` in its pair (difference in means)."""
    y = inputs.y[outcome].to_numpy()
    return float(y[inputs.arm == "mens"].mean() - y[inputs.arm == config.CONTROL].mean())


def build_policies(
    inputs: PolicyInputs,
    winner: str,
    rng: np.random.Generator,
    budget: int = config.BUDGET,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Assign an arm to every customer under each policy in ``POLICIES``.

    Parameters
    ----------
    inputs : PolicyInputs
        Scores, arms and pre-treatment columns.
    winner : str
        The A/B winner email, sent by the random, recency-rule and all-winner policies.
    rng : numpy.random.Generator
        Draws the one random permutation of the customers that defines random targeting:
        the random policy emails its first ``budget`` customers, and the budget curve its
        first ``k`` at every budget ``k``.
    budget : int
        Maximum emails for the budgeted policies; the suppression policy withholds this many.

    Returns
    -------
    tuple
        ``(assignments, info)``: arm label per customer for each policy, and ``info`` with
        the masks, constants and ranking scores used (``targeted``, ``suppressed``,
        ``mens_constant``, and one score per ranking of :func:`budget_curve`, where a
        higher score is emailed first).
    """
    n, control = inputs.n, config.CONTROL
    emails = np.array(config.TREATMENTS)
    tau = inputs.tau[list(config.TREATMENTS)].to_numpy()
    best_email = emails[np.argmax(tau, axis=1)]
    best_tau = tau.max(axis=1)

    def send(mask: np.ndarray, email: np.ndarray | str) -> np.ndarray:
        out = np.full(n, control, dtype=object)
        out[mask] = email if isinstance(email, str) else email[mask]
        return out.astype(str)

    random_order = rng.permutation(n)
    random_mask = np.zeros(n, dtype=bool)
    random_mask[random_order[:budget]] = True
    random_score = np.empty(n)
    random_score[random_order] = np.arange(n, 0, -1)  # first in the permutation ranks highest
    recency = recency_rule_score(
        inputs.pre["recency"].to_numpy(), inputs.pre["history"].to_numpy(), inputs.customer_id
    )
    resp = inputs.response[list(config.TREATMENTS)].to_numpy()
    resp_email, resp_best = emails[np.argmax(resp, axis=1)], resp.max(axis=1)
    uplift_mask = top_k(best_tau, budget) & (best_tau > 0)
    const = mens_constant(inputs)
    womens_tau = inputs.tau["womens"].to_numpy()
    default_email = np.where(womens_tau > const, "womens", "mens")
    default_tau = np.maximum(womens_tau, const)
    default_mask = top_k(default_tau, budget) & (default_tau > 0)
    suppressed = top_k(-best_tau, budget)

    assignments = {
        "nobody": np.full(n, control),
        "all_mens": np.full(n, "mens"),
        "all_womens": np.full(n, "womens"),
        "random": send(random_mask, winner),
        "recency_rule": send(top_k(recency, budget), winner),
        "response_model": send(top_k(resp_best, budget), resp_email),
        "uplift": send(uplift_mask, best_email),
        "uplift_mens_default": send(default_mask, default_email),
        "suppression": send(~suppressed, best_email),
    }
    info = {
        "targeted": uplift_mask,
        "suppressed": suppressed,
        "mens_constant": const,
        "best_email": best_email,
        "best_tau": best_tau,
        "response_email": resp_email,
        "response_best": resp_best,
        "recency_score": recency,
        "random_score": random_score,
    }
    return assignments, info


def email_counts(assignment: np.ndarray) -> dict[str, int]:
    """Count the customers assigned to each email and in total."""
    counts = {email: int((assignment == email).sum()) for email in config.TREATMENTS}
    return {"n_emailed": sum(counts.values()), **{f"n_{e}": c for e, c in counts.items()}}


# --- evaluation --------------------------------------------------------------------------------


def _interval(reps: np.ndarray, alpha: float) -> tuple[np.ndarray, ...]:
    """Percentile interval ends and their MCSE along axis 0; NaN where a column is undefined."""
    shape = reps.shape[1:]
    cols = reps.reshape(reps.shape[0], -1).T
    out = np.full((4, cols.shape[0]), np.nan)
    for j, col in enumerate(cols):
        finite = np.sort(col[np.isfinite(col)])
        if finite.size == 0:
            continue
        out[:, j] = (
            np.quantile(finite, alpha / 2),
            np.quantile(finite, 1 - alpha / 2),
            _quantile_mcse(finite, alpha / 2),
            _quantile_mcse(finite, 1 - alpha / 2),
        )
    return tuple(o.reshape(shape) for o in out)


def evaluate_policies(
    inputs: PolicyInputs,
    assignments: dict[str, np.ndarray],
    suppressed: np.ndarray,
    best_email: np.ndarray,
    winner: str,
    rng: np.random.Generator,
    n_resamples: int = N_RESAMPLES,
    cost_per_email: float = config.COST_PER_EMAIL,
    alpha: float = config.ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Value of every policy with paired bootstrap intervals, comparisons and the suppressed group.

    Parameters
    ----------
    inputs : PolicyInputs
        Arms and outcomes.
    assignments : dict of str to numpy.ndarray
        From :func:`build_policies`.
    suppressed : numpy.ndarray
        Mask of the customers the suppression policy withholds.
    best_email : numpy.ndarray
        Each customer's better email by predicted uplift.
    winner : str
        The A/B winner; "everyone gets the winner" is the reference for the share column.
    rng : numpy.random.Generator
        Bootstrap resamples.
    n_resamples : int
        Number of resamples.
    cost_per_email : float
        Cost per email, subtracted from incremental spend to give the net value.
    alpha : float
        1 - interval level.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(policy_table, comparison_table, suppressed_table)``.
    """
    names = list(assignments)
    n, p = inputs.n, config.ARM_PROBABILITY
    Y = inputs.y[list(OUTCOMES)].to_numpy()
    M = np.stack([(assignments[k] == inputs.arm).astype(float) / p for k in names])  # (P, n)
    counts = {k: email_counts(assignments[k]) for k in names}
    n_emailed = np.array([counts[k]["n_emailed"] for k in names], dtype=float)
    i_nobody, i_all = names.index("nobody"), names.index(f"all_{winner}")
    s_best = (suppressed & (inputs.arm == best_email)).astype(float) / p
    s_ctrl = (suppressed & (inputs.arm == config.CONTROL)).astype(float) / p
    i_spend = list(OUTCOMES).index("spend")

    def evaluate(w: np.ndarray) -> dict[str, np.ndarray]:
        num = (M * w) @ Y  # (P, m): sum of w * match * y / p
        ipw = num / w.sum()
        inc = ipw - ipw[i_nobody]
        total = n * inc
        with np.errstate(divide="ignore", invalid="ignore"):
            sn = num / (M @ w)[:, None]
            per_email = np.where(n_emailed[:, None] > 0, total / n_emailed[:, None], np.nan)
            share = inc / inc[i_all]
        net = total[:, i_spend] - cost_per_email * n_emailed
        sup = ((w * s_best) @ Y - (w * s_ctrl) @ Y) / (w @ suppressed.astype(float))
        return {
            "ipw": ipw,
            "sn": sn,
            "inc_sn": n * (sn - sn[i_nobody]),
            "total": total,
            "per_email": per_email,
            "share": share,
            "net": net,
            "suppressed": sup,
        }

    est = evaluate(np.ones(n))
    reps = {k: np.empty((n_resamples, *v.shape)) for k, v in est.items()}
    probs = np.full(n, 1.0 / n)
    for b in range(n_resamples):
        w = rng.multinomial(n, probs).astype(float)
        for k, v in evaluate(w).items():
            reps[k][b] = v
    ci = {k: _interval(v, alpha) for k, v in reps.items()}

    rows = []
    for i, name in enumerate(names):
        for j, outcome in enumerate(OUTCOMES):
            rows.append(
                {
                    "policy": name,
                    "label": POLICY_LABELS.get(name, name),
                    "exploratory": name in EXPLORATORY,
                    **counts[name],
                    "outcome": outcome,
                    "value_ipw": est["ipw"][i, j],
                    "value_sn": est["sn"][i, j],
                    "incremental_total": est["total"][i, j],
                    "ci_low": ci["total"][0][i, j],
                    "ci_high": ci["total"][1][i, j],
                    "mcse_low": ci["total"][2][i, j],
                    "mcse_high": ci["total"][3][i, j],
                    "incremental_total_sn": est["inc_sn"][i, j],
                    "sn_ci_low": ci["inc_sn"][0][i, j],
                    "sn_ci_high": ci["inc_sn"][1][i, j],
                    "per_email": est["per_email"][i, j],
                    "per_email_ci_low": ci["per_email"][0][i, j],
                    "per_email_ci_high": ci["per_email"][1][i, j],
                    "share_of_all_winner": est["share"][i, j],
                    "share_ci_low": ci["share"][0][i, j],
                    "share_ci_high": ci["share"][1][i, j],
                    "net_value": est["net"][i] if outcome == "spend" else np.nan,
                    "net_ci_low": ci["net"][0][i] if outcome == "spend" else np.nan,
                    "net_ci_high": ci["net"][1][i] if outcome == "spend" else np.nan,
                    "cost_per_email": cost_per_email,
                    "n_resamples": n_resamples,
                }
            )
    policy_table = pd.DataFrame(rows)

    comp = []
    for ref in REFERENCES:
        r = names.index(ref)
        for i, name in enumerate(names):
            if name == ref:
                continue
            d = reps["total"][:, r, :] - reps["total"][:, i, :]
            lo, hi = np.nanquantile(d, [alpha / 2, 1 - alpha / 2], axis=0)
            for j, outcome in enumerate(OUTCOMES):
                comp.append(
                    {
                        "reference": ref,
                        "policy": name,
                        "outcome": outcome,
                        "diff_incremental_total": est["total"][r, j] - est["total"][i, j],
                        "ci_low": lo[j],
                        "ci_high": hi[j],
                        "meaningful": bool(lo[j] > 0 or hi[j] < 0),
                        "reference_better": bool(lo[j] > 0),
                    }
                )
    comparison_table = pd.DataFrame(comp)

    sup_rows = [
        {
            "outcome": outcome,
            "n_suppressed": int(suppressed.sum()),
            "uplift_per_customer": est["suppressed"][j],
            "ci_low": ci["suppressed"][0][j],
            "ci_high": ci["suppressed"][1][j],
            "at_or_below_zero": bool(est["suppressed"][j] <= 0),
            "ci_excludes_positive": bool(ci["suppressed"][1][j] <= 0),
        }
        for j, outcome in enumerate(OUTCOMES)
    ]
    return policy_table, comparison_table, pd.DataFrame(sup_rows)


def budget_curve(
    inputs: PolicyInputs,
    info: dict[str, Any],
    winner: str,
    step: int = BUDGET_STEP,
    cost_per_email: float = config.COST_PER_EMAIL,
) -> pd.DataFrame:
    """Incremental visits and spend against the number emailed, for four rankings.

    The uplift policy, the response model, the recency rule and random targeting are each
    extended from the fixed budget to every budget on the grid: the ranking's first ``k``
    customers are emailed. The recency rule and random targeting send the A/B winner. Each
    ranking is the one behind the fixed-budget policy of the same name, so the curve at the
    fixed budget reproduces that policy's value; for random targeting this holds because
    both use the same random permutation.

    Parameters
    ----------
    inputs : PolicyInputs
        Arms and outcomes.
    info : dict
        From :func:`build_policies` (rankings and email choices).
    winner : str
        Email sent by the recency rule and by random targeting.
    step : int
        Budget grid step; the grid runs from 0 to the population in steps of ``step``.
    cost_per_email : float
        For the net-value column.

    Returns
    -------
    pandas.DataFrame
        Columns ``policy``, ``n_emailed``, ``incremental_visits``, ``incremental_spend``,
        ``net_value``.
    """
    n = inputs.n
    y = inputs.y[["visit", "spend"]].to_numpy()
    nobody, _ = policy_value(np.full(n, config.CONTROL), inputs.arm, y)
    rankings = {
        "uplift": (info["best_tau"], info["best_email"]),
        "response_model": (info["response_best"], info["response_email"]),
        "recency_rule": (info["recency_score"], np.full(n, winner)),
        "random": (info["random_score"], np.full(n, winner)),
    }
    grid = [*range(0, n, step), n]
    rows = []
    for name, (score, email) in rankings.items():
        order = np.argsort(-score, kind="stable")
        for k in grid:
            assignment = np.full(n, config.CONTROL, dtype=object)
            assignment[order[:k]] = email[order[:k]]
            v, _ = policy_value(assignment.astype(str), inputs.arm, y)
            inc = n * (v - nobody)
            rows.append(
                {
                    "policy": name,
                    "n_emailed": k,
                    "incremental_visits": inc[0],
                    "incremental_spend": inc[1],
                    "net_value": inc[1] - cost_per_email * k,
                }
            )
    return pd.DataFrame(rows)


def calibration_check(inputs: PolicyInputs, outcome: str = POLICY_OUTCOME) -> pd.DataFrame:
    """Mean predicted uplift against the observed average effect in each pair.

    The uplift policy compares predicted levels across two models, so both must be on the
    observed scale. For each email: the mean predicted uplift over the pair's customers,
    the observed difference in the outcome rate (email minus control) with a Newcombe
    interval, and whether the predicted mean lies inside it.
    """
    y = inputs.y[outcome].to_numpy()
    rows = []
    for email in config.TREATMENTS:
        in_pair = np.isin(inputs.arm, [email, config.CONTROL])
        t, c = inputs.arm == email, inputs.arm == config.CONTROL
        res = diff_in_proportions(y[t].sum(), t.sum(), y[c].sum(), c.sum())
        predicted = float(inputs.tau.loc[in_pair, email].mean())
        rows.append(
            {
                "email": email,
                "outcome": outcome,
                "n_pair": int(in_pair.sum()),
                "predicted_mean_uplift": predicted,
                "observed_ate": float(res.diff),
                "ci_low": float(res.ci_low),
                "ci_high": float(res.ci_high),
                "predicted_within_ci": bool(res.ci_low <= predicted <= res.ci_high),
                "level_gap": predicted - float(res.diff),
            }
        )
    return pd.DataFrame(rows)


def profiles(
    inputs: PolicyInputs, groups: dict[str, np.ndarray], info: dict[str, Any]
) -> pd.DataFrame:
    """Aggregate pre-treatment profile of customer groups (no individual rows).

    Parameters
    ----------
    inputs : PolicyInputs
        Pre-treatment columns and scores.
    groups : dict of str to numpy.ndarray
        Group name to boolean mask.
    info : dict
        From :func:`build_policies`, for the email assigned and predicted uplift.

    Returns
    -------
    pandas.DataFrame
        Long table: ``group``, ``variable``, ``value``.
    """
    pre = inputs.pre
    rows = []
    for group, mask in groups.items():
        sub = pre.loc[mask]
        stats = {
            "n": int(mask.sum()),
            "share_assigned_mens": float((info["best_email"][mask] == "mens").mean()),
            "share_assigned_womens": float((info["best_email"][mask] == "womens").mean()),
            "mean_predicted_uplift_best": float(info["best_tau"][mask].mean()),
            "mean_predicted_uplift_mens": float(inputs.tau.loc[mask, "mens"].mean()),
            "mean_predicted_uplift_womens": float(inputs.tau.loc[mask, "womens"].mean()),
            "mean_recency": float(sub["recency"].mean()),
            "median_recency": float(sub["recency"].median()),
            "share_recency_le_3": float((sub["recency"] <= 3).mean()),
            "mean_history": float(sub["history"].mean()),
            "median_history": float(sub["history"].median()),
            "share_mens_buyers": float(sub["mens"].mean()),
            "share_womens_buyers": float(sub["womens"].mean()),
            "share_both_buyers": float(((sub["mens"] == 1) & (sub["womens"] == 1)).mean()),
            "share_newbie": float(sub["newbie"].mean()),
        }
        for col, levels in PROFILE_SHARES.items():
            for level in levels:
                stats[f"share_{col}_{level}"] = float((sub[col] == level).mean())
        rows += [{"group": group, "variable": k, "value": v} for k, v in stats.items()]
    return pd.DataFrame(rows)


# --- stage -------------------------------------------------------------------------------------


def run_policy(
    db_path: Path = config.DUCKDB_PATH,
    reports_dir: Path = config.REPORTS,
    scores_dir: Path = config.DATA_PROCESSED,
    n_resamples: int = N_RESAMPLES,
    budget: int = config.BUDGET,
    cost_per_email: float = config.COST_PER_EMAIL,
    winner: str | None = None,
) -> pd.DataFrame:
    """Run the targeting-policy stage and write its tables, figure and results section.

    Parameters
    ----------
    db_path : Path
        DuckDB database with the ``customers`` table.
    reports_dir : Path
        Reports directory; its ``results.json`` supplies the A/B winner unless ``winner`` is
        given.
    scores_dir : Path
        Directory holding the out-of-fold scores parquet.
    n_resamples : int
        Bootstrap resamples.
    budget : int
        Email budget.
    cost_per_email : float
        Cost per email for the net value.
    winner : str, optional
        A/B winner email; read from ``results.json`` by the plan's rule when omitted.

    Returns
    -------
    pandas.DataFrame
        The policy table.
    """
    rng = np.random.default_rng(config.SEEDS["uplift"])
    inputs = load_policy_inputs(db_path, scores_dir / SCORES_NAME)
    if winner is None:
        winner, basis = ab_winner(reports_dir / "results.json")
    else:
        basis = "given"
    assignments, info = build_policies(inputs, winner, rng, budget)
    boot_rng = np.random.default_rng(config.SEEDS["bootstrap"])
    policy_table, comparison, suppressed = evaluate_policies(
        inputs,
        assignments,
        info["suppressed"],
        info["best_email"],
        winner,
        boot_rng,
        n_resamples,
        cost_per_email,
    )
    curve = budget_curve(inputs, info, winner, cost_per_email=cost_per_email)
    calibration = calibration_check(inputs)
    profile_table = profiles(
        inputs,
        {
            "targeted_uplift": info["targeted"],
            "suppressed": info["suppressed"],
            "all": np.ones(inputs.n, bool),
        },
        info,
    )
    tables, figures = reports_dir / "tables", reports_dir / "figures"
    write_table(policy_table, TABLE_NAME, tables)
    write_table(comparison, COMPARISON_TABLE_NAME, tables)
    write_table(curve, CURVE_TABLE_NAME, tables)
    write_table(profile_table, PROFILE_TABLE_NAME, tables)
    write_table(calibration, CALIBRATION_TABLE_NAME, tables)
    write_table(suppressed, SUPPRESSED_TABLE_NAME, tables)
    figure = plots.plot_budget_curve(curve, figures / FIGURE_NAME, budget, comparison)

    def headline(policy: str, outcome: str) -> dict[str, Any]:
        r = policy_table[
            (policy_table["policy"] == policy) & (policy_table["outcome"] == outcome)
        ].iloc[0]
        return {
            k: _json_value(r[k])
            for k in (
                "n_emailed", "incremental_total", "ci_low", "ci_high", "per_email",
                "share_of_all_winner", "share_ci_low", "share_ci_high", "net_value",
            )
        }  # fmt: skip

    def versus(policy: str, outcome: str) -> dict[str, Any]:
        r = comparison[
            (comparison["reference"] == "uplift")
            & (comparison["policy"] == policy)
            & (comparison["outcome"] == outcome)
        ].iloc[0]
        return {
            k: _json_value(r[k])
            for k in (
                "diff_incremental_total",
                "ci_low",
                "ci_high",
                "meaningful",
                "reference_better",
            )
        }

    update_results(
        LABEL,
        {
            "label": LABEL,
            "plan_section": "11",
            "deviation": DEVIATION,
            "learner": POLICY_LEARNER,
            "learner_outcome": POLICY_OUTCOME,
            "learner_reason": LEARNER_REASON,
            "carry_over_note": (
                "Scores were trained on visit; conversion and spend values measure how well a "
                "visit-based targeting carries over to those outcomes."
            ),
            "ab_winner": {"email": winner, "basis": basis},
            "budget": budget,
            "cost_per_email": cost_per_email,
            "assignment_probability": config.ARM_PROBABILITY,
            "n_customers": inputs.n,
            "seeds": {
                "random_policy": config.SEEDS["uplift"],
                "bootstrap": config.SEEDS["bootstrap"],
            },
            "bootstrap": {"n_resamples": n_resamples, "paired": True},
            "mens_constant_uplift": float(info["mens_constant"]),
            "policies": {name: POLICY_LABELS[name] for name in POLICIES},
            "exploratory": list(EXPLORATORY),
            "headline": {
                outcome: {policy: headline(policy, outcome) for policy in POLICIES}
                for outcome in OUTCOMES
            },
            "uplift_versus": {
                outcome: {
                    policy: versus(policy, outcome) for policy in POLICIES if policy != "uplift"
                }
                for outcome in OUTCOMES
            },
            "uplift_beats_recency_rule": {
                outcome: bool(versus("recency_rule", outcome)["reference_better"])
                for outcome in OUTCOMES
            },
            "uplift_beats_response_model": {
                outcome: bool(versus("response_model", outcome)["reference_better"])
                for outcome in OUTCOMES
            },
            "uplift_beats_random": {
                outcome: bool(versus("random", outcome)["reference_better"]) for outcome in OUTCOMES
            },
            "calibration": json_records(calibration),
            "suppressed_group": json_records(suppressed),
            "profiles": json_records(profile_table),
            "table": f"tables/{TABLE_NAME}.csv",
            "comparison_table": f"tables/{COMPARISON_TABLE_NAME}.csv",
            "budget_curve_table": f"tables/{CURVE_TABLE_NAME}.csv",
            "profiles_table": f"tables/{PROFILE_TABLE_NAME}.csv",
            "calibration_table": f"tables/{CALIBRATION_TABLE_NAME}.csv",
            "suppressed_table": f"tables/{SUPPRESSED_TABLE_NAME}.csv",
            "figure": f"figures/{figure.name}",
            "rows": json_records(policy_table),
            "comparison_rows": json_records(comparison),
        },
        reports_dir / "results.json",
    )
    logger.info("Policy tables written: %d policy rows.", len(policy_table))
    return policy_table
