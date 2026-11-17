r"""Text of the generated README blocks, built from ``results.json`` and the tables it names.

Every number in ``README.md`` sits between ``<!-- RESULTS:name -->`` markers and is written
here, so the memo cannot drift from the analysis: ``python -m abkit.report update-readme``
regenerates the blocks and a test fails when the committed README differs from what
``results.json`` gives. Wording that depends on a result (which email won, whether two
policies are separated) is chosen from the flags in ``results.json``, not typed.

Dollar signs are written as ``\$`` because GitHub renders text between two bare dollar
signs as mathematics.
"""

import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

Results = dict[str, Any]
Row = dict[str, Any]

EMAIL_NAMES: dict[str, str] = {"mens": "Mens", "womens": "Womens", "control": "no email"}
METRIC_NAMES: dict[str, str] = {
    "spend": "Spend per customer",
    "conversion": "Conversion rate",
    "visit": "Visit rate",
}
RANKING_NAMES: dict[str, str] = {
    "s_learner": "S-learner",
    "t_learner": "T-learner",
    "x_learner": "X-learner",
    "response_model": "Response model (baseline)",
    "recency_rule": "Recency rule (baseline)",
    "random": "Random order (baseline)",
}
POLICY_NAMES: dict[str, str] = {
    "uplift": "Uplift policy (X-learner)",
    "response_model": "Response model",
    "recency_rule": "Recency rule",
    "random": "Random targeting (one draw)",
}
POLICY_PROSE: dict[str, str] = {
    "response_model": "the response model",
    "recency_rule": "the recency rule",
    "random": "random targeting",
}
RANKINGS: tuple[str, ...] = ("uplift", "response_model", "recency_rule")
RANKING_PROSE: dict[str, str] = {
    "uplift": "the uplift policy",
    "response_model": "the response model",
    "recency_rule": "the recency rule",
}
POLICY_OUTCOMES: tuple[str, ...] = ("visit", "conversion", "spend")
ADJUSTMENT_NAMES: dict[str, str] = {"holm": "Holm", "bh": "Benjamini-Hochberg", "none": "none"}
FAMILY_PROSE: dict[str, str] = {
    "primary": "each email against no email on spend",
    "secondary": "every other comparison of spend, conversion and visits",
    "sensitivity": "each email against no email on winsorised spend",
    "sensitivity_secondary": "Mens against Womens on winsorised spend",
}
OUTCOME_PLURALS: dict[str, str] = {"visit": "visits", "conversion": "conversions", "spend": "spend"}
SEQUENTIAL_NAMES: dict[str, str] = {
    "fixed": "One test at the end (no peeking)",
    "naive": "Unadjusted test at every look, stop at the first p < 0.05",
    "bayes": "Bayesian rule at every look, stop when P(better) leaves (0.05, 0.95)",
    "pocock": "Pocock boundary",
    "obf": "O'Brien-Fleming boundary",
    "msprt": "Mixture SPRT (always-valid p-value)",
}


# --- formatting --------------------------------------------------------------------------------


def usd(x: float, decimals: int = 2, signed: bool = False) -> str:
    r"""Format a dollar amount with an escaped dollar sign, e.g. ``+\$0.77`` or ``\$7,138``."""
    sign = "-" if x < 0 else ("+" if signed else "")
    return f"{sign}\\${abs(x):,.{decimals}f}"


def usd_range(low: float, high: float, decimals: int = 2) -> str:
    """Format an interval of dollar amounts as ``low to high``."""
    return f"{usd(low, decimals)} to {usd(high, decimals)}"


def pp(x: float, decimals: int = 1, signed: bool = False) -> str:
    """Format a difference of proportions in percentage points."""
    return f"{100 * x:{'+' if signed else ''}.{decimals}f} pp"


def pp_range(low: float, high: float, decimals: int = 1) -> str:
    """Format an interval of proportion differences in percentage points."""
    return f"{100 * low:.{decimals}f} to {100 * high:.{decimals}f} pp"


def pct(x: float, decimals: int = 1, signed: bool = False) -> str:
    """Format a proportion as a percentage."""
    return f"{100 * x:{'+' if signed else ''}.{decimals}f}%"


def num(x: float, decimals: int = 0) -> str:
    """Format a number with thousands separators."""
    return f"{x:,.{decimals}f}"


def num_range(low: float, high: float, decimals: int = 0) -> str:
    """Format an interval of plain numbers as ``low to high``."""
    return f"{num(low, decimals)} to {num(high, decimals)}"


def fixed(x: float, decimals: int = 3) -> str:
    """Format a number with fixed decimals, writing a rounded negative zero as zero."""
    text = f"{x:.{decimals}f}"
    return text.removeprefix("-") if float(text) == 0 else text


def p_value(p: float) -> str:
    """Format a p-value: three decimals, or ``<0.001``."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def join_words(items: Iterable[str]) -> str:
    """Join words as ``a``, ``a and b`` or ``a, b and c``."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def table(header: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def find(rows: list[Row], **where: Any) -> Row:
    """Return the single row whose fields equal ``where``.

    Raises
    ------
    LookupError
        If no row or more than one row matches.
    """
    matches = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    if len(matches) != 1:
        raise LookupError(f"{len(matches)} rows match {where}")
    return matches[0]


def _effect(row: Row, signed: bool = True) -> tuple[str, str]:
    """Estimate and interval of a readout row, in dollars for spend and points for rates."""
    if row["metric"].startswith("spend"):
        return usd(row["diff"], signed=signed), usd_range(row["ci_low"], row["ci_high"])
    decimals = 2 if row["metric"] == "conversion" else 1
    return pp(row["diff"], decimals, signed), pp_range(row["ci_low"], row["ci_high"], decimals)


def _versus(row: Row) -> str:
    return f"{EMAIL_NAMES[row['treatment']]} vs {EMAIL_NAMES[row['control']]}"


# --- blocks ------------------------------------------------------------------------------------


def _freq(results: Results, metric: str, comparison: str) -> Row:
    return find(results["frequentist"]["rows"], metric=metric, comparison=comparison)


def _winner(results: Results) -> tuple[str, str]:
    """Return the A/B winner and the other email."""
    winner = results["policy"]["ab_winner"]["email"]
    other = "womens" if winner == "mens" else "mens"
    return winner, other


def _head_to_head(results: Results, metric: str) -> tuple[Row, float]:
    """Return the mens-vs-womens row of a metric and the sign that favours the A/B winner."""
    winner, _ = _winner(results)
    return _freq(results, metric, "mens_vs_womens"), 1.0 if winner == "mens" else -1.0


def _policy(results: Results, policy: str, outcome: str) -> Row:
    return results["policy"]["headline"][outcome][policy]


def intro(results: Results) -> str:
    """One sentence on the experiment's design."""
    counts = results["validity"]["srm"]["counts"]
    days = results["sequential"]["days"]
    return (
        f"{num(sum(counts.values()))} customers were randomised to a Mens email, a Womens email "
        f"or no email, and their visits, purchases and spend were tracked for {days} days "
        "(Hillstrom's MineThatData challenge, 2008)."
    )


def _paired(results: Results, first: str, second: str, outcome: str) -> Row:
    """Paired bootstrap difference ``first - second`` in incremental total, with its interval.

    The comparison table stores each pair once per reference policy; the row is read from
    whichever side holds it and its sign is reversed if needed.
    """
    rows = results["policy"]["comparison_rows"]
    for reference, other, sign in ((first, second, 1.0), (second, first, -1.0)):
        matches = [
            r
            for r in rows
            if r["reference"] == reference and r["policy"] == other and r["outcome"] == outcome
        ]
        if matches:
            r = matches[0]
            low, high = sorted((sign * r["ci_low"], sign * r["ci_high"]))
            return {
                "diff": sign * r["diff_incremental_total"],
                "ci_low": low,
                "ci_high": high,
                "meaningful": r["meaningful"],
            }
    raise LookupError(f"no paired comparison of {first} and {second} on {outcome}")


def _separated_rankings(results: Results) -> list[tuple[str, str, str]]:
    """Pairs of rankings, with the outcome, whose paired difference excludes zero."""
    pairs = [(a, b) for i, a in enumerate(RANKINGS) for b in RANKINGS[i + 1 :]]
    return [
        (a, b, outcome)
        for a, b in pairs
        for outcome in POLICY_OUTCOMES
        if _paired(results, a, b, outcome)["meaningful"]
    ]


def _signed_range(d: Row) -> str:
    return f"{d['diff']:+,.0f} ({num_range(d['ci_low'], d['ci_high'])})"


def _versus_random(results: Results, outcome: str) -> tuple[dict[str, Row], str]:
    """Each ranking's paired difference against the random draw, and which exclude zero."""
    diffs = {name: _paired(results, name, "random", outcome) for name in RANKINGS}
    apart = [RANKING_PROSE[name] for name, d in diffs.items() if d["meaningful"]]
    if not apart:
        verdict = "none excludes zero"
    elif len(apart) == len(diffs):
        verdict = f"all {len(apart)} exclude zero"
    else:
        verdict = f"only {join_words(apart)} exclude{'s' if len(apart) == 1 else ''} zero"
    return diffs, verdict


def _expected_random(results: Results, outcome: str) -> tuple[float, float, float]:
    """Return random targeting's expected increment: budget times the winner's average effect."""
    winner, _ = _winner(results)
    budget = results["policy"]["budget"]
    ate = _freq(results, outcome, f"{winner}_vs_control")
    return budget * ate["diff"], budget * ate["ci_low"], budget * ate["ci_high"]


def _targeting_reason(results: Results) -> str:
    """Write the recommendation under a budget and the reasoning behind it."""
    winner, _ = _winner(results)
    name, budget = EMAIL_NAMES[winner], num(results["policy"]["budget"])
    if _separated_rankings(results):
        return (
            f"If only {budget} emails can be sent, the rankings are not all tied: see "
            "[Who to email](#5-who-to-email) for the one that is ahead."
        )
    return (
        f"If only {budget} emails can be sent, send the {name} email to the {budget} most "
        "recent purchasers: no ranking is separated from the others, so use the simplest one, "
        "the recency rule, which needs no model."
    )


def _adjustment_note(results: Results, rows: list[Row]) -> str:
    """Say what the adjusted p-value of each headline row is adjusted for."""
    families = results["frequentist"]["families"]
    by_family: dict[str, list[int]] = {}
    for i, row in enumerate(rows, 1):
        by_family.setdefault(row["family"], []).append(i)
    parts = []
    for family, numbers in by_family.items():
        spec = families[family]
        parts.append(
            f"{'rows' if len(numbers) > 1 else 'row'} {join_words(str(n) for n in numbers)}, "
            f"{ADJUSTMENT_NAMES[spec['adjustment']]} across the {spec['n_tests']} tests of the "
            f"{family} family ({FAMILY_PROSE[family]})"
        )
    return f"Adjusted p: {'; '.join(parts)}. Intervals are unadjusted."


def summary(results: Results) -> str:
    """Write the decision and the headline numbers with their intervals."""
    winner, other = _winner(results)
    name, other_name = EMAIL_NAMES[winner], EMAIL_NAMES[other]
    policy = results["policy"]
    budget = policy["budget"]
    spend = {c: _freq(results, "spend", c) for c in ("mens_vs_control", "womens_vs_control")}
    head, sign = _head_to_head(results, "spend")
    visit = _freq(results, "visit", f"{winner}_vs_control")
    both = all(r["significant"] for r in spend.values())
    winner_ahead = head["significant"] and sign * head["diff"] > 0
    lead = (
        f"Both emails beat sending nothing on spend, and the {name} email "
        + (
            f"beats the {other_name} email."
            if winner_ahead
            else f"has the larger estimate but is not separated from the {other_name} email."
        )
        if both
        else f"The {name} email is the one supported against sending nothing on spend."
    )
    headline_rows = [*spend.values(), head, visit]
    rows = [
        [f"{_versus(r)}: spend per customer", *_effect(r), p_value(r["p_adj"])]
        for r in headline_rows[:3]
    ]
    rows.append(
        [
            f"{_versus(visit)}: visit rate ({pct(visit['mean_control'])} → "
            f"{pct(visit['mean_treatment'])})",
            *_effect(visit),
            p_value(visit["p_adj"]),
        ]
    )
    everyone = _policy(results, f"all_{winner}", "spend")
    thousands = [round(everyone[k], -3) for k in ("incremental_total", "ci_low", "ci_high")]

    values = {r: _policy(results, r, "visit") for r in RANKINGS}
    ordered = sorted(RANKINGS, key=lambda r: -values[r]["incremental_total"])
    listed = ", ".join(
        f"{RANKING_PROSE[r]} {num(values[r]['incremental_total'])} "
        f"({num_range(values[r]['ci_low'], values[r]['ci_high'])})"
        for r in ordered
    )
    best = ordered[0]
    if best == "recency_rule":
        lead_note = "The recency rule has the highest point estimate."
    else:
        d = _paired(results, best, "recency_rule", "visit")
        lead_note = (
            f"{RANKING_PROSE[best].capitalize()} has the highest point estimate, but its lead "
            f"over the recency rule, {num(d['diff'])} visits "
            f"({num_range(d['ci_low'], d['ci_high'])}), "
            f"{'excludes' if d['meaningful'] else 'includes'} zero."
        )
    draw = _policy(results, "random", "visit")
    expected, _, _ = _expected_random(results, "visit")
    diffs, verdict = _versus_random(results, "visit")
    _, verdict_conversion = _versus_random(results, "conversion")
    against = ", ".join(f"{RANKING_PROSE[r]} {_signed_range(diffs[r])}" for r in ordered)

    suppressed = find(policy["suppressed_group"], outcome="visit")
    no_group = suppressed["ci_low"] > 0
    return "\n".join(
        [
            f"**Decision: send the {name} email, to everyone who can be emailed.** {lead} "
            f"{_targeting_reason(results)}",
            "",
            table(
                ["Headline (two-week window)", "Estimate", "95% interval", "Adjusted p"],
                rows,
            ),
            "",
            _adjustment_note(results, headline_rows),
            "",
            f"- **Value.** Sent to all {num(policy['n_customers'])} customers, the {name} email "
            f"brings about {usd(thousands[0], 0)} of extra revenue in two weeks "
            f"({usd_range(thousands[1], thousands[2], 0)}). Profit cannot be computed: the "
            "data hold no cost or margin.",
            f"- **Targeting {num(budget)} customers.** Extra visits: {listed}. {lead_note} "
            f"Random targeting gives {num(draw['incremental_total'])} in the pre-registered "
            f"draw and {num(expected)} in expectation (budget times the average effect). "
            f"Paired differences against that draw: {against}; {verdict} on visits, and "
            f"{verdict_conversion} on conversions.",
            "- **Whom not to email.** "
            + (
                f"Nobody identified. The {num(suppressed['n_suppressed'])} customers the model "
                f"ranks lowest still visit more when emailed: "
                f"{pp(suppressed['uplift_per_customer'], signed=True)} "
                f"({pp_range(suppressed['ci_low'], suppressed['ci_high'])})."
                if no_group
                else f"The {num(suppressed['n_suppressed'])} customers the model ranks lowest "
                f"show no clear gain from an email: "
                f"{pp(suppressed['uplift_per_customer'], signed=True)} "
                f"({pp_range(suppressed['ci_low'], suppressed['ci_high'])})."
            ),
        ]
    )


def questions(results: Results) -> str:
    """Short answers to the five business questions."""
    winner, other = _winner(results)
    name, other_name = EMAIL_NAMES[winner], EMAIL_NAMES[other]
    policy = results["policy"]
    budget = num(policy["budget"])

    heads = []
    for metric in ("visit", "conversion", "spend"):
        row, sign = _head_to_head(results, metric)
        estimate, interval = _effect({**row, "diff": sign * row["diff"]})
        low, high = sorted((sign * row["ci_low"], sign * row["ci_high"]))
        interval = _effect({**row, "ci_low": low, "ci_high": high})[1]
        heads.append(
            (
                row["significant"] and sign * row["diff"] > 0,
                f"{METRIC_NAMES[metric].lower()} {estimate} ({interval})",
            )
        )
    all_ahead = all(flag for flag, _ in heads)
    q1 = (
        f"The {name} email, on {'all three metrics' if all_ahead else 'the metrics listed'}. "
        f"Against the {other_name} email: {'; '.join(text for _, text in heads)}"
        + (", each significant after adjustment." if all_ahead else ".")
    )

    parts = []
    for email in ("mens", "womens"):
        r = _freq(results, "spend", f"{email}_vs_control")
        estimate, interval = _effect(r)
        parts.append(
            f"{EMAIL_NAMES[email]} {estimate} ({interval}), a relative lift of "
            f"{pct(r['rel_lift'], 0, signed=True)}"
        )
    control_mean = _freq(results, "spend", "mens_vs_control")["mean_control"]
    q2 = f"{'; '.join(parts)}, on a no-email baseline of {usd(control_mean)} per customer."

    uplift = _policy(results, "uplift", "visit")
    best = max(RANKINGS, key=lambda r: _policy(results, r, "visit")["incremental_total"])
    q3 = (
        f"The uplift policy (each customer's better email, top {budget} by predicted uplift) "
        f"brings {num(uplift['incremental_total'])} extra visits "
        f"({num_range(uplift['ci_low'], uplift['ci_high'])}). "
        + (
            f"No ranking is separated from the others ({RANKING_PROSE[best]} has the highest "
            f"point estimate), so use the simplest one: the {name} email to the {budget} most "
            "recent purchasers."
            if not _separated_rankings(results)
            else "The rankings are not all tied; see [Who to email](#5-who-to-email)."
        )
    )

    sup = find(policy["suppressed_group"], outcome="visit")
    q4 = (
        (
            f"No such group was found. The {num(sup['n_suppressed'])} customers with the lowest "
            "predicted uplift still respond to their better email: "
        )
        if sup["ci_low"] > 0
        else (
            f"The {num(sup['n_suppressed'])} customers with the lowest predicted uplift show no "
            "clear response to their better email: "
        )
    ) + (
        f"{pp(sup['uplift_per_customer'], signed=True)} visits "
        f"({pp_range(sup['ci_low'], sup['ci_high'])})."
    )

    segments = results["segments"]
    inter = segments["interaction_rows"]
    n_mens = sum(r["significant_fdr"] for r in inter if r["comparison"] == "mens_vs_control")
    n_womens = sum(r["significant_fdr"] for r in inter if r["comparison"] == "womens_vs_control")
    n_each = sum(r["comparison"] == "mens_vs_control" for r in inter)
    cq = find(segments["challenge_question"], email="womens", outcome="visit")
    q5 = (
        f"For the Womens email, yes: it lifts visits by {pp(cq['past_buyers_diff'], signed=True)} "
        f"({pp_range(cq['past_buyers_ci_low'], cq['past_buyers_ci_high'])}) among past buyers "
        f"of women's merchandise and by {pp(cq['non_buyers_diff'], signed=True)} "
        f"({pp_range(cq['non_buyers_ci_low'], cq['non_buyers_ci_high'])}) among the rest "
        f"(interaction adjusted p {p_value(cq['interaction_p_adj'])}). {n_womens} of its "
        f"{n_each} interaction tests are significant after Benjamini-Hochberg, against "
        f"{n_mens} of {n_each} for the Mens email, whose effect is close to uniform. "
        "Exploratory, not a decision."
    )
    answers = [
        ("Which email performed better, and on which metric?", q1),
        ("How much incremental spend per customer did each email drive?", q2),
        (f"If only {budget} customers can be emailed, who should get which email?", q3),
        (f"Which {budget} customers should not be emailed?", q4),
        ("Do the effects differ across customer segments?", q5),
    ]
    return table(
        ["#", "Question", "Answer"],
        [[f"Q{i}", question, answer] for i, (question, answer) in enumerate(answers, 1)],
    )


def validity(results: Results) -> str:
    """Sample ratio mismatch and covariate balance."""
    srm, balance = results["validity"]["srm"], results["validity"]["balance"]
    counts = srm["counts"]
    chi2 = balance["categorical_chi2"]
    level = f"`{balance['max_abs_smd_variable']}` = `{balance['max_abs_smd_level']}`"
    return "\n".join(
        [
            f"- **Sample ratio mismatch: {'none' if srm['passed'] else 'detected'}.** Arm sizes "
            f"are {num(counts['mens'])} (Mens), {num(counts['womens'])} (Womens) and "
            f"{num(counts['control'])} (no email) against an expected "
            f"{num(srm['expected']['control'])} each: chi-square {srm['statistic']:.2f} on "
            f"{srm['dof']} degrees of freedom, p = {srm['p_value']:.2f} "
            f"(alarm threshold {srm['alpha']}).",
            f"- **Balance: {'no variable flagged' if balance['n_flagged'] == 0 else 'flags'}.** "
            f"The largest absolute standardised mean difference between any two arms, over "
            f"every pre-treatment variable and level, is {balance['max_abs_smd']:.3f} ({level}), "
            f"against a threshold of {balance['threshold']}; {balance['n_flagged']} are flagged. "
            f"Chi-square tests of the categorical variables give p = "
            + join_words(f"{c['p_value']:.2f} (`{c['variable']}`)" for c in chi2)
            + ".",
        ]
    )


def readout(results: Results) -> str:
    """Frequentist table, the sensitivity analysis and the Bayesian readout."""
    freq = results["frequentist"]
    rows = []
    for metric in ("spend", "conversion", "visit"):
        for comparison in ("mens_vs_control", "womens_vs_control", "mens_vs_womens"):
            r = _freq(results, metric, comparison)
            rows.append(
                [
                    METRIC_NAMES[metric],
                    _versus(r),
                    *_effect(r),
                    f"{pct(r['rel_lift'], 0, signed=True)} "
                    f"({pct(r['rel_ci_low'], 0)} to {pct(r['rel_ci_high'], 0)})",
                    p_value(r["p_adj"]),
                ]
            )
    win = [_freq(results, "spend_winsorised", f"{e}_vs_control") for e in ("mens", "womens")]
    boot = [_freq(results, "spend", f"{e}_vs_control") for e in ("mens", "womens")]
    bayes = results["bayes_spend"]
    rule = bayes["decision_rule"]
    posterior = [
        find(bayes["rows"], metric="spend", model="bayesian_bootstrap", comparison=c)
        for c in ("mens_vs_control", "womens_vs_control", "mens_vs_womens")
    ]
    recommendation = bayes["recommendation"]
    families = freq["families"]
    return "\n".join(
        [
            table(
                [
                    "Metric",
                    "Comparison",
                    "Difference",
                    "95% interval",
                    "Relative lift",
                    "Adjusted p",
                ],
                rows,
            ),
            "",
            f"Spend against control is the primary family ({families['primary']['n_tests']} "
            f"tests, Holm); the other {families['secondary']['n_tests']} comparisons are "
            "secondary (Benjamini-Hochberg). Spend uses Welch's t-test, rates a two-proportion "
            "z-test with Newcombe intervals.",
            "",
            f"- **Sensitivity.** With converters' spend winsorised at "
            f"{usd(freq['spend_winsorisation_cap'])}, the differences against control are "
            + join_words(
                f"{_effect(r)[0]} ({_effect(r)[1]}) for {EMAIL_NAMES[r['treatment']]}" for r in win
            )
            + ". A percentile bootstrap of raw spend "
            f"({num(freq['bootstrap']['n_resamples'])} resamples) gives "
            + join_words(
                f"{usd_range(r['boot_ci_low'], r['boot_ci_high'])} for "
                f"{EMAIL_NAMES[r['treatment']]}"
                for r in boot
            )
            + ", so the normal approximation is not doing the work.",
            f"- **Bayesian readout** (Bayesian bootstrap on spend, "
            f"{num(bayes['n_draws'])} draws). "
            + " ".join(
                f"P({EMAIL_NAMES[r['treatment']]} better than {EMAIL_NAMES[r['control']]}) = "
                f"{r['p_better']:.3f}, credible interval "
                f"{usd_range(r['ci_low'], r['ci_high'])}."
                for r in posterior
            )
            + f" The pre-registered rule (ship if P(better) is at least "
            f"{rule['p_better_threshold']} and the expected loss is below "
            f"{pct(rule['loss_fraction_of_control_mean'], 0)} of the control mean) recommends "
            f"the {EMAIL_NAMES[recommendation['recommended_email']]} email. The frequentist "
            "and Bayesian readouts agree, as expected with arms this large and a flat prior.",
        ]
    )


def cuped(results: Results) -> str:
    """Why covariate adjustment barely reduces variance here, with the numbers."""
    by_metric = results["cuped"]["variance_reduction_by_metric"]
    spend, visit = by_metric["spend"], by_metric["visit"]
    row = find(results["cuped"]["rows"], metric="spend", comparison="mens_vs_control")
    mde = {
        flag: find(results["power"]["mde_rows"], metric="spend", alpha_label="holm", cuped=flag)
        for flag in (False, True)
    }
    zero = results["eda"]["pooled_rates"]["share_zero_spend"]
    n_covariates = len(results["cuped"]["lin_covariates"])
    return "\n".join(
        [
            "**CUPED and the regression adjustment barely reduce variance here.** CUPED removes "
            "a share of the variance equal to the squared correlation between the outcome and "
            "the pre-experiment covariate, and that correlation is tiny:",
            "",
            table(
                [
                    "Metric",
                    "Correlation with past-year spend",
                    "Variance removed by CUPED",
                    f"Variance removed by Lin regression ({n_covariates} covariates)",
                ],
                [
                    [
                        METRIC_NAMES[m],
                        f"{by_metric[m]['rho_min']:.3f} to {by_metric[m]['rho_max']:.3f}",
                        f"{pct(by_metric[m]['cuped_variance_reduction_min'], 2)} to "
                        f"{pct(by_metric[m]['cuped_variance_reduction_max'], 2)}",
                        f"{pct(by_metric[m]['lin_variance_reduction_min'], 2)} to "
                        f"{pct(by_metric[m]['lin_variance_reduction_max'], 2)}",
                    ]
                    for m in ("spend", "conversion", "visit")
                ],
            ),
            "",
            f"- For spend, the standard error of the Mens effect goes from "
            f"{usd(row['raw_se'], 5)} to {usd(row['cuped_se'], 5)} with CUPED and to "
            f"{usd(row['lin_se'], 5)} with the regression, and the minimum detectable effect "
            f"stays at {usd(mde[False]['mde_absolute'], 4)} ({usd(mde[True]['mde_absolute'], 4)} "
            "with CUPED). A negative value in the table means the adjustment added a little "
            "noise.",
            f"- **Why.** {pct(zero)} of customers spend nothing in the two weeks, so spend is "
            "almost entirely a rare, noisy purchase event that last year's spend cannot "
            f"predict (correlation at most {spend['rho_max']:.3f}). Visits are more "
            f"predictable, and there the full regression removes up to "
            f"{pct(visit['lin_variance_reduction_max'])}. CUPED pays off when the covariate is "
            "the same metric measured before the experiment on returning users; a one-off "
            "campaign with a two-week window is not that setting.",
        ]
    )


def peeking(results: Results) -> str:
    """Tabulate the false positive rate and power of each stopping rule under daily looks."""
    seq = results["sequential"]
    looks = seq["days"]

    def row(scenario: str, method: str) -> Row:
        matches = [
            r
            for r in seq["rows"]
            if r["scenario"] == scenario
            and r["metric"] == "visit"
            and r["n_looks"] == looks
            and r["method"] == method
            and (method != "msprt" or math.isclose(r["tau"], seq["tau"]["visit"]))
        ]
        if len(matches) != 1:
            raise LookupError(f"{len(matches)} sequential rows for {scenario}/{method}")
        return matches[0]

    rows = []
    for method, label in SEQUENTIAL_NAMES.items():
        aa, ab = row("aa", method), row("ab_mde", method)
        rows.append(
            [
                label,
                f"{pct(aa['reject_rate'])} (±{pct(aa['reject_mcse'])})",
                f"{pct(ab['reject_rate'])} (±{pct(ab['reject_mcse'])})",
                f"{ab['mean_stop_day']:.1f}",
            ]
        )
    naive, fixed = row("aa", "naive"), row("aa", "fixed")
    mde = seq["effects"]["ab_mde"]
    k5, published = seq["constants"]["5"], seq["published_k5_jennison_turnbull"]
    return "\n".join(
        [
            f"Simulated experiments shaped like this one ({num(seq['n_per_arm'])} customers "
            f"per arm arriving evenly over {looks} days, visit rate as in the control arm, "
            f"{num(seq['n_runs'])} runs). Looking every day and stopping at the first "
            f"p < {seq['alpha']} declares a winner in {pct(naive['reject_rate'])} of "
            f"experiments in which the email does nothing, against {pct(fixed['reject_rate'])} "
            "for a single test at the end.",
            "",
            table(
                [
                    f"Rule, {looks} daily looks",
                    "False positive rate, no true effect (± Monte Carlo SE)",
                    f"Power at a {pct(mde['relative'])} relative lift",
                    "Mean stopping day at that lift",
                ],
                rows,
            ),
            "",
            f"The group-sequential boundaries are calibrated by simulation; for five looks "
            f"they come out at {k5['pocock']:.3f} (Pocock) and {k5['obf']:.3f} "
            f"(O'Brien-Fleming), against the published {published['pocock']} and "
            f"{published['obf']}.",
        ]
    )


def multiple_testing(results: Results) -> str:
    """Summarise the false-discovery simulation and what BH changes on the real data."""
    segments = results["segments"]
    sim = segments["fdr_simulation"]

    def row(scenario: str, method: str) -> Row:
        return find(sim["rows"], scenario=scenario, method=method)

    null_raw, null_bh = row("global_null", "raw"), row("global_null", "bh")
    true_scenario = next(r["scenario"] for r in sim["rows"] if r["n_true"] > 0)
    true_raw, true_bh = row(true_scenario, "raw"), row(true_scenario, "bh")
    families = segments["families"]
    real = [
        f"{f['n_significant_raw']} of {f['n_tests']} unadjusted and {f['n_significant_fdr']} "
        f"after Benjamini-Hochberg for {OUTCOME_PLURALS[name.removeprefix('segments_')]}"
        for name, f in families.items()
    ]
    return "\n".join(
        [
            f"Simulation: {sim['n_segments']} segments of {num(sim['n_per_arm'])} customers "
            f"per arm, {num(sim['n_runs'])} runs.",
            "",
            table(
                ["Scenario", "Unadjusted tests at 0.05", "Benjamini-Hochberg at 0.05"],
                [
                    [
                        "No segment has an effect: chance of at least one false discovery",
                        f"{pct(null_raw['any_false_discovery'])} (theory "
                        f"{pct(null_raw['any_false_discovery_reference'])})",
                        pct(null_bh["any_false_discovery"]),
                    ],
                    [
                        f"{sim['n_true']} of {sim['n_segments']} segments have an effect: "
                        "share of discoveries that are false",
                        pct(true_raw["fdr"]),
                        pct(true_bh["fdr"]),
                    ],
                    [
                        "Same scenario: power to find a true effect",
                        pct(true_raw["power"]),
                        pct(true_bh["power"]),
                    ],
                ],
            ),
            "",
            "**On the real data the correction changes little, because most of the effects "
            f"are real.** Across {segments['n_levels']} segment levels and two emails, the "
            f"significant tests are {'; '.join(real)}. Both emails lift visits in nearly every "
            "segment, so there are few true nulls for the correction to protect against. The "
            "correction matters for the claim of a *difference between* segments, which is "
            "what the interaction tests in Q5 address.",
        ]
    )


def uplift_models(results: Results) -> str:
    """Normalised Qini coefficients of every ranking and the seed robustness."""
    models = results["uplift_models"]
    rows = [r for r in models["rows"] if r["metric"] == "qini_normalised"]

    def cell(arm: str, ranking: str) -> str:
        r = find(rows, arm=arm, outcome="visit", ranking=ranking)
        return f"{fixed(r['estimate'])} ({fixed(r['ci_low'])} to {fixed(r['ci_high'])})"

    lines = [
        table(
            ["Ranking", "Mens email vs no email", "Womens email vs no email"],
            [[RANKING_NAMES[k], cell("mens", k), cell("womens", k)] for k in models["rankings"]],
        ),
        "",
        "Normalised Qini coefficient on visits, out of fold, with 95% bootstrap intervals "
        f"({num(models['bootstrap']['n_resamples'])} resamples): 0 is a random ranking, 1 the "
        "best ranking the outcomes allow.",
        "",
    ]
    for arm in ("womens", "mens"):
        s = models["summary"][f"{arm}_visit"]
        best = RANKING_NAMES[s["best_ranking"]].removesuffix(" (baseline)")
        ahead = join_words(
            RANKING_NAMES[k].removesuffix(" (baseline)").lower()
            if k in models["baselines"]
            else RANKING_NAMES[k]
            for k in s["meaningfully_better_than"]
        )
        tied = join_words(
            RANKING_NAMES[k].removesuffix(" (baseline)").lower()
            if k in models["baselines"]
            else RANKING_NAMES[k]
            for k in s["not_separated_from"]
        )
        is_model = s["best_ranking"] in models["uplift_learners"]
        text = f"- **{EMAIL_NAMES[arm]} email.** The {best if is_model else best.lower()} ranks "
        text += "best"
        text += f", ahead of the {ahead} by a paired bootstrap difference" if ahead else ""
        text += f", and is not separated from the {tied}" if tied else ""
        text += "."
        if not is_model:
            text += (
                " No uplift learner beats the baselines: the email's effect on visits is "
                "close to the same for everyone, so there is little to rank."
            )
        lines.append(text)
    robustness = results.get("seed_robustness")
    if robustness:
        x = find(robustness["spread"], arm="womens", ranking="x_learner", metric="qini_normalised")
        beats = robustness["x_beats_response_model_womens"]
        lines.append(
            f"- **Stability.** Repeating the cross-fitting under {robustness['n_seeds']} fold "
            f"seeds, the X-learner's coefficient on the Womens pair has median {x['median']:.3f} "
            f"(range {x['min']:.3f} to {x['max']:.3f}) and is above the response model in "
            f"{beats['count']} of {beats['out_of']} seeds."
        )
    lines.append(
        "- **The choice of the X-learner is post hoc (deviation D8).** It was picked for the "
        "policy after these results were seen, from three learners evaluated on the same "
        "data, so its coefficient is the largest of several noisy estimates and is optimistic. "
        'Read "best" as "chosen", not as "shown to be better".'
    )
    return "\n".join(lines)


def policy(results: Results, reports_dir: Path) -> str:
    """Tabulate the fixed-budget policies, the budget curve and the do-not-email check."""
    pol = results["policy"]
    budget = pol["budget"]
    winner, _ = _winner(results)
    rows_by_policy = {
        name: find(pol["rows"], policy=name, outcome="visit") for name in POLICY_NAMES
    }
    table_rows = []
    for name, label in POLICY_NAMES.items():
        visit, spend = _policy(results, name, "visit"), _policy(results, name, "spend")
        counts = rows_by_policy[name]
        if name == "uplift":
            versus = "reference"
        else:
            d = pol["uplift_versus"]["visit"][name]
            versus = f"{d['diff_incremental_total']:+,.0f} ({num_range(d['ci_low'], d['ci_high'])})"
        table_rows.append(
            [
                label,
                f"{num(counts['n_mens'])} / {num(counts['n_womens'])}",
                f"{num(visit['incremental_total'])} "
                f"({num_range(visit['ci_low'], visit['ci_high'])})",
                f"{usd(spend['incremental_total'], 0)} "
                f"({usd_range(spend['ci_low'], spend['ci_high'], 0)})",
                versus,
            ]
        )

    def separated(name: str) -> list[str]:
        return [
            outcome
            for outcome in ("visit", "conversion", "spend")
            if pol["uplift_versus"][outcome][name]["meaningful"]
        ]

    verdicts = []
    for name in ("response_model", "recency_rule"):
        outcomes = separated(name)
        verdicts.append(
            f"**The uplift policy is not separated from {POLICY_PROSE[name]}** on visits, "
            "conversions or spend: every paired bootstrap interval includes zero."
            if not outcomes
            else f"The uplift policy is separated from {POLICY_PROSE[name]} on "
            f"{join_words(OUTCOME_PLURALS[o] for o in outcomes)}."
        )
    if not _separated_rankings(results):
        best = max(RANKINGS, key=lambda r: _policy(results, r, "visit")["incremental_total"])
        verdicts.append(
            "The recency rule and the response model are not separated from each other "
            f"either. {RANKING_PROSE[best].capitalize()} has the highest point estimate on "
            "visits, but with no ranking separated from another the recommendation is the "
            "simplest one, the recency rule. The analysis plan pre-registered no rule for "
            "this choice."
        )
    visit_diffs, _ = _versus_random(results, "visit")
    random_verdicts = join_words(
        f"{_versus_random(results, outcome)[1]} on {OUTCOME_PLURALS[outcome]}"
        for outcome in POLICY_OUTCOMES
    )
    random_line = (
        "**Against random targeting** (the pre-registered draw), the paired differences in "
        "visits are "
        + join_words(f"{_signed_range(visit_diffs[r])} for {RANKING_PROSE[r]}" for r in RANKINGS)
        + f": {random_verdicts}."
    )

    curve = pd.read_csv(reports_dir / pol["budget_curve_table"])
    grid = curve[(curve["n_emailed"] > 0) & (curve["n_emailed"] < pol["n_customers"])]
    counts = {}
    for column in ("incremental_visits", "incremental_spend"):
        wide = grid.pivot_table(index="n_emailed", columns="policy", values=column)
        counts[column] = int((wide["recency_rule"] >= wide["uplift"]).sum()), len(wide)
    step = int(curve["n_emailed"].drop_duplicates().sort_values().iloc[1])
    at_budget = {
        name: _policy(results, name, "visit")["incremental_total"]
        for name in ("recency_rule", "uplift")
    }
    order = "above" if at_budget["recency_rule"] >= at_budget["uplift"] else "below"
    sup = {o: find(pol["suppressed_group"], outcome=o) for o in ("visit", "spend")}
    no_group = sup["visit"]["ci_low"] > 0
    return "\n".join(
        [
            table(
                [
                    f"Policy, {num(budget)} emails",
                    "Mens / Womens emails",
                    "Incremental visits (95% interval)",
                    "Incremental spend (95% interval)",
                    "Uplift policy minus this policy, visits",
                ],
                table_rows,
            ),
            "",
            "Values are increments over emailing nobody, estimated by inverse propensity "
            f"weighting with the design probability, with a paired bootstrap "
            f"({num(pol['bootstrap']['n_resamples'])} resamples). The recency rule and random "
            f"targeting send the A/B winner ({EMAIL_NAMES[winner]}).",
            "",
            f"- {' '.join(verdicts)}",
            f"- {random_line}",
            f"- **Budget curve.** Extending each ranking to every budget in steps of "
            f"{num(step)}, the recency rule is at or above the uplift policy at "
            f"{counts['incremental_visits'][0]} of {counts['incremental_visits'][1]} budgets on "
            f"visits and at {counts['incremental_spend'][0]} of "
            f"{counts['incremental_spend'][1]} on spend. At the budget of {num(budget)} it is "
            f"{order} the uplift policy on visits ({num(at_budget['recency_rule'])} against "
            f"{num(at_budget['uplift'])}). These are point estimates; the curves cross "
            "repeatedly and no ranking leads throughout.",
            f"- **Do-not-email list.** The {num(sup['visit']['n_suppressed'])} customers with "
            "the lowest predicted uplift, given their better email, gain "
            f"{pp(sup['visit']['uplift_per_customer'], signed=True)} in visit rate "
            f"({pp_range(sup['visit']['ci_low'], sup['visit']['ci_high'])}) and "
            f"{usd(sup['spend']['uplift_per_customer'], signed=True)} in spend "
            f"({usd_range(sup['spend']['ci_low'], sup['spend']['ci_high'])}). "
            + (
                "The model does not find customers who are better left alone."
                if no_group
                else "The interval for visits includes zero."
            ),
        ]
    )


def random_targeting(results: Results) -> str:
    """Put the pre-registered random draw next to its expectation, budget times ATE."""
    pol = results["policy"]
    budget = pol["budget"]
    winner, _ = _winner(results)
    rows = []
    formats: dict[str, Callable[[float], str]] = {
        "visit": lambda x: num(x),
        "spend": lambda x: usd(x, 0),
    }
    for outcome, label in (("visit", "Incremental visits"), ("spend", "Incremental spend")):
        fmt = formats[outcome]
        draw = _policy(results, "random", outcome)
        ate = _freq(results, outcome, f"{winner}_vs_control")
        rows.append(
            [
                label,
                f"{fmt(draw['incremental_total'])} "
                f"({fmt(draw['ci_low'])} to {fmt(draw['ci_high'])})",
                f"{fmt(budget * ate['diff'])} "
                f"({fmt(budget * ate['ci_low'])} to {fmt(budget * ate['ci_high'])})",
            ]
        )
    return "\n".join(
        [
            table(
                [
                    f"Random targeting, {num(budget)} {EMAIL_NAMES[winner]} emails",
                    "Pre-registered draw (bootstrap interval)",
                    "Exact expectation, budget times average effect (interval from the effect's)",
                ],
                rows,
            ),
            "",
            "The draw is one random set of customers (analysis plan §11, deviation D9) and is "
            "what enters the paired comparisons above. The expectation needs no draw. The two "
            "differ by sampling noise, and the draw's interval is wider because it rests on "
            "the customers who happened to be drawn.",
        ]
    )


def verification(results: Results) -> str:
    """List what the simulation-based tests check and what each rules out."""
    seq = results["sequential"]
    k5, published = seq["constants"]["5"], seq["published_k5_jennison_turnbull"]
    sim = results["segments"]["fdr_simulation"]
    null_raw = find(sim["rows"], scenario="global_null", method="raw")
    checks = [
        (
            "Interval coverage",
            "The 95% intervals for rates, spend, CUPED, the Lin regression and the Qini "
            "bootstrap are computed on repeated simulated experiments with a known effect and "
            "must cover it about 95% of the time, within Monte Carlo error.",
            "Intervals that are too narrow on zero-inflated, heavy-tailed spend.",
        ),
        (
            "Calibration",
            "Credible intervals cover parameters drawn from the prior at their nominal rate; "
            "the two-part spend model recovers known parameters; out-of-fold uplift scores are "
            "checked by decile against observed effects.",
            "A posterior or a model whose stated uncertainty or level is wrong.",
        ),
        (
            "Published boundary constants",
            f"The simulated five-look boundaries ({k5['pocock']:.3f} and {k5['obf']:.3f}) match "
            f"the published {published['pocock']} and {published['obf']}; with one look both "
            "reduce to the normal quantile; every valid rule holds its false positive rate.",
            "A sequential simulation that is wrong in a way that still looks plausible.",
        ),
        (
            "Oracle rankings",
            "On synthetic customers with a known individual effect, the true-effect ranking "
            "scores a normalised Qini of 1, random rankings average 0, and each learner ranks "
            "customers in line with the planted effect.",
            "An uplift metric or learner that rewards likely responders, not persuadable ones.",
        ),
        (
            "Known answers",
            "Every estimator is compared with scipy or statsmodels, a closed form or planted "
            f"data: for example {sim['n_segments']} null segments give at least one false "
            f"discovery in {pct(null_raw['any_false_discovery'])} of runs against "
            f"{pct(null_raw['any_false_discovery_reference'])} in theory, and inverse "
            "propensity weighting recovers the known value of a fixed policy.",
            "Formula and indexing errors.",
        ),
        (
            "Leakage guard",
            "The feature builder refuses any outcome or arm column, and every score is out of "
            "fold.",
            "Results inflated by information from the outcome.",
        ),
    ]
    return table(
        ["Check", "What is tested", "What it rules out"], [list(check) for check in checks]
    )


def build_blocks(results: Results, reports_dir: Path) -> dict[str, str]:
    """Return the markdown of every generated README block.

    Parameters
    ----------
    results : dict
        Parsed ``results.json``.
    reports_dir : Path
        Directory holding ``results.json``; table paths in it are relative to this.

    Returns
    -------
    dict of str to str
        Block name (the ``name`` of ``<!-- RESULTS:name -->``) to its markdown.
    """
    return {
        "intro": intro(results),
        "summary": summary(results),
        "questions": questions(results),
        "validity": validity(results),
        "readout": readout(results),
        "cuped": cuped(results),
        "peeking": peeking(results),
        "multiple_testing": multiple_testing(results),
        "uplift_models": uplift_models(results),
        "policy": policy(results, reports_dir),
        "random_targeting": random_targeting(results),
        "verification": verification(results),
    }
