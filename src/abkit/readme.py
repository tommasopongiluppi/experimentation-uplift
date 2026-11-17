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
    "random": "Random customers, A/B winner (one draw; plan baseline 2)",
    "random_any_email": "Random customers, random email (one draw; plan baseline 1)",
}
RANKING_LABELS: dict[str, str] = {
    "recency_rule": "recency rule",
    "response_model": "response model",
    "uplift": "uplift model",
}
ARM_COUNT_GROUPS: dict[str, str] = {
    "uplift": "Uplift policy, targeted",
    "response_model": "Response model, targeted",
    "uplift_mens_default": "Mens-by-default policy (exploratory), targeted",
    "do_not_email": "Do-not-email list",
    "recency_rule": "Recency rule, targeted (no model)",
    "random": "Random draw (no model)",
}
RANKINGS: tuple[str, ...] = ("uplift", "response_model", "recency_rule")
#: Outcomes on which a ranking can qualify under the decision rule (analysis plan D10).
DECISION_OUTCOMES: tuple[str, ...] = ("spend", "visit")
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


def _signed_range(d: Row, fmt: Callable[[float], str] = num) -> str:
    """Format a paired difference with its interval; a star marks one that excludes zero."""
    sign = "+" if d["diff"] > 0 else ("-" if d["diff"] < 0 else "")
    star = "*" if d["meaningful"] else ""
    return (
        f"{sign}{fmt(abs(d['diff']))}{star} "
        f"({_signed(d['ci_low'], fmt)} to {_signed(d['ci_high'], fmt)})"
    )


def _signed(x: float, fmt: Callable[[float], str]) -> str:
    return f"-{fmt(abs(x))}" if x < 0 else fmt(x)


def _vs_expected(results: Results, policy: str, outcome: str) -> Row:
    """Row of the comparison with the exact expectation of random targeting."""
    return find(results["policy"]["versus_random_expected"], policy=policy, outcome=outcome)


def _adjustment_note(results: Results, rows: list[Row]) -> str:
    """Say what the adjusted p-value of each headline row is adjusted for."""
    families = results["frequentist"]["families"]
    parts = []
    for family in dict.fromkeys(row["family"] for row in rows):
        spec = families[family]
        parts.append(
            f"{ADJUSTMENT_NAMES[spec['adjustment']]} is the correction across the "
            f"{spec['n_tests']} {family} tests ({FAMILY_PROSE[family]})"
        )
    return f"{'; '.join(parts)}. Intervals are unadjusted."


def _adjustment_label(results: Results, row: Row) -> str:
    """Name the correction behind a row's adjusted p-value, e.g. ``Holm, 2 primary tests``."""
    spec = results["frequentist"]["families"][row["family"]]
    return f"{ADJUSTMENT_NAMES[spec['adjustment']]}, {spec['n_tests']} {row['family']} tests"


def summary(results: Results) -> str:
    """Write the decision and the headline numbers with their intervals."""
    winner, other = _winner(results)
    name, other_name = EMAIL_NAMES[winner], EMAIL_NAMES[other]
    policy = results["policy"]
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
        [
            f"{_versus(r)}: spend per customer",
            *_effect(r),
            p_value(r["p_adj"]),
            _adjustment_label(results, r),
        ]
        for r in headline_rows[:3]
    ]
    rows.append(
        [
            f"{_versus(visit)}: visit rate ({pct(visit['mean_control'])} → "
            f"{pct(visit['mean_treatment'])})",
            *_effect(visit),
            p_value(visit["p_adj"]),
            _adjustment_label(results, visit),
        ]
    )
    everyone = _policy(results, f"all_{winner}", "spend")
    thousands = [round(everyone[k], -3) for k in ("incremental_total", "ci_low", "ci_high")]
    suppressed = find(policy["suppressed_group"], outcome="visit")
    return "\n".join(
        [
            f"**Decision: send the {name} email, to everyone who can be emailed.** {lead} "
            "This assumes an email costs nothing: the data hold no cost, margin or "
            "unsubscribe figures.",
            "",
            table(
                [
                    "Headline (two-week window)",
                    "Estimate",
                    "95% interval",
                    "Adjusted p",
                    "Correction",
                ],
                rows,
            ),
            "",
            _adjustment_note(results, headline_rows),
            "",
            f"- **Value.** Sent to all {num(policy['n_customers'])} customers, the {name} email "
            f"brings about {usd(thousands[0], 0)} of extra revenue in two weeks "
            f"({usd_range(thousands[1], thousands[2], 0)}). Profit cannot be computed: the "
            "data hold no cost or margin.",
            _targeting_summary(results),
            "- **Whom not to email.** "
            + (
                f"Nobody identified: the {num(suppressed['n_suppressed'])} customers the model "
                "ranks lowest still visit more when emailed, "
                f"{pp(suppressed['uplift_per_customer'], signed=True)} "
                f"({pp_range(suppressed['ci_low'], suppressed['ci_high'])})."
                if suppressed["ci_low"] > 0
                else f"The {num(suppressed['n_suppressed'])} customers the model ranks lowest "
                "show no clear gain from an email: "
                f"{pp(suppressed['uplift_per_customer'], signed=True)} "
                f"({pp_range(suppressed['ci_low'], suppressed['ci_high'])})."
            ),
        ]
    )


NUMBER_WORDS: dict[int, str] = {2: "two", 3: "three", 4: "four", 6: "six", 8: "eight", 9: "nine"}


def _targeting_summary(results: Results) -> str:
    """Write the one first-screen statement of what to do with a limited budget.

    The recommendation is the decision rule's (analysis plan D10), read from
    ``results.json``; the caveats listed after it are each included only when the results
    show them.
    """
    pol = results["policy"]
    budget = num(pol["budget"])
    lead = (
        f"- **If only {budget} emails can be sent.** Three rankings were tested: an uplift "
        f"model, a response model predicting who visits, and the {budget} most recent "
        "purchasers. "
    )
    rec = pol["recommendation"]
    chosen = rec["recommended"]
    if chosen is None:
        return (
            lead + "By the decision rule fixed before the corrected results were computed "
            "(deviation D10), none is recommended: none beats random targeting on spend or "
            f"on visits. Targeting is not shown to add value at this budget; any of the "
            f"three, or a random {budget} customers, is defensible."
        )
    label = RANKING_LABELS[chosen]
    outcome = rec["qualifying"][chosen][0]
    fmt = OUTCOME_FORMATS[outcome]
    gain = _vs_expected(results, chosen, outcome)
    only = len(rec["qualifying"]) == 1
    sign = "+" if gain["diff"] > 0 else ""
    text = (
        lead + "The decision rule fixed before the corrected results were computed (deviation "
        f"D10) picks the {label}, "
        + ("the only ranking that beats" if only else "the simplest unseparated ranking to beat")
        + f" random targeting on {OUTCOME_PLURALS[outcome]}: {sign}{_signed(gain['diff'], fmt)} "
        f"({_signed(gain['ci_low'], fmt)} to {_signed(gain['ci_high'], fmt)})"
    )
    caveats = []
    if gain["ci_low"] < 0.1 * gain["diff"]:
        caveats.append("the lower bound is near zero")
    n_comparisons = len(RANKING_LABELS) * len(DECISION_OUTCOMES)
    caveats.append(
        f"it is one of {NUMBER_WORDS.get(n_comparisons, str(n_comparisons))} unadjusted comparisons"
    )
    trained_on = pol["learner_outcome"]
    if (
        chosen != "recency_rule"
        and outcome != trained_on
        and not _vs_expected(results, chosen, trained_on)["policy_better"]
    ):
        caveats.append(
            f"it does not beat random targeting on {OUTCOME_PLURALS[trained_on]} (the outcome "
            "it was trained on)"
        )
    if not any(chosen in (a, b) for a, b, _ in _separated_rankings(results)):
        caveats.append("it is not separated from the other two rankings")
    seeds = (results.get("seed_robustness") or {}).get("beats_random_expectation", {})
    count = seeds.get(chosen, {}).get(outcome)
    stable = ""
    if count and count["out_of"]:
        if count["count"] < count["out_of"]:
            caveats.append(
                f"it qualifies on {OUTCOME_PLURALS[outcome]} in {count['count']} of "
                f"{count['out_of']} fold seeds"
            )
        else:
            stable = (
                f" It qualifies on {OUTCOME_PLURALS[outcome]} in all {count['out_of']} fold seeds."
            )
    rival = "random targeting" if chosen == "recency_rule" else "the recency rule"
    return (
        f"{text}. The evidence is thin: {join_words(caveats)}.{stable} Before relying on it, "
        f"test it directly: a new randomised test of {label} targeting against {rival}."
    )


def _recommendation_text(results: Results) -> str:
    """State what the decision rule of analysis plan D10 gives on the current results."""
    pol = results["policy"]
    budget = num(pol["budget"])
    winner, _ = _winner(results)
    rec = pol["recommendation"]
    qualifying = rec["qualifying"]
    if rec["recommended"] is None:
        return (
            f"By the decision rule (analysis plan D10), no ranking is recommended: none of "
            f"the recency rule, the response model and the uplift model has a paired "
            f"difference against random targeting that excludes zero on spend or on visits. "
            f"Targeting is not shown to add value at this budget; any of the rankings, or a "
            f"random {budget} customers given the {EMAIL_NAMES[winner]} email, is defensible."
        )
    qualifiers = join_words(
        f"the {RANKING_LABELS[r]} (on {join_words(OUTCOME_PLURALS[o] for o in outcomes)})"
        for r, outcomes in qualifying.items()
    )
    chosen = RANKING_LABELS[rec["recommended"]]
    simplest = next(iter(qualifying))
    reason = (
        "it is the only ranking that qualifies"
        if len(qualifying) == 1
        else "it is the simplest ranking that qualifies"
        if rec["recommended"] == simplest
        else "it qualifies and is separated from the simpler rankings that qualify"
    )
    return (
        f"By the decision rule (analysis plan D10), use the {chosen}: {reason}. A ranking "
        "qualifies when its paired difference against random targeting excludes zero on "
        f"spend or on visits; here that holds for {qualifiers}."
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

    q3 = _recommendation_text(results)

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
            f"the {EMAIL_NAMES[recommendation['recommended_email']]} email. The Bayesian "
            "bootstrap puts a uniform Dirichlet prior on the weights of the observed spend "
            "values and no prior on their size, so with arms this large it is expected to "
            "agree with the frequentist readout, and it does.",
        ]
    )


def cuped(results: Results) -> str:
    """Why covariate adjustment barely reduces variance here, with the numbers."""
    by_metric = results["cuped"]["variance_reduction_by_metric"]
    spend, visit = by_metric["spend"], by_metric["visit"]
    row = find(results["cuped"]["rows"], metric="spend", comparison="mens_vs_control")
    mde = {
        (email, flag): find(
            results["power"]["mde_per_arm_rows"],
            metric="spend",
            treatment=email,
            alpha_label="holm",
            cuped=flag,
        )
        for email in ("mens", "womens")
        for flag in (False, True)
    }
    mde_text = join_words(
        f"{usd(mde[email, False]['mde_absolute'])} for {EMAIL_NAMES[email]} "
        f"({usd(mde[email, True]['mde_absolute'])} with CUPED; observed "
        f"{usd(mde[email, False]['observed_diff'])})"
        for email in ("mens", "womens")
    )
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
            f"{usd(row['lin_se'], 5)} with the regression. A negative value in the table "
            "means the adjustment added a little noise.",
            f"- **Minimum detectable effect.** With each arm's own variance, at the Holm "
            f"level and {pct(mde['mens', False]['power'], 0)} power, the smallest spend effect "
            f"the experiment could detect is {mde_text}. CUPED does not move it. The emails "
            "raise the variance of spend in their own arm, so these are larger than a "
            "planning figure based on the control variance alone.",
            f"- **Why.** {pct(zero)} of customers spend nothing in the two weeks, so spend is "
            "almost entirely a rare, noisy purchase event that last year's spend cannot "
            f"predict (correlation at most {spend['rho_max']:.3f}). Visits are more "
            f"predictable, and there the full regression removes up to "
            f"{pct(visit['lin_variance_reduction_max'])}. CUPED pays off when the covariate is "
            "the same metric measured before the experiment on returning users; a one-off "
            "campaign with a two-week window is not that setting.",
        ]
    )


def peeking(results: Results, reports_dir: Path) -> str:
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

    looks_table = pd.read_csv(reports_dir / seq["looks_table"])
    single = looks_table[(looks_table["metric"] == "visit") & (looks_table["n_looks"] == 1)]
    single = single.set_index("method")
    rows = []
    for method, label in SEQUENTIAL_NAMES.items():
        aa, ab = row("aa", method), row("ab_mde", method)
        one = single.loc[method]
        rows.append(
            [
                label,
                f"{pct(one['reject_rate'])} (±{pct(one['reject_mcse'])})",
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
                    "Rule",
                    "False positive rate with a single look (± Monte Carlo SE)",
                    f"False positive rate with {looks} daily looks",
                    f"Power at a {pct(mde['relative'])} relative lift",
                    "Mean stopping day at that lift",
                ],
                rows,
            ),
            "",
            "Compare each rule with its own single-look rate. The Bayesian rule is two-sided "
            "at 10% by construction, so it starts from about twice the rate of the 5% test; "
            "peeking multiplies both. The single-look column uses "
            f"{num(int(single['n_runs'].iloc[0]))} runs.",
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
        seed_rows = []
        spread = {(s["ranking"], s["metric"]): s for s in robustness["spread"]}
        beats = robustness.get("beats_random_expectation", {})
        for ranking, label in (("uplift", "Uplift model"), ("response_model", "Response model")):
            for outcome in POLICY_OUTCOMES:
                metric = f"{ranking}_policy_incremental_{OUTCOME_PLURALS[outcome]}"
                s = spread.get((ranking, metric))
                if s is None:
                    continue
                fmt = OUTCOME_FORMATS[outcome]
                count = beats.get(ranking, {}).get(outcome)
                seed_rows.append(
                    [
                        f"{label}, incremental {OUTCOME_PLURALS[outcome]}",
                        fmt(s["seed_0"]),
                        fmt(s["median"]),
                        f"{fmt(s['min'])} to {fmt(s['max'])}",
                        f"{count['count']} of {count['out_of']}" if count else "not computed",
                    ]
                )
        if seed_rows:
            lines += [
                "",
                f"**Policy values across fold seeds** ({num(robustness['budget'])} emails, "
                f"{robustness['n_seeds']} seeds). The bootstrap intervals in the next section "
                "hold the folds fixed, so this spread comes on top of them. The last column "
                "counts the seeds in which the policy's interval against the random "
                "expectation lies above zero; it is reported for robustness and does not "
                "feed the decision rule, which uses the committed seed.",
                "",
                table(
                    [
                        "Policy and outcome",
                        "Committed seed",
                        "Median",
                        "Range",
                        "Seeds beating the random expectation",
                    ],
                    seed_rows,
                ),
                "",
            ]
    lines.append(
        "- **The choice of the X-learner for the policy is post hoc (deviation D8).** It was "
        "picked from three learners after an earlier evaluation on the same data, which the "
        "correction of deviation D11 has since replaced; the choice was not revisited, so "
        "that the policy is not selected a second time on the corrected results. Read "
        '"chosen", not "shown to be better": '
        + (
            "in the Qini table it is the top-ranked learner on the Womens pair"
            if models["summary"]["womens_visit"]["best_ranking"] == "x_learner"
            else "in the Qini table it is not the top-ranked learner on the Womens pair"
        )
        + ", and it is not separated from the other learners that rank near it."
    )
    return "\n".join(lines)


def _money(x: float) -> str:
    return usd(x, 0)


OUTCOME_FORMATS: dict[str, Callable[[float], str]] = {
    "visit": num,
    "conversion": num,
    "spend": _money,
}


def _value_cell(row: Row, fmt: Callable[[float], str]) -> str:
    return (
        f"{_signed(row['incremental_total'], fmt)} "
        f"({_signed(row['ci_low'], fmt)} to {_signed(row['ci_high'], fmt)})"
    )


def policy(results: Results, reports_dir: Path) -> str:
    """Tabulate the fixed-budget policies, their comparison with random targeting and the rule."""
    pol = results["policy"]
    budget = pol["budget"]
    winner, _ = _winner(results)
    value_rows, versus_rows = [], []
    for name, label in POLICY_NAMES.items():
        counts = find(pol["rows"], policy=name, outcome="visit")
        value_rows.append(
            [
                label,
                f"{num(counts['n_mens'])} / {num(counts['n_womens'])}",
                *(
                    _value_cell(_policy(results, name, o), OUTCOME_FORMATS[o])
                    for o in POLICY_OUTCOMES
                ),
            ]
        )
        versus_rows.append(
            [
                label,
                *(
                    _signed_range(_vs_expected(results, name, o), OUTCOME_FORMATS[o])
                    for o in POLICY_OUTCOMES
                ),
            ]
        )
    expected = {o: _vs_expected(results, "recency_rule", o) for o in POLICY_OUTCOMES}

    pairs = [(a, b) for i, a in enumerate(RANKINGS) for b in RANKINGS[i + 1 :]]
    separated = _separated_rankings(results)
    pair_text = join_words(
        f"{RANKING_PROSE[a]} minus {RANKING_PROSE[b]} "
        f"{_signed_range(_paired(results, a, b, 'visit'))}"
        for a, b in pairs
    )
    between = (
        "**No ranking is separated from another** on visits, conversions or spend. In "
        f"visits: {pair_text}."
        if not separated
        else "**Between rankings**, the paired differences that exclude zero are "
        + join_words(
            f"{RANKING_PROSE[a]} against {RANKING_PROSE[b]} on {OUTCOME_PLURALS[o]}"
            for a, b, o in separated
        )
        + f". In visits: {pair_text}."
    )

    curve = pd.read_csv(reports_dir / pol["budget_curve_table"])
    grid = curve[(curve["n_emailed"] > 0) & (curve["n_emailed"] < pol["n_customers"])]
    counts = {}
    for column in ("incremental_visits", "incremental_spend"):
        wide = grid.pivot_table(index="n_emailed", columns="policy", values=column)
        counts[column] = int((wide["recency_rule"] >= wide["uplift"]).sum()), len(wide)
    step = int(curve["n_emailed"].drop_duplicates().sort_values().iloc[1])
    sup = {o: find(pol["suppressed_group"], outcome=o) for o in ("visit", "spend")}
    no_group = sup["visit"]["ci_low"] > 0

    arm_rows = []
    by_group = {r["group"]: r for r in pol["arm_counts"]}
    for group, label in ARM_COUNT_GROUPS.items():
        r = by_group[group]
        arm_rows.append(
            [
                label,
                num(r["n_mens"]),
                num(r["n_womens"]),
                num(r["n_control"]),
                f"{num(r['expected_mens'])} / {num(r['expected_womens'])} / "
                f"{num(r['expected_control'])}",
                f"{r['p_value']:.2f}" if r["p_value"] >= 0.005 else p_value(r["p_value"]),
                "yes" if r["passed"] else "**no**",
            ]
        )
    all_pass = all(r["passed"] for r in pol["arm_counts"])
    return "\n".join(
        [
            table(
                [
                    f"Policy, {num(budget)} emails",
                    "Mens / Womens emails",
                    "Incremental visits",
                    "Incremental conversions",
                    "Incremental spend",
                ],
                value_rows,
            ),
            "",
            "Increments over emailing nobody with 95% intervals, estimated by inverse "
            f"propensity weighting with the design probability and a paired bootstrap "
            f"({num(pol['bootstrap']['n_resamples'])} resamples). The recency rule sends the "
            f"A/B winner ({EMAIL_NAMES[winner]}).",
            "",
            "**Against random targeting.** Random targeting of the same number of customers "
            f"with the {EMAIL_NAMES[winner]} email has a known expectation, the number of "
            f"emails times the average effect: {num(expected['visit']['expected_random'])} "
            f"visits, {num(expected['conversion']['expected_random'])} conversions and "
            f"{usd(expected['spend']['expected_random'], 0)} of spend. The table gives each "
            "policy minus that expectation, with both recomputed on the same bootstrap "
            "resamples; a star marks an interval that excludes zero.",
            "",
            table(
                [
                    "Policy minus the random expectation",
                    "Visits",
                    "Conversions",
                    "Spend",
                ],
                versus_rows,
            ),
            "",
            f"- **Decision rule.** {_recommendation_text(results)} The rule was written after "
            "the uncorrected results had been seen and before these numbers were computed.",
            f"- {between}",
            f"- **Budget curve.** Extending each ranking to every budget in steps of "
            f"{num(step)}, the recency rule is at or above the uplift policy at "
            f"{counts['incremental_visits'][0]} of {counts['incremental_visits'][1]} budgets on "
            f"visits and at {counts['incremental_spend'][0]} of "
            f"{counts['incremental_spend'][1]} on spend. These are point estimates.",
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
            "",
            "**Does the selection depend on the arm?** A policy built from pre-treatment "
            "variables and out-of-fold scores must pick customers from the three arms in the "
            "proportions of the whole population; if it does not, its value above is biased. "
            "Counts of the selected customers by the arm they were randomised to:",
            "",
            table(
                [
                    "Selected group",
                    "Mens arm",
                    "Womens arm",
                    "Control",
                    "Expected (Mens / Womens / Control)",
                    "p",
                    "Passes",
                ],
                arm_rows,
            ),
            "",
            (
                "Every group passes the sample-ratio test"
                if all_pass
                else "Not every group passes the sample-ratio test"
            )
            + f" (alarm threshold {pol['arm_counts'][0]['alpha']}). An earlier version of the "
            "scoring failed it; that error and its correction are deviation D11.",
        ]
    )


def random_targeting(results: Results) -> str:
    """Put the random draws next to the exact expectation, for every outcome."""
    pol = results["policy"]
    budget = pol["budget"]
    winner, _ = _winner(results)
    labels = {
        "visit": "Incremental visits",
        "conversion": "Incremental conversions",
        "spend": "Incremental spend",
    }
    rows = []
    for outcome, label in labels.items():
        fmt = OUTCOME_FORMATS[outcome]
        row = _vs_expected(results, "random", outcome)
        rows.append(
            [
                label,
                f"{fmt(row['expected_random'])} "
                f"({_signed(row['expected_ci_low'], fmt)} to "
                f"{_signed(row['expected_ci_high'], fmt)})",
                _value_cell(_policy(results, "random", outcome), fmt),
                _signed_range(row, fmt),
                _value_cell(_policy(results, "random_any_email", outcome), fmt),
            ]
        )
    return "\n".join(
        [
            table(
                [
                    f"{num(budget)} random customers",
                    f"Exact expectation, {EMAIL_NAMES[winner]} email",
                    f"One draw, {EMAIL_NAMES[winner]} email (baseline 2)",
                    "That draw minus the expectation",
                    "Same draw, random email (baseline 1)",
                ],
                rows,
            ),
            "",
            "The expectation is the budget times the average effect and needs no draw; its "
            "interval comes from the bootstrap of that effect. The draw is one random set of "
            "customers (analysis plan §11, deviations D9 and D13). Its difference from the "
            "expectation is the luck of that one draw; a star marks an outcome on which the "
            "drawn customers landed measurably off. That is why the rankings are compared "
            "with the expectation and not with the draw (deviation D12).",
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
            "The feature builder refuses any outcome or arm column by name, and before any "
            "fit a feature whose values copy or recode the outcome or the treatment is "
            "rejected; every score is out of fold. A noisy function of an outcome cannot be "
            "detected from values.",
            "Results inflated by information from the outcome.",
        ),
        (
            "Selection independent of the arm",
            "On pure-noise features a rule that picks the highest scores selects customers "
            "from the three arms in the design proportions, and the earlier fold-averaged "
            "scoring fails the same sample-ratio test; the committed policies must pass it.",
            "A policy that selects on the realised arm, which biases its estimated value.",
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
        "peeking": peeking(results, reports_dir),
        "multiple_testing": multiple_testing(results),
        "uplift_models": uplift_models(results),
        "policy": policy(results, reports_dir),
        "random_targeting": random_targeting(results),
        "verification": verification(results),
    }
