"""Readout: one comparison and metric at a time, frequentist and Bayesian side by side."""

import altair as alt
import pandas as pd
import streamlit as st

from abkit import appdata as ad
from abkit import plots

ad.page_setup("Readout")
results = ad.load_results()

left, right = st.columns(2)
comparison = left.selectbox(
    "Comparison",
    list(ad.COMPARISON_LABELS),
    format_func=ad.COMPARISON_LABELS.get,
    key="comparison",
)
metric = right.selectbox(
    "Metric", list(ad.METRIC_LABELS), format_func=ad.METRIC_LABELS.get, key="metric"
)

freq = ad.find_row(results["frequentist"]["rows"], comparison=comparison, metric=metric)
adjusted = ad.find_row(results["cuped"]["rows"], comparison=comparison, metric=metric)
if metric in ad.RATE_METRICS:
    bayes_section, bayes_filter = results["bayes_rates"], {"prior": "flat"}
    bayes_model = "Beta-Binomial with a flat prior"
else:
    bayes_section, bayes_filter = results["bayes_spend"], {"model": "bayesian_bootstrap"}
    bayes_model = "Bayesian bootstrap"
loss_fraction = next(
    v for k, v in bayes_section["decision_rule"].items() if k.startswith("loss_fraction")
)
bayes = ad.find_row(bayes_section["rows"], comparison=comparison, metric=metric, **bayes_filter)
treatment, control = ad.ARM_LABELS[freq["treatment"]], ad.ARM_LABELS[freq["control"]]

# --- Validity ------------------------------------------------------------------------------------

st.header("Is the experiment valid?")
srm, balance = results["validity"]["srm"], results["validity"]["balance"]
srm_text = (
    f"**Sample ratio:** p = {ad.format_p(srm['p_value'])} for equal thirds "
    f"(the readout is blocked below {srm['alpha']})."
)
(st.success if srm["passed"] else st.error)(srm_text)
balance_text = (
    f"**Covariate balance:** {balance['n_flagged']} covariates flagged. Largest standardised "
    f"mean difference {balance['max_abs_smd']:.3f} against a threshold of "
    f"{balance['threshold']}."
)
(st.success if balance["n_flagged"] == 0 else st.warning)(balance_text)

# --- Effect --------------------------------------------------------------------------------------

st.header("Effect")
st.caption(f"{treatment} minus {control.lower()}, {ad.METRIC_LABELS[metric].lower()}.")
a, b, c = st.columns(3)
a.metric("Effect (raw difference)", ad.format_effect(freq["diff"], metric))
b.metric(treatment, ad.format_level(freq["mean_treatment"], metric))
c.metric(control, ad.format_level(freq["mean_control"], metric))
st.caption(
    f"95% CI {ad.format_interval(freq['ci_low'], freq['ci_high'], metric)} "
    f"({freq['ci_method']}). Relative lift {100 * freq['rel_lift']:+.1f}% "
    f"({100 * freq['rel_ci_low']:+.1f}% to {100 * freq['rel_ci_high']:+.1f}%)."
)

estimates = pd.DataFrame(
    [
        {
            "estimate": "Raw",
            "effect": freq["diff"],
            "low": freq["ci_low"],
            "high": freq["ci_high"],
            "p_adj": freq["p_adj"],
        },
        {
            "estimate": "CUPED",
            "effect": adjusted["cuped_diff"],
            "low": adjusted["cuped_ci_low"],
            "high": adjusted["cuped_ci_high"],
            "p_adj": adjusted["cuped_p_adj"],
        },
        {
            "estimate": "Regression-adjusted",
            "effect": adjusted["lin_diff"],
            "low": adjusted["lin_ci_low"],
            "high": adjusted["lin_ci_high"],
            "p_adj": adjusted["lin_p_adj"],
        },
    ]
)
scale, unit = ad.display_scale(metric)
shown = estimates.assign(**{k: estimates[k] * scale for k in ("effect", "low", "high")})
order = list(estimates["estimate"])
base = alt.Chart(shown).encode(
    y=alt.Y("estimate:N", sort=order, title=None, axis=alt.Axis(labelLimit=180))
)
chart = (
    alt.Chart(pd.DataFrame({"zero": [0.0]}))
    .mark_rule(color=plots.TEXT_SECONDARY, strokeDash=[4, 3])
    .encode(x="zero:Q")
    + base.mark_rule(color=plots.SERIES, strokeWidth=3).encode(
        x=alt.X("low:Q", title=f"Effect with 95% CI, {unit}"), x2="high:Q"
    )
    + base.mark_point(color=plots.SERIES, filled=True, size=110, opacity=1).encode(
        x="effect:Q",
        tooltip=[
            alt.Tooltip("estimate:N", title="Estimate"),
            alt.Tooltip("effect:Q", title="Effect", format=".3f"),
            alt.Tooltip("low:Q", title="CI low", format=".3f"),
            alt.Tooltip("high:Q", title="CI high", format=".3f"),
        ],
    )
).properties(height=150)
st.altair_chart(ad.chart_style(chart), theme=None)
st.dataframe(
    pd.DataFrame(
        {
            "Estimate": estimates["estimate"],
            "Effect": [ad.format_effect(v, metric) for v in estimates["effect"]],
            "95% CI": [
                ad.format_interval(lo, hi, metric)
                for lo, hi in zip(estimates["low"], estimates["high"], strict=True)
            ],
            "Adjusted p-value": [ad.format_p(p) for p in estimates["p_adj"]],
        }
    ),
    hide_index=True,
)
st.caption(
    "The raw difference is the headline estimate. CUPED uses past-year spend as its covariate "
    "and removed "
    f"{100 * adjusted['cuped_variance_reduction']:.2f}% of the variance here; the regression "
    f"adjustment (Lin, {adjusted['lin_n_covariates']} covariates) removed "
    f"{100 * adjusted['lin_variance_reduction']:.2f}%."
)

# --- Frequentist decision ------------------------------------------------------------------------

st.header("Frequentist test")
family = results["frequentist"]["families"][freq["family"]]
a, b = st.columns(2)
a.metric("Adjusted p-value", ad.format_p(freq["p_adj"]))
b.metric(
    f"Significant at {results['frequentist']['alpha']}", "Yes" if freq["significant"] else "No"
)
st.caption(
    f"{freq['test']}; unadjusted p = {ad.format_p(freq['p_value'])}. Adjusted by "
    f"{ad.ADJUSTMENT_LABELS[freq['adjustment']]} within the {freq['family'].replace('_', ' ')} "
    f"family of {family['n_tests']} tests."
)

# --- Bayesian decision ---------------------------------------------------------------------------

st.header("Bayesian readout")
a, b = st.columns(2)
a.metric(f"P({treatment} is better)", ad.format_probability(bayes["p_better"]))
b.metric("Expected loss of shipping", ad.format_small(bayes["loss_ship"], metric))
st.caption(
    f"{bayes_model}, {bayes['n_draws']:,} posterior draws. Monte Carlo standard error of the "
    f"probability {100 * bayes['p_better_mcse']:.3f} pp. The loss threshold is "
    f"{ad.format_small(bayes['loss_threshold'], metric)}, {100 * loss_fraction:g}% of the "
    f"{control.lower()} level."
)
decision = bayes["decision"]
box = {"ship": st.success, "don't ship": st.error}.get(decision, st.warning)
box(f"**Decision: {decision}.** {bayes['reason']}.")
st.caption(
    "The pre-registered rule decides on spend per customer; for the other metrics it is shown "
    "for information. In the email-against-email comparison, 'ship' means preferring the first "
    "email over the second."
)
