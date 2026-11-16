"""Targeting: incremental visits and spend against the email budget, and who gets targeted."""

import altair as alt
import pandas as pd
import streamlit as st

from abkit import appdata as ad
from abkit import plots

ad.page_setup("Targeting")
policy = ad.load_results()["policy"]
curve = ad.load_table(policy["budget_curve_table"])

policies = [p for p in ad.POLICY_LABELS if p in set(curve["policy"])]
grid = sorted(int(k) for k in curve["n_emailed"].unique())
st.markdown(
    "Emailing everyone is not always possible. Choose how many customers can be emailed and "
    "compare ways of choosing them: by predicted uplift, by predicted response, or by simpler "
    "rules."
)
budget = st.select_slider(
    "Customers emailed",
    options=grid,
    value=policy["budget"],
    format_func=lambda k: f"{k:,}",
    key="budget",
)
at_budget = curve[curve["n_emailed"] == budget].set_index("policy")

# --- At the chosen budget ------------------------------------------------------------------------

st.header(f"At {budget:,} emails")
st.dataframe(
    pd.DataFrame(
        {
            "Policy": [ad.POLICY_LABELS[p] for p in policies],
            "Incremental visits": [
                f"{at_budget.loc[p, 'incremental_visits']:,.0f}" for p in policies
            ],
            "Incremental spend": [
                f"${at_budget.loc[p, 'incremental_spend']:,.0f}" for p in policies
            ],
        }
    ),
    hide_index=True,
)
st.caption(
    "Increments are relative to emailing nobody, estimated by inverse propensity weighting. "
    "The curve holds point estimates only, on a grid of "
    f"{grid[1] - grid[0]:,} customers."
)

panels = (
    ("incremental_visits", "Incremental visits", ",.0f"),
    ("incremental_spend", "Incremental spend, $", "$,.0f"),
)
plotted = curve[curve["policy"].isin(policies)].assign(
    label=lambda d: d["policy"].map(ad.POLICY_LABELS)
)
colours = alt.Scale(
    domain=[ad.POLICY_LABELS[p] for p in policies],
    range=[ad.POLICY_COLORS[p] for p in policies],
)
dashes = alt.Scale(
    domain=[ad.POLICY_LABELS[p] for p in policies],
    range=[[1, 0], [6, 3], [6, 2, 2, 2], [2, 2]][: len(policies)],
)
for column, title, number_format in panels:
    lines = (
        alt.Chart(plotted)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("n_emailed:Q", title="Customers emailed", axis=alt.Axis(format="~s")),
            y=alt.Y(f"{column}:Q", title=title),
            color=alt.Color("label:N", scale=colours, title=None),
            strokeDash=alt.StrokeDash("label:N", scale=dashes, title=None),
            tooltip=[
                alt.Tooltip("label:N", title="Policy"),
                alt.Tooltip("n_emailed:Q", title="Emailed", format=","),
                alt.Tooltip(f"{column}:Q", title=title, format=number_format),
            ],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"budget": [budget]}))
        .mark_rule(color=plots.TEXT_SECONDARY, strokeDash=[2, 3])
        .encode(x="budget:Q")
    )
    st.altair_chart(
        ad.chart_style((lines + marker).properties(height=280, title=f"{title} by emails sent")),
        theme=None,
    )

# --- The pre-registered budget, with uncertainty -------------------------------------------------

fixed = policy["budget"]
st.header(f"At the {fixed:,} budget, with uncertainty")
names = ("uplift", "response_model", "recency_rule", "random")


def interval(policy_name: str, outcome: str, money: bool) -> str:
    """Point estimate and bootstrap interval of one policy's increment at the fixed budget."""
    row = policy["headline"][outcome][policy_name]
    unit = "$" if money else ""
    return (
        f"{unit}{row['incremental_total']:,.0f} "
        f"({unit}{row['ci_low']:,.0f} to {unit}{row['ci_high']:,.0f})"
    )


st.dataframe(
    pd.DataFrame(
        {
            "Policy": [ad.POLICY_LABELS[p] for p in names],
            "Incremental visits (95% interval)": [interval(p, "visit", False) for p in names],
            "Incremental spend (95% interval)": [interval(p, "spend", True) for p in names],
        }
    ),
    hide_index=True,
)
versus = policy["uplift_versus"]["visit"]
separated = [ad.POLICY_LABELS[p].lower() for p in names[1:] if versus[p]["meaningful"]]
within = [ad.POLICY_LABELS[p].lower() for p in names[1:] if not versus[p]["meaningful"]]
verdict = []
if within:
    verdict.append("is not separated from " + ", ".join(within))
if separated:
    verdict.append("differs meaningfully from " + ", ".join(separated))
st.info(
    "On incremental visits, by paired bootstrap, the uplift policy " + " and ".join(verdict) + "."
)
st.caption(
    f"Bootstrap intervals from {policy['bootstrap']['n_resamples']:,} resamples that hold the "
    "model scores fixed. The uplift policy uses a learner chosen after the model evaluation "
    f"was read ([analysis plan, deviation D8]({ad.doc_url('docs/analysis_plan.md')})), so its "
    "values are optimistic."
)

# --- Who is targeted -----------------------------------------------------------------------------

st.header(f"Who the uplift policy targets (fixed at the {fixed:,} budget)")
st.caption("This profile does not move with the slider.")
profiles = pd.DataFrame(policy["profiles"]).pivot_table(
    index="variable", columns="group", values="value", aggfunc="first"
)
rows = (
    ("share_assigned_mens", "Assigned the Mens email", "share"),
    ("share_assigned_womens", "Assigned the Womens email", "share"),
    ("mean_recency", "Months since last purchase, mean", "number"),
    ("share_recency_le_3", "Purchased in the last 3 months", "share"),
    ("mean_history", "Past-year spend, mean", "money"),
    ("median_history", "Past-year spend, median", "money"),
    ("share_mens_buyers", "Bought men's merchandise", "share"),
    ("share_womens_buyers", "Bought women's merchandise", "share"),
    ("share_both_buyers", "Bought both", "share"),
    ("share_newbie", "New customer in the past year", "share"),
    ("share_zip_code_urban", "Urban", "share"),
    ("share_zip_code_suburban", "Suburban", "share"),
    ("share_zip_code_rural", "Rural", "share"),
    ("share_channel_web", "Bought on the web", "share"),
    ("share_channel_phone", "Bought by phone", "share"),
    ("share_channel_multichannel", "Bought on both channels", "share"),
)


def show(value: float, kind: str) -> str:
    """Format one profile value; a missing value (not defined for the group) shows a dash."""
    if not ad.is_finite(value):
        return "-"
    if kind == "share":
        return f"{100 * value:.1f}%"
    return f"${value:,.0f}" if kind == "money" else f"{value:.1f}"


st.dataframe(
    pd.DataFrame(
        {
            "Characteristic": [label for _, label, _ in rows],
            f"Targeted ({profiles.loc['n', 'targeted_uplift']:,.0f})": [
                show(profiles.loc[v, "targeted_uplift"], kind) for v, _, kind in rows
            ],
            f"All customers ({profiles.loc['n', 'all']:,.0f})": [
                show(profiles.loc[v, "all"], kind) for v, _, kind in rows
            ],
        }
    ),
    hide_index=True,
)
