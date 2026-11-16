"""Peeking simulator: a small live simulation of checking an A/B test every day."""

import altair as alt
import streamlit as st

from abkit import appdata as ad
from abkit import plots, sequential

ad.page_setup("Peeking simulator")
committed = ad.load_results()["sequential"]

st.markdown(
    "Checking a running test every day and stopping at the first significant result inflates "
    "the false positive rate. Set up an experiment, choose how it is monitored, and the page "
    "simulates it from scratch."
)

default_rate = round(100 * committed["baselines"][sequential.HEADLINE_METRIC]["mean"], 1)
left, right = st.columns(2)
baseline_pct = left.number_input(
    "Baseline rate, %", min_value=0.5, max_value=60.0, value=default_rate, step=0.5, key="rate"
)
lift_pct = right.number_input(
    "True relative lift, %",
    min_value=0.0,
    max_value=50.0,
    value=0.0,
    step=1.0,
    key="lift",
    help="0 simulates an A/A test: every declared winner is a false positive.",
)
users_per_day = left.number_input(
    "Users per day (both arms)",
    min_value=100,
    max_value=ad.LIVE_MAX_USERS_PER_DAY,
    value=2 * committed["customers_per_arm_per_day"],
    step=100,
    key="users",
)
n_days = right.number_input(
    "Days", min_value=2, max_value=ad.LIVE_MAX_DAYS, value=committed["days"], step=1, key="days"
)
method = st.selectbox(
    "Method", list(ad.PEEKING_METHODS), format_func=ad.PEEKING_METHODS.get, key="method"
)

result = ad.cached_peeking(
    baseline_pct / 100, lift_pct / 100, int(users_per_day), int(n_days), method
)

# --- Result --------------------------------------------------------------------------------------

label = "False positive rate" if result.is_null else "Power"
a, b = st.columns(2)
a.metric(label, f"{100 * result.reject_rate:.1f}%")
b.metric("Average day the test ends", f"{result.mean_stop_day:.1f} of {result.n_days}")
if result.is_null:
    low = result.reject_rate - 1.96 * result.reject_mcse
    high = result.reject_rate + 1.96 * result.reject_mcse
    nominal = f"the nominal {100 * result.alpha:.0f}%"
    if low > result.alpha:
        st.error(f"Above {nominal}: this way of monitoring declares too many false winners.")
    elif high < result.alpha:
        st.info(f"Below {nominal}: the rule is valid, and conservative for this design.")
    else:
        st.success(f"Consistent with {nominal}, within simulation error.")
else:
    st.info(
        "With a real lift, the share of experiments that declare a winner is the power. "
        "Set the lift to 0 to see the false positive rate of the same rule."
    )
st.caption(ad.monte_carlo_note(result))

st.subheader("When the test stops")
table = ad.stop_day_table(result)
table["share_pct"] = 100 * table["share"]
colours = alt.Scale(domain=list(dict.fromkeys(table["outcome"])), range=[plots.SERIES, plots.MUTED])
bars = (
    alt.Chart(table)
    .mark_bar()
    .encode(
        x=alt.X(
            "day:N",
            sort=alt.SortField("order"),
            title="Day the test stops",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y("share_pct:Q", title="Share of simulated experiments, %"),
        color=alt.Color("outcome:N", scale=colours, title=None),
        tooltip=[
            alt.Tooltip("day:N", title="Day"),
            alt.Tooltip("share_pct:Q", title="Share, %", format=".1f"),
            alt.Tooltip("outcome:N", title="Outcome"),
        ],
    )
    .properties(height=300)
)
st.altair_chart(ad.chart_style(bars), theme=None)

naive = ad.find_row(
    committed["rows"],
    scenario=sequential.SCENARIO_AA,
    metric=sequential.HEADLINE_METRIC,
    method="naive",
)
st.caption(
    "This is a small simulation sized to answer in about two seconds. The full study behind "
    f"the project ({naive['n_runs']:,} simulations at the experiment's own size) found a "
    f"false positive rate of {100 * naive['reject_rate']:.1f}% for naive peeking over "
    f"{naive['n_looks']} daily looks (Monte Carlo standard error "
    f"{100 * naive['reject_mcse']:.2f} pp)."
)
