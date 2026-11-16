"""Home: the project in three sentences, the decision and three headline numbers."""

import streamlit as st

from abkit import appdata as ad
from abkit import config, sequential

ad.page_setup("Hillstrom email experiment")
results = ad.load_results()

recommendation = results["bayes_spend"]["recommendation"]
email = recommendation["recommended_email"]
comparison = f"{email}_vs_{config.CONTROL}"
freq = ad.find_row(results["frequentist"]["rows"], comparison=comparison, metric="spend")
bayes = ad.find_row(
    results["bayes_spend"]["rows"],
    comparison=comparison,
    metric="spend",
    model="bayesian_bootstrap",
)
peeking = results["sequential"]
naive = ad.find_row(
    peeking["rows"],
    scenario=sequential.SCENARIO_AA,
    metric=sequential.HEADLINE_METRIC,
    method="naive",
)
policy = results["policy"]
n_customers = sum(results["validity"]["srm"]["counts"].values())

st.markdown(
    f"A retailer randomly assigned {n_customers:,} recent customers to a Mens email, a Womens "
    f"email or no email and tracked visits, purchases and spend for {peeking['days']} days "
    "(the Hillstrom MineThatData challenge). "
    "This project reads that experiment the way a product team would: a pre-registered plan, "
    "frequentist and Bayesian readouts with variance reduction, simulations of peeking and "
    "multiple testing, and uplift models for targeting. "
    "Every number in this app is read from the pipeline's committed results or simulated live."
)

# --- The decision --------------------------------------------------------------------------------

st.header("The decision")
others = [e for e in config.TREATMENTS if e != email]
qualifying = recommendation["qualifying_emails"]
p_mens_over_womens = recommendation["p_mens_better_than_womens"]
p_recommended = p_mens_over_womens if email == "mens" else 1 - p_mens_over_womens
if len(qualifying) == len(config.TREATMENTS):
    why = (
        "Both emails beat no email on spend per customer, the primary metric. "
        f"The {ad.ARM_LABELS[email]} has the higher posterior mean spend, with probability "
        f"{ad.format_probability(p_recommended)} of beating the {ad.ARM_LABELS[others[0]]}."
    )
else:
    why = (
        f"The {ad.ARM_LABELS[email]} is the only email that meets the pre-registered rule on "
        "spend per customer, the primary metric."
    )
st.success(f"**Send the {ad.ARM_LABELS[email]}.** {why}")


def targeting_sentence() -> str:
    """Say how the uplift policy compares with the two simpler rules, as the results have it."""
    versus = policy["uplift_versus"]["visit"]
    names = {"response_model": "the response model", "recency_rule": "the recency rule"}
    noise = [names[p] for p in names if not versus[p]["meaningful"]]
    beats = [names[p] for p in names if versus[p]["meaningful"] and versus[p]["reference_better"]]
    trails = [
        names[p] for p in names if versus[p]["meaningful"] and not versus[p]["reference_better"]
    ]
    parts = []
    if noise:
        parts.append("is not separated from " + " or ".join(noise))
    if beats:
        parts.append("beats " + " and ".join(beats))
    if trails:
        parts.append("trails " + " and ".join(trails))
    sentence = (
        f"With a budget of {policy['budget']:,} emails, the uplift policy "
        + ", ".join(parts)
        + " on incremental visits."
    )
    if len(noise) == len(names):
        sentence += (
            " The experiment gives no evidence that the uplift model picks customers better "
            "than these simpler rules."
        )
    return sentence


st.info(f"**Whom to email.** {targeting_sentence()}")

# --- Three headline numbers ----------------------------------------------------------------------

st.header("Three numbers")
left, middle, right = st.columns(3)
with left:
    st.metric("Incremental spend per customer", ad.format_effect(freq["diff"], "spend"))
    st.caption(
        f"{ad.COMPARISON_LABELS[comparison]}. 95% CI "
        f"{ad.format_interval(freq['ci_low'], freq['ci_high'], 'spend')} ({freq['ci_method']})."
    )
with middle:
    st.metric("Probability the email beats no email", ad.format_probability(bayes["p_better"]))
    st.caption(
        f"On spend per customer. Bayesian bootstrap, {bayes['n_draws']:,} draws; Monte Carlo "
        f"standard error {100 * bayes['p_better_mcse']:.3f} pp."
    )
with right:
    st.metric("False positive rate of naive peeking", f"{100 * naive['reject_rate']:.1f}%")
    st.caption(
        f"Against the nominal {100 * peeking['alpha']:.0f}%, when an A/A test is checked on "
        f"each of {naive['n_looks']} days. {naive['n_runs']:,} simulations; Monte Carlo "
        f"standard error {100 * naive['reject_mcse']:.2f} pp."
    )

# --- Where to go next ----------------------------------------------------------------------------

st.header("Explore")
st.page_link("pages/1_Readout.py", label="Readout: every comparison and metric")
st.page_link("pages/2_Peeking_simulator.py", label="Peeking simulator: run it live")
st.page_link("pages/3_Targeting.py", label="Targeting: whom to email under a budget")
st.page_link("pages/4_Method.py", label="Method: how each number is computed")
st.caption(f"Code, analysis plan and methodology: [{ad.REPO_URL}]({ad.REPO_URL})")
