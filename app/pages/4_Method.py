"""Method: a short explanation of each method, linked to the full write-up in the repository."""

import streamlit as st

from abkit import appdata as ad

ad.page_setup("Method")

METHODOLOGY = "docs/methodology.md"
#: (title, explanation, heading of docs/methodology.md the link points to)
METHODS: tuple[tuple[str, str, str], ...] = (
    (
        "Validity checks",
        "Before any outcome is compared, a chi-square test checks that the three arms have the "
        "sizes randomisation should give, and standardised mean differences check that the "
        "arms look alike on every pre-treatment covariate.",
        "Sample ratio mismatch (SRM) test",
    ),
    (
        "Frequentist readout",
        "Spend is compared with Welch's difference in means and rates with a two-proportion "
        "z-test and a Newcombe interval. Holm's correction covers the two primary comparisons "
        "and Benjamini-Hochberg the secondary ones.",
        "Frequentist readout",
    ),
    (
        "CUPED",
        "Past-year spend, measured before the experiment, is used to remove predictable noise "
        "from each outcome. The estimate stays unbiased and its interval narrows by as much as "
        "the covariate explains.",
        "CUPED (variance reduction with a pre-experiment covariate)",
    ),
    (
        "Regression adjustment",
        "Lin's estimator regresses the outcome on the arm, all pre-treatment covariates and "
        "their interactions, with robust standard errors. It captures what a single linear "
        "covariate misses.",
        "Regression adjustment (Lin 2013)",
    ),
    (
        "Bayesian readout",
        "Rates get a Beta-Binomial posterior and spend a Bayesian bootstrap. Each comparison "
        "reports the probability that the email is better and the expected loss of shipping "
        "it, and a pre-registered rule turns the two into a decision.",
        "Beta-Binomial Bayesian readout for rates",
    ),
    (
        "Power and sample size",
        "The minimum detectable effect says how small a lift the experiment could reliably "
        "find at its size, which puts every non-significant result in context.",
        "Minimum detectable effect, sample size and power",
    ),
    (
        "Peeking and sequential testing",
        "Simulated experiments show how often daily checking declares a false winner, and how "
        "Pocock, O'Brien-Fleming and always-valid p-values restore the error rate.",
        "Sequential testing: peeking and its valid fixes (simulation)",
    ),
    (
        "Segments",
        "Effects are estimated within pre-specified customer segments, with interaction tests "
        "for whether they differ and a false discovery correction over the whole family. All "
        "segment results are exploratory.",
        "Segment analysis (exploratory)",
    ),
    (
        "Uplift models",
        "S-, T- and X-learners predict how much an email changes each customer's chance of "
        "visiting, using pre-treatment features only. Every score comes from a model that "
        "never saw that customer.",
        "Uplift meta-learners: S, T and X",
    ),
    (
        "Targeting policies",
        "A policy says who gets which email. Its value is estimated from the experiment itself "
        "by inverse propensity weighting and compared with simpler rules by paired bootstrap.",
        "Targeting policies and their value (inverse propensity weighting)",
    ),
    (
        "Robustness to the data split",
        "The cross-fitting is repeated over several random splits to show how much the uplift "
        "results move with the split alone.",
        "Repeated cross-fitting over fold seeds",
    ),
)

st.markdown(
    "Each method below is explained in full, with its formula, assumptions and tests, in the "
    f"[methodology document]({ad.doc_url(METHODOLOGY)})."
)
for title, text, heading in METHODS:
    st.subheader(title)
    st.markdown(f"{text} [Read more]({ad.doc_url(METHODOLOGY, ad.heading_anchor(heading))})")

st.header("Documents")
st.markdown(
    f"- [Analysis plan]({ad.doc_url('docs/analysis_plan.md')}): written before any outcome "
    "was compared, with every later change listed under Deviations.\n"
    f"- [Methodology]({ad.doc_url(METHODOLOGY)}): every method, formula and test.\n"
    f"- [Results schema]({ad.doc_url('docs/results_schema.md')}): the structure of the results "
    "file this app reads.\n"
    f"- [Data card]({ad.doc_url('docs/data_card.md')}): the dataset and its known quirks.\n"
    f"- [Source code]({ad.REPO_URL})"
)
