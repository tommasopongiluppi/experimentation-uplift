"""Streamlit app: every page runs, shows the numbers in results.json, and the live simulation."""

import json
import re
import time

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from abkit import appdata as ad
from abkit import config, sequential
from abkit.power import mde, power_for_effect

APP = config.ROOT / "app"
PAGES = {
    "home": APP / "Home.py",
    "readout": APP / "pages" / "1_Readout.py",
    "peeking": APP / "pages" / "2_Peeking_simulator.py",
    "targeting": APP / "pages" / "3_Targeting.py",
    "method": APP / "pages" / "4_Method.py",
}


@pytest.fixture(scope="module")
def results():
    return json.loads(config.RESULTS_JSON.read_text())


def run_page(name: str) -> AppTest:
    at = AppTest.from_file(str(PAGES[name]), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def metric_values(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def page_text(at: AppTest) -> str:
    blocks = [*at.markdown, *at.caption, *at.success, *at.info, *at.warning, *at.error]
    return "\n".join(b.value for b in blocks)


# --- every page runs -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PAGES))
def test_every_page_runs_without_exceptions(name):
    at = run_page(name)
    assert len(at.title) == 1


def test_the_app_never_reads_the_raw_data():
    for path in [*PAGES.values(), config.ROOT / "src" / "abkit" / "appdata.py"]:
        source = path.read_text()
        for forbidden in ("duckdb", "DUCKDB_PATH", "DATA_DIR", "DATA_RAW", "read_parquet"):
            assert forbidden not in source, f"{path.name} mentions {forbidden}"


# --- Home ----------------------------------------------------------------------------------------


def test_home_shows_the_decision_and_three_headline_numbers(results):
    at = run_page("home")
    email = results["bayes_spend"]["recommendation"]["recommended_email"]
    freq = ad.find_row(
        results["frequentist"]["rows"], comparison=f"{email}_vs_control", metric="spend"
    )
    naive = ad.find_row(
        results["sequential"]["rows"], scenario="aa", metric="visit", method="naive"
    )
    values = metric_values(at)
    assert len(values) == 3
    assert values["Incremental spend per customer"] == ad.format_effect(freq["diff"], "spend")
    assert values["False positive rate of naive peeking"] == f"{100 * naive['reject_rate']:.1f}%"
    assert f"Send the {ad.ARM_LABELS[email]}" in at.success[0].value
    text = page_text(at)
    assert f"{100 * naive['reject_mcse']:.2f} pp" in text
    versus = results["policy"]["uplift_versus"]["visit"]
    assert not versus["response_model"]["meaningful"] and not versus["recency_rule"]["meaningful"]
    assert "is not separated from the response model or the recency rule" in at.info[0].value


# --- Readout -------------------------------------------------------------------------------------


def test_readout_headline_number_equals_results_json(results):
    at = run_page("readout")
    assert at.selectbox(key="comparison").value == "mens_vs_control"
    assert at.selectbox(key="metric").value == "spend"
    freq = ad.find_row(results["frequentist"]["rows"], comparison="mens_vs_control", metric="spend")
    shown = metric_values(at)["Effect (raw difference)"]
    assert shown == ad.format_effect(freq["diff"], "spend")
    # The string on screen parses back to the stored number, to the cent.
    assert float(re.sub(r"[^0-9.\-]", "", shown)) == pytest.approx(freq["diff"], abs=0.005)
    assert metric_values(at)["Adjusted p-value"] == ad.format_p(freq["p_adj"])


@pytest.mark.parametrize("comparison", list(ad.COMPARISON_LABELS))
@pytest.mark.parametrize("metric", list(ad.METRIC_LABELS))
def test_readout_shows_every_comparison_and_metric(results, comparison, metric):
    at = AppTest.from_file(str(PAGES["readout"]), default_timeout=60).run()
    at.selectbox(key="comparison").set_value(comparison)
    at.selectbox(key="metric").set_value(metric).run()
    assert not at.exception, [e.value for e in at.exception]
    freq = ad.find_row(results["frequentist"]["rows"], comparison=comparison, metric=metric)
    cuped = ad.find_row(results["cuped"]["rows"], comparison=comparison, metric=metric)
    values = metric_values(at)
    assert values["Effect (raw difference)"] == ad.format_effect(freq["diff"], metric)
    assert values["Adjusted p-value"] == ad.format_p(freq["p_adj"])
    table = at.dataframe[0].value
    assert list(table["Estimate"]) == ["Raw", "CUPED", "Regression-adjusted"]
    assert list(table["Effect"]) == [
        ad.format_effect(v, metric) for v in (freq["diff"], cuped["cuped_diff"], cuped["lin_diff"])
    ]
    if metric in ad.RATE_METRICS:
        bayes = ad.find_row(
            results["bayes_rates"]["rows"], comparison=comparison, metric=metric, prior="flat"
        )
    else:
        bayes = ad.find_row(
            results["bayes_spend"]["rows"],
            comparison=comparison,
            metric=metric,
            model="bayesian_bootstrap",
        )
    text = page_text(at)
    assert f"Decision: {bayes['decision']}." in text and bayes["reason"] in text
    assert "Sample ratio" in text and "Covariate balance" in text


# --- Targeting -----------------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [0, 10_000, 37_000, 64_000])
def test_targeting_slider_reads_the_budget_curve(results, budget):
    at = AppTest.from_file(str(PAGES["targeting"]), default_timeout=60).run()
    assert at.select_slider(key="budget").value == results["policy"]["budget"]
    at.select_slider(key="budget").set_value(budget).run()
    assert not at.exception, [e.value for e in at.exception]
    curve = ad.load_table(results["policy"]["budget_curve_table"])
    rows = curve[curve["n_emailed"] == budget].set_index("policy")
    table = at.dataframe[0].value.set_index("Policy")
    assert len(table) == rows.index.nunique() >= 3
    for policy in rows.index:
        label = ad.POLICY_LABELS[policy]
        assert (
            table.loc[label, "Incremental visits"]
            == f"{rows.loc[policy, 'incremental_visits']:,.0f}"
        )
        assert (
            table.loc[label, "Incremental spend"]
            == f"${rows.loc[policy, 'incremental_spend']:,.0f}"
        )


def test_targeting_shows_the_recency_rule_and_the_fixed_profile(results):
    at = run_page("targeting")
    fixed = at.dataframe[1].value.set_index("Policy")
    row = results["policy"]["headline"]["visit"]["recency_rule"]
    assert fixed.loc["Recency rule", "Incremental visits (95% interval)"] == (
        f"{row['incremental_total']:,.0f} ({row['ci_low']:,.0f} to {row['ci_high']:,.0f})"
    )
    profile = at.dataframe[2].value
    assert len(profile) == 16 and "-" not in set(profile.iloc[:, 1])
    assert f"fixed at the {results['policy']['budget']:,} budget" in at.header[-1].value


# --- Method --------------------------------------------------------------------------------------


def test_method_links_point_at_headings_that_exist():
    at = run_page("method")
    methodology = (config.ROOT / "docs" / "methodology.md").read_text()
    anchors = {
        ad.heading_anchor(line[3:].strip())
        for line in methodology.splitlines()
        if line.startswith("## ")
    }
    links = re.findall(r"methodology\.md#([^)]+)\)", "\n".join(m.value for m in at.markdown))
    assert len(links) == 11 and set(links) <= anchors
    for doc in re.findall(r"/blob/main/([\w/.]+)", "\n".join(m.value for m in at.markdown)):
        assert (config.ROOT / doc).is_file(), doc


def test_heading_anchor_follows_the_github_rule():
    assert ad.heading_anchor("Regression adjustment (Lin 2013)") == "regression-adjustment-lin-2013"
    assert ad.heading_anchor("Uplift meta-learners: S, T and X") == "uplift-meta-learners-s-t-and-x"
    assert ad.doc_url("docs/x.md", "a-b") == f"{ad.REPO_URL}/blob/main/docs/x.md#a-b"


# --- formatting and lookups ----------------------------------------------------------------------


def test_formatters():
    assert ad.format_effect(0.7698, "spend") == "+$0.77"
    assert ad.format_effect(-1234.5, "spend") == "-$1,234.50"
    assert ad.format_effect(0.0766, "visit") == "+7.66 pp"
    assert ad.format_effect(0.0766, "visit", signed=False) == "7.66 pp"
    assert ad.format_level(0.1062, "visit") == "10.62%"
    assert ad.format_level(0.65, "spend") == "$0.65"
    assert ad.format_p(0.0312) == "0.031" and ad.format_p(2.3e-7) == "2.3e-07"
    assert ad.format_probability(1.0) == "> 99.9%" and ad.format_probability(0.0) == "< 0.1%"
    assert ad.format_probability(0.985175) == "98.5%"
    assert ad.format_small(0.00653, "spend") == "$0.0065"
    assert ad.format_small(8.84e-05, "conversion") == "0.0088 pp"
    assert ad.display_scale("visit") == (100.0, "percentage points")
    assert not ad.is_finite(None) and not ad.is_finite(float("nan")) and ad.is_finite(0)


def test_find_row_refuses_missing_and_ambiguous_rows():
    rows = [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}]
    assert ad.find_row(rows, a=1, b="y") == {"a": 1, "b": "y"}
    with pytest.raises(LookupError, match="found 2"):
        ad.find_row(rows, a=1)
    with pytest.raises(LookupError, match="found 0"):
        ad.find_row(rows, a=2)


# --- the live simulation -------------------------------------------------------------------------


def test_live_simulation_fixed_horizon_keeps_alpha_and_naive_peeking_does_not(rng):
    fixed = ad.simulate_peeking(0.10, 0.0, 3000, 14, "fixed", rng, n_runs=20_000)
    naive = ad.simulate_peeking(0.10, 0.0, 3000, 14, "naive", rng, n_runs=20_000)
    assert fixed.is_null and fixed.n_per_arm_per_day == 1500
    assert abs(fixed.reject_rate - config.ALPHA) < 3 * fixed.reject_mcse
    assert naive.reject_rate - 5 * naive.reject_mcse > config.ALPHA
    # A fixed-horizon test can only stop on the last day.
    assert fixed.stop_day_share[:-1].sum() == 0
    assert fixed.stop_day_share[-1] == pytest.approx(fixed.reject_rate)
    assert fixed.mean_stop_day == 14


def test_live_simulation_power_matches_the_closed_form(rng):
    p, lift, per_day, days = 0.10, 0.08, 1500, 14
    result = ad.simulate_peeking(p, lift, 2 * per_day, days, "fixed", rng, n_runs=20_000)
    q = p * (1 + lift)
    # Unpooled variance of the difference, expressed through the two-arm power formula.
    expected = power_for_effect(q - p, (p * (1 - p) + q * (1 - q)) / 2, per_day * days)
    assert not result.is_null and result.rate_treatment == pytest.approx(q)
    assert abs(result.reject_rate - expected) < 4 * result.reject_mcse


@pytest.mark.parametrize("method", list(ad.PEEKING_METHODS))
def test_live_simulation_matches_the_pipeline_rules(method):
    seed, n_runs, days, per_day, p = 7, 1500, 10, 800, 0.2
    result = ad.simulate_peeking(
        p, 0.0, 2 * per_day, days, method, np.random.default_rng(seed), n_runs=n_runs
    )
    rng_paths, rng_boundaries = np.random.default_rng(seed).spawn(2)
    paths = sequential.simulate_rate_paths(p, p, per_day * days, days, n_runs, rng_paths)
    constants = sequential.calibrate_boundaries(days, ad.LIVE_PATHS, rng_boundaries)
    tau = mde(p, p * (1 - p), per_day * days).absolute
    # The pipeline route: every rule at once, the Bayesian one through its count boundary.
    reference = sequential.rejections(paths, constants, tau)[method]
    summary = sequential.summarise_rule(reference)
    assert result.reject_rate == summary["reject_rate"]
    assert result.mean_stop_day == summary["mean_stop_day"]
    table = ad.stop_day_table(result)
    assert len(table) == days + 1 and table["share"].sum() == pytest.approx(1.0)
    assert table["share"].iloc[-1] == pytest.approx(1 - result.reject_rate)
    assert f"{n_runs:,} simulated experiments" in ad.monte_carlo_note(result)


def test_live_simulation_input_errors(rng):
    for args in [
        (0.0, 0.0, 1000, 14, "naive"),
        (0.6, 1.0, 1000, 14, "naive"),
        (0.1, 0.0, 1, 14, "naive"),
        (0.1, 0.0, 1000, ad.LIVE_MAX_DAYS + 1, "naive"),
        (0.1, 0.0, 1000, 14, "coin_flip"),
    ]:
        with pytest.raises(ValueError):
            ad.simulate_peeking(*args, rng)


def test_live_runs_are_capped_for_the_bayesian_rule():
    assert ad.live_runs("naive", 30) == ad.LIVE_RUNS
    assert ad.live_runs("bayes", 30) * 30 <= ad.LIVE_BAYES_EVALUATIONS
    assert ad.live_runs("bayes", 2) == ad.LIVE_RUNS


def test_peeking_page_reports_the_simulated_rate():
    at = AppTest.from_file(str(PAGES["peeking"]), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="method").value == "naive"
    expected = ad.cached_peeking(
        at.number_input(key="rate").value / 100,
        0.0,
        int(at.number_input(key="users").value),
        int(at.number_input(key="days").value),
        "naive",
    )
    assert metric_values(at)["False positive rate"] == f"{100 * expected.reject_rate:.1f}%"
    assert expected.reject_rate - 5 * expected.reject_mcse > config.ALPHA
    assert "Above the nominal" in at.error[0].value
    at.number_input(key="lift").set_value(10.0)
    at.selectbox(key="method").set_value("obf").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Power" in metric_values(at)


@pytest.mark.parametrize("method", list(ad.PEEKING_METHODS))
def test_live_simulation_answers_quickly_at_the_largest_inputs(method):
    start = time.perf_counter()
    ad.cached_peeking(0.5, 0.0, ad.LIVE_MAX_USERS_PER_DAY, ad.LIVE_MAX_DAYS, method)
    assert time.perf_counter() - start < 10  # about 2 s on the dev machine; generous for CI
