"""Segment analysis: interaction tests vs statsmodels, BH families and the README simulation."""

import json
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from abkit import config
from abkit.frequentist import benjamini_hochberg, diff_in_means, diff_in_proportions
from abkit.metrics import run_sql
from abkit.segments import (
    COMPARISONS,
    FDR_Q,
    LABEL,
    N_SIM_SEGMENTS,
    SEGMENT_OUTCOMES,
    SEGMENT_VARIABLES,
    bh_rows,
    build_segment_tables,
    challenge_answer,
    interaction_test,
    interaction_tests,
    ols_hc2,
    recency_band,
    run_segments,
    segment_column,
    segment_effects,
    simulate_segment_discoveries,
)
from tests.conftest import customers_con

N_LEVELS = 23


# --- the plan's segment list ----------------------------------------------------------------------


def test_segment_variables_are_the_plan_list():
    assert list(SEGMENT_VARIABLES) == [
        "newbie",
        "channel",
        "zip_code",
        "mens",
        "womens",
        "recency",
        "history_segment",
    ]
    assert sum(len(v) for v in SEGMENT_VARIABLES.values()) == N_LEVELS
    assert set(SEGMENT_VARIABLES) <= set(config.PRE_TREATMENT_COLS)
    assert set(SEGMENT_OUTCOMES).isdisjoint(SEGMENT_VARIABLES)
    assert all(c == config.CONTROL for _, c in COMPARISONS)


def test_recency_band():
    r = np.arange(1, 13)
    expected = ["1_3"] * 3 + ["4_6"] * 3 + ["7_9"] * 3 + ["10_12"] * 3
    assert list(recency_band(r)) == expected
    with pytest.raises(ValueError):
        recency_band([0, 5])


def test_segment_columns_match_the_sql_counts(customers):
    con = customers_con(customers)
    seg_stats = run_sql(con, "segment_stats").set_index(["variable", "level", "arm"])
    for variable, levels in SEGMENT_VARIABLES.items():
        col = segment_column(customers, variable)
        assert set(col.unique()) <= set(levels)
        for level in levels:
            for arm in config.ARMS:
                n = int(((col == level) & (customers["arm"] == arm)).sum())
                key = (variable, level, arm)
                assert (int(seg_stats.loc[key, "n"]) if key in seg_stats.index else 0) == n
    with pytest.raises(ValueError):
        segment_column(customers, "visit")


# --- OLS with HC2 and the joint interaction test vs statsmodels -----------------------------------


def _heterogeneous_sample(rng, n=4_000, interaction=0.3):
    seg = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    d = rng.integers(0, 2, size=n)
    shift = {"a": 0.0, "b": interaction, "c": -interaction}
    noise = rng.normal(0, 1 + (seg == "c"), size=n)  # heteroskedastic
    y = 1.0 + 0.5 * d + 0.2 * (seg == "b") + d * np.vectorize(shift.get)(seg) + noise
    return pd.DataFrame({"y": y, "d": d, "seg": seg})


def test_ols_hc2_matches_statsmodels(rng):
    df = _heterogeneous_sample(rng)
    x = np.column_stack([np.ones(len(df)), df["d"], df["seg"] == "b", df["seg"] == "c"]).astype(
        float
    )
    beta, cov = ols_hc2(x, df["y"].to_numpy())
    ref = smf.ols("y ~ d + C(seg)", df).fit(cov_type="HC2")
    names = ["Intercept", "d", "C(seg)[T.b]", "C(seg)[T.c]"]
    np.testing.assert_allclose(beta, ref.params[names].to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(cov, ref.cov_params().loc[names, names].to_numpy(), rtol=1e-8)
    with pytest.raises(ValueError):
        ols_hc2(np.column_stack([x, x[:, 1]]), df["y"].to_numpy())


def test_interaction_test_matches_statsmodels_formula(rng):
    df = _heterogeneous_sample(rng, interaction=0.6)
    res = interaction_test(df["y"], df["d"], df["seg"])
    ref = smf.ols("y ~ d * C(seg)", df).fit(cov_type="HC2")
    names = [n for n in ref.params.index if ":" in n]
    r = np.zeros((len(names), len(ref.params)))
    for i, name in enumerate(names):
        r[i, ref.params.index.get_loc(name)] = 1.0
    wald = ref.wald_test(r, use_f=False, scalar=True)
    assert res.dof == len(names) == 2
    assert res.statistic == pytest.approx(float(wald.statistic), rel=1e-8)
    assert res.p_value == pytest.approx(float(wald.pvalue), rel=1e-6)
    assert res.p_value == pytest.approx(stats.chi2.sf(res.statistic, res.dof))
    assert res.effects["a"][0] == pytest.approx(ref.params["d"], rel=1e-10)
    assert res.effects["a"][1] == pytest.approx(ref.bse["d"], rel=1e-8)
    assert res.p_value < 1e-6  # the planted heterogeneity is found
    assert res.n == len(df) and res.n_levels == 3


def test_interaction_test_within_level_effects_are_the_raw_differences(rng):
    df = _heterogeneous_sample(rng)
    res = interaction_test(df["y"], df["d"], df["seg"])
    for level, (effect, se) in res.effects.items():
        part = df[df["seg"] == level]
        raw = part.loc[part["d"] == 1, "y"].mean() - part.loc[part["d"] == 0, "y"].mean()
        assert effect == pytest.approx(raw, rel=1e-10)
        assert se > 0


def test_interaction_test_binary_segment_equals_the_squared_contrast(rng):
    df = _heterogeneous_sample(rng)
    df = df[df["seg"] != "c"]
    res = interaction_test(df["y"], df["d"], df["seg"])
    ref = smf.ols("y ~ d * C(seg)", df).fit(cov_type="HC2")
    name = next(n for n in ref.params.index if ":" in n)
    assert res.dof == 1
    assert res.statistic == pytest.approx((ref.params[name] / ref.bse[name]) ** 2, rel=1e-8)


def test_interaction_test_input_errors(rng):
    df = _heterogeneous_sample(rng, n=300)
    with pytest.raises(ValueError):
        interaction_test(df["y"], df["d"], np.full(len(df), "a"))
    with pytest.raises(ValueError):
        interaction_test(df["y"], np.ones(len(df)), df["seg"])
    missing = df.copy()
    missing = missing[~((missing["seg"] == "c") & (missing["d"] == 1))]
    with pytest.raises(ValueError, match="missing"):
        interaction_test(missing["y"], missing["d"], missing["seg"])


# --- tables on the fixture ---------------------------------------------------------------------


def test_segment_effects_match_the_readout_tests(customers):
    con = customers_con(customers)
    effects = segment_effects(run_sql(con, "segment_stats"))
    assert len(effects) == len(SEGMENT_OUTCOMES) * N_LEVELS * len(COMPARISONS)
    assert (effects["label"] == LABEL).all()
    assert effects["p_value"].between(0, 1).all()
    # The fixture's top history band has zero spend in both arms: p = 1, point interval.
    constant = effects[(effects["outcome"] == "spend") & (effects["level"] == "usd_1000_plus")]
    assert (constant["p_value"] == 1).all() and (constant["ci_low"] == constant["diff"]).all()
    for outcome, variable, level, treatment in (
        ("visit", "channel", "web", "mens"),
        ("conversion", "womens", "1", "womens"),
        ("spend", "recency", "4_6", "mens"),
    ):
        row = effects[
            (effects["outcome"] == outcome)
            & (effects["variable"] == variable)
            & (effects["level"] == level)
            & (effects["treatment"] == treatment)
        ].iloc[0]
        col = segment_column(customers, variable)
        yt = customers.loc[(col == level) & (customers["arm"] == treatment), outcome]
        yc = customers.loc[(col == level) & (customers["arm"] == config.CONTROL), outcome]
        if outcome == "spend":
            ref = diff_in_means(
                yt.mean(), yt.var(ddof=1), len(yt), yc.mean(), yc.var(ddof=1), len(yc)
            )
        else:
            ref = diff_in_proportions(yt.sum(), len(yt), yc.sum(), len(yc))
        assert row["n_treatment"] == len(yt) and row["n_control"] == len(yc)
        assert row["diff"] == pytest.approx(ref.diff, rel=1e-10)
        assert row["ci_low"] == pytest.approx(ref.ci_low, rel=1e-10)
        assert row["p_value"] == pytest.approx(ref.p_value, rel=1e-10)


def test_families_are_bh_per_outcome_over_levels_and_interactions(customers):
    con = customers_con(customers)
    effects, interactions = build_segment_tables(con)
    assert len(interactions) == len(SEGMENT_OUTCOMES) * len(SEGMENT_VARIABLES) * len(COMPARISONS)
    for outcome in SEGMENT_OUTCOMES:
        family = f"segments_{outcome}"
        e = effects[effects["family"] == family]
        i = interactions[interactions["family"] == family]
        assert len(e) == 46 and len(i) == 14
        assert (e["family_size"] == 60).all() and (i["family_size"] == 60).all()
        p = np.concatenate([e["p_value"], i["p_value"]])
        adj = benjamini_hochberg(p)
        np.testing.assert_allclose(np.concatenate([e["p_adj"], i["p_adj"]]), adj)
        assert (e["p_adj"] >= e["p_value"]).all()
        assert (e["significant_fdr"] == (e["p_adj"] <= FDR_Q)).all()
    assert (effects["adjustment"] == "bh").all() and (interactions["label"] == LABEL).all()
    answer = challenge_answer(effects, interactions)
    assert len(answer) == 2 * len(SEGMENT_OUTCOMES)
    for rec in answer:
        assert rec["email"] == rec["segment_variable"] and rec["label"] == LABEL
        assert rec["larger_among_past_buyers"] == (rec["past_buyers_diff"] > rec["non_buyers_diff"])


def test_interaction_tests_use_only_the_two_arms(customers):
    rows = run_sql(customers_con(customers), "adjustment_rows")
    table = interaction_tests(rows)
    two_arms = rows["arm"].isin(("mens", config.CONTROL)).sum()
    assert (table.loc[table["treatment"] == "mens", "n"] == two_arms).all()
    assert (table["dof"] == table["n_levels"] - 1).all()


# --- the README simulation ---------------------------------------------------------------------


def test_bh_rows_matches_benjamini_hochberg(rng):
    p = rng.uniform(size=(50, 20)) ** 2
    out = bh_rows(p)
    for i in range(p.shape[0]):
        np.testing.assert_allclose(out[i], benjamini_hochberg(p[i]))


def test_twenty_segments_under_the_null_reproduce_one_minus_095_to_the_20():
    rng = np.random.default_rng(config.SEEDS["fixture"] + 10)
    rows = {r["method"]: r for r in simulate_segment_discoveries(20, 1_000, 0.1, rng, n_runs=4_000)}
    raw, bh = rows["raw"], rows["bh"]
    expected = 1 - 0.95**20
    assert raw["any_false_discovery_reference"] == pytest.approx(expected)
    assert abs(raw["any_false_discovery"] - expected) <= 3 * raw["any_false_discovery_mcse"]
    assert abs(bh["any_false_discovery"] - FDR_Q) <= 3 * bh["any_false_discovery_mcse"]
    assert bh["fdr"] == bh["any_false_discovery"]  # under the global null they coincide
    assert math.isnan(raw["power"])


def test_bh_controls_the_false_discovery_rate_with_real_effects():
    rng = np.random.default_rng(config.SEEDS["fixture"] + 11)
    n, p = 1_000, 0.1
    effect = 2.8 * math.sqrt(2 * p * (1 - p) / n)
    rows = {
        r["method"]: r
        for r in simulate_segment_discoveries(20, n, p, rng, n_runs=4_000, n_true=5, effect=effect)
    }
    raw, bh = rows["raw"], rows["bh"]
    assert bh["fdr_reference"] == pytest.approx(FDR_Q * 15 / 20)
    assert abs(bh["fdr"] - bh["fdr_reference"]) <= 3 * bh["fdr_mcse"]
    assert raw["fdr"] > bh["fdr"] + 3 * raw["fdr_mcse"]
    assert 0.7 < raw["power"] < 0.85 and 0 < bh["power"] < raw["power"]
    with pytest.raises(ValueError):
        simulate_segment_discoveries(20, n, p, rng, n_runs=10, n_true=21)


# --- end to end on the fixture -----------------------------------------------------------------


def test_run_segments_writes_outputs(fixture_db, tmp_path):
    effects = run_segments(fixture_db, tmp_path, n_sim_runs=200)
    assert len(effects) == len(SEGMENT_OUTCOMES) * N_LEVELS * len(COMPARISONS)
    for name in ("segments", "segments_interactions", "segments_fdr_simulation"):
        table = pd.read_csv(tmp_path / "tables" / f"{name}.csv")
        assert len(table) > 0
        if name != "segments_fdr_simulation":
            assert (table["label"] == LABEL).all()
    assert (tmp_path / "figures" / "forest_segments.png").stat().st_size > 0
    text = (tmp_path / "results.json").read_text()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite value {c} in results"))
    section = json.loads(text)["segments"]
    assert section["label"] == LABEL and "D5" in section["deviation"]
    assert section["n_levels"] == N_LEVELS
    assert all(f["n_tests"] == 60 for f in section["families"].values())
    assert set(section["families"]) == {f"segments_{o}" for o in SEGMENT_OUTCOMES}
    assert len(section["challenge_question"]) == 2 * len(SEGMENT_OUTCOMES)
    assert section["fdr_simulation"]["n_segments"] == N_SIM_SEGMENTS
    assert len(section["fdr_simulation"]["rows"]) == 4
    assert len(section["rows"]) == len(effects) and len(section["interaction_rows"]) == 42
