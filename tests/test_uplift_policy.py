"""Targeting policies: IPW known answer, budget constraints, comparisons and the stage."""

import json

import numpy as np
import pandas as pd
import pytest

from abkit import config
from abkit.frequentist import diff_in_proportions
from abkit.uplift.models import SCORES_NAME
from abkit.uplift.policy import (
    BUDGETED,
    EXPLORATORY,
    LABEL,
    POLICIES,
    PolicyInputs,
    ab_winner,
    budget_curve,
    build_policies,
    calibration_check,
    email_counts,
    evaluate_policies,
    mens_constant,
    policy_value,
    profiles,
    run_policy,
    top_k,
)

ARMS = np.array(config.ARMS)


def _inputs(rng, n, tau_scale=0.1):
    """Random arms, outcomes and scores with the cleaned-schema pre-treatment columns."""
    arm = rng.choice(ARMS, n)
    tau = pd.DataFrame({e: tau_scale * rng.standard_normal(n) for e in config.TREATMENTS})
    response = pd.DataFrame({e: rng.random(n) for e in config.TREATMENTS})
    visit = (rng.random(n) < 0.15).astype(float)
    conversion = visit * (rng.random(n) < 0.1)
    spend = conversion * rng.gamma(2.0, 50.0, n)
    pre = pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history_segment": "usd_0_100",
            "history": rng.gamma(2.0, 100.0, n),
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["urban", "suburban", "rural"], n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["phone", "web", "multichannel"], n),
        }
    )
    return PolicyInputs(
        customer_id=np.arange(1, n + 1),
        arm=arm.astype(str),
        y=pd.DataFrame({"visit": visit, "conversion": conversion, "spend": spend}),
        tau=tau,
        response=response,
        pre=pre,
    )


# --- IPW known answer ----------------------------------------------------------------------------


def test_ipw_recovers_the_known_value_of_a_fixed_policy(rng):
    """Potential outcomes with known means; the policy's true value is a closed form."""
    n = 90_000
    x = rng.random(n)
    arm = rng.choice(ARMS, n)  # P = 1/3 each
    p_control = np.full(n, 0.10)
    p_mens = 0.20 + 0.20 * (x < 0.5)
    p_womens = np.full(n, 0.15)
    rates = {"control": p_control, "mens": p_mens, "womens": p_womens}
    y = (rng.random(n) < np.select([arm == a for a in ARMS], [rates[a] for a in ARMS])).astype(
        float
    )
    policy = np.where(x < 0.3, "mens", np.where(x < 0.6, "womens", "control"))
    # True value: 0.3 * 0.40 + 0.3 * 0.15 + 0.4 * 0.10
    truth = 0.3 * 0.40 + 0.3 * 0.15 + 0.4 * 0.10
    ipw, sn = policy_value(policy, arm, y)
    se = np.sqrt(3 * truth / n)  # rough SE of the IPW mean: var(y * 3 * match) ~ 3 * truth
    assert abs(ipw[0] - truth) < 4 * se, (ipw, truth, se)
    assert abs(sn[0] - truth) < 4 * se, (sn, truth, se)
    nobody, _ = policy_value(np.full(n, "control"), arm, y)
    assert abs(nobody[0] - 0.10) < 4 * np.sqrt(3 * 0.1 / n)


def test_policy_value_matrix_and_weights(rng):
    inputs = _inputs(rng, 2_000)
    assignment = np.full(inputs.n, "mens")
    y = inputs.y.to_numpy()
    ipw, sn = policy_value(assignment, inputs.arm, y)
    assert ipw.shape == sn.shape == (3,)
    ipw_w, sn_w = policy_value(assignment, inputs.arm, y, weights=np.ones(inputs.n))
    np.testing.assert_allclose(ipw, ipw_w)
    np.testing.assert_allclose(sn, sn_w)
    # Self-normalised value of "everyone gets mens" is exactly the mens arm's mean outcome.
    np.testing.assert_allclose(sn, y[inputs.arm == "mens"].mean(axis=0))
    # IPW rescales by the realised share of the mens arm against the design probability.
    share = (inputs.arm == "mens").mean()
    np.testing.assert_allclose(ipw, sn * share / config.ARM_PROBABILITY)


# --- policies and the budget -------------------------------------------------------------------


def test_top_k_and_email_counts():
    mask = top_k(np.array([0.1, 0.9, 0.5, 0.9]), 2)
    np.testing.assert_array_equal(mask, [False, True, False, True])
    assert not top_k(np.array([1.0, 2.0]), 0).any()
    counts = email_counts(np.array(["mens", "control", "womens", "mens"]))
    assert counts == {"n_emailed": 3, "n_mens": 2, "n_womens": 1}


def test_every_policy_respects_the_budget(rng):
    n, budget = 5_000, 800
    inputs = _inputs(rng, n)
    assignments, info = build_policies(inputs, "mens", rng, budget)
    assert set(assignments) == set(POLICIES)
    for name, a in assignments.items():
        assert a.shape == (n,) and set(a) <= set(config.ARMS), name
    counts = {name: email_counts(a) for name, a in assignments.items()}
    for name in BUDGETED:
        assert counts[name]["n_emailed"] <= budget, name
    for name in ("random", "recency_rule", "response_model"):
        assert counts[name]["n_emailed"] == budget, name
        assert counts[name]["n_womens" if name != "response_model" else "n_emailed"] >= 0
    assert counts["random"]["n_womens"] == 0 and counts["recency_rule"]["n_womens"] == 0
    assert counts["nobody"]["n_emailed"] == 0
    assert counts["all_mens"] == {"n_emailed": n, "n_mens": n, "n_womens": 0}
    assert counts["all_womens"] == {"n_emailed": n, "n_mens": 0, "n_womens": n}
    assert counts["suppression"]["n_emailed"] == n - budget
    assert info["suppressed"].sum() == budget and info["targeted"].sum() <= budget


def test_uplift_policy_targets_highest_positive_uplift_with_the_better_email(rng):
    inputs = _inputs(rng, 3_000)
    assignments, info = build_policies(inputs, "mens", rng, 500)
    tau = inputs.tau[list(config.TREATMENTS)].to_numpy()
    best, best_tau = np.array(config.TREATMENTS)[tau.argmax(axis=1)], tau.max(axis=1)
    emailed = assignments["uplift"] != config.CONTROL
    assert emailed.sum() == 500
    np.testing.assert_array_equal(assignments["uplift"][emailed], best[emailed])
    assert best_tau[emailed].min() > best_tau[~emailed].max()
    assert best_tau[emailed].min() > 0
    # Suppression withholds exactly the 500 lowest best uplifts and emails the rest their best.
    held = assignments["suppression"] == config.CONTROL
    assert held.sum() == 500 and best_tau[held].max() < best_tau[~held].min()
    np.testing.assert_array_equal(assignments["suppression"][~held], best[~held])
    np.testing.assert_array_equal(info["best_email"], best)


def test_uplift_policy_skips_non_positive_uplift(rng):
    inputs = _inputs(rng, 1_000)
    inputs.tau.loc[:, :] = -0.01  # nobody has positive predicted uplift
    assignments, _ = build_policies(inputs, "mens", rng, 300)
    assert email_counts(assignments["uplift"])["n_emailed"] == 0
    assert email_counts(assignments["uplift_mens_default"])["n_emailed"] <= 300


def test_mens_by_default_policy_uses_the_constant(rng):
    inputs = _inputs(rng, 3_000)
    const = mens_constant(inputs)
    y = inputs.y["visit"].to_numpy()
    assert const == pytest.approx(
        y[inputs.arm == "mens"].mean() - y[inputs.arm == "control"].mean()
    )
    assignments, _ = build_policies(inputs, "mens", rng, 400)
    a = assignments["uplift_mens_default"]
    womens_tau = inputs.tau["womens"].to_numpy()
    assert (womens_tau[a == "womens"] > const).all()
    assert (womens_tau[a == "mens"] <= const).all()
    chosen = np.maximum(womens_tau, const)
    emailed = a != config.CONTROL
    assert chosen[emailed].min() >= chosen[~emailed].max()


def test_response_model_policy_uses_the_higher_probability_email(rng):
    inputs = _inputs(rng, 2_000)
    assignments, _ = build_policies(inputs, "womens", rng, 300)
    a = assignments["response_model"]
    resp = inputs.response[list(config.TREATMENTS)].to_numpy()
    emailed = a != config.CONTROL
    np.testing.assert_array_equal(
        a[emailed], np.array(config.TREATMENTS)[resp.argmax(axis=1)][emailed]
    )
    assert resp.max(axis=1)[emailed].min() >= resp.max(axis=1)[~emailed].max()


def test_random_policy_is_seeded(rng):
    inputs = _inputs(rng, 1_000)
    a, _ = build_policies(inputs, "mens", np.random.default_rng(3), 100)
    b, _ = build_policies(inputs, "mens", np.random.default_rng(3), 100)
    c, _ = build_policies(inputs, "mens", np.random.default_rng(4), 100)
    np.testing.assert_array_equal(a["random"], b["random"])
    assert not np.array_equal(a["random"], c["random"])


# --- evaluation ----------------------------------------------------------------------------------


def test_evaluate_policies_tables(rng):
    inputs = _inputs(rng, 3_000)
    assignments, info = build_policies(inputs, "mens", rng, 500)
    table, comparison, suppressed = evaluate_policies(
        inputs,
        assignments,
        info["suppressed"],
        info["best_email"],
        "mens",
        rng,
        n_resamples=40,
        cost_per_email=0.5,
    )
    assert len(table) == len(POLICIES) * 3
    t = table.set_index(["policy", "outcome"])
    for outcome in config.OUTCOME_COLS:
        assert t.loc[("nobody", outcome), "incremental_total"] == 0
        assert t.loc[("all_mens", outcome), "share_of_all_winner"] == pytest.approx(1.0)
    assert (table["ci_low"] <= table["incremental_total"]).all()
    assert (table["incremental_total"] <= table["ci_high"]).all()
    spend = table[table["outcome"] == "spend"].set_index("policy")
    np.testing.assert_allclose(
        spend["net_value"], spend["incremental_total"] - 0.5 * spend["n_emailed"]
    )
    assert table.loc[table["outcome"] == "visit", "net_value"].isna().all()
    assert (table.loc[table["policy"].isin(EXPLORATORY), "exploratory"]).all()
    assert set(comparison["reference"]) == {"uplift", "recency_rule"}
    assert len(comparison) == 2 * (len(POLICIES) - 1) * 3
    assert comparison["meaningful"].dtype == bool
    assert len(suppressed) == 3 and (suppressed["n_suppressed"] == 500).all()


def test_paired_comparison_is_exact_for_identical_policies(rng):
    inputs = _inputs(rng, 2_000)
    assignments, info = build_policies(inputs, "mens", rng, 300)
    assignments["uplift_copy"] = assignments["uplift"].copy()
    _, comparison, _ = evaluate_policies(
        inputs, assignments, info["suppressed"], info["best_email"], "mens", rng, n_resamples=30
    )
    rows = comparison[
        (comparison["reference"] == "uplift") & (comparison["policy"] == "uplift_copy")
    ]
    # Identical policies differ only by floating-point summation order (values are O(1e3)).
    np.testing.assert_allclose(rows["diff_incremental_total"], 0, atol=1e-6)
    np.testing.assert_allclose(rows[["ci_low", "ci_high"]], 0, atol=1e-6)
    assert not rows["meaningful"].any()


def test_budget_curve_at_the_budget_matches_the_policy_table(rng):
    """One definition per policy: the curve at the fixed budget is the policy table's value."""
    inputs = _inputs(rng, 2_000)
    budget = 500
    assignments, info = build_policies(inputs, "womens", rng, budget)
    table, _, _ = evaluate_policies(
        inputs, assignments, info["suppressed"], info["best_email"], "womens", rng, n_resamples=5
    )
    curve = budget_curve(inputs, info, "womens", step=budget)
    assert set(curve["policy"]) == {"uplift", "response_model", "recency_rule", "random"}
    for name in ("uplift", "response_model", "recency_rule", "random"):
        assert email_counts(assignments[name])["n_emailed"] == budget
        at_budget = curve[(curve["policy"] == name) & (curve["n_emailed"] == budget)].iloc[0]
        for outcome, column in (("visit", "incremental_visits"), ("spend", "incremental_spend")):
            expected = table[(table["policy"] == name) & (table["outcome"] == outcome)]
            assert at_budget[column] == pytest.approx(expected["incremental_total"].iloc[0])
    # The random ranking is the random policy's own permutation: its first `budget` customers
    # are exactly the ones the policy emails, and smaller budgets are nested inside them.
    order = np.argsort(-info["random_score"], kind="stable")
    assert set(order[:budget]) == set(np.flatnonzero(assignments["random"] == "womens"))
    assert len(set(info["random_score"])) == inputs.n


def test_budget_curve_endpoints(rng):
    inputs = _inputs(rng, 2_000)
    _, info = build_policies(inputs, "womens", rng, 300)
    curve = budget_curve(inputs, info, "womens", step=500, cost_per_email=0.1)
    assert set(curve["policy"]) == {"uplift", "response_model", "recency_rule", "random"}
    # At the fixed budget each curve reproduces the policy of the same name: the recency rule
    # emails exactly the customers build_policies gave it, and the first k of its ranking are
    # the k most recent purchasers.
    y = inputs.y[["visit", "spend"]].to_numpy()
    nobody, _ = policy_value(np.full(inputs.n, "control"), inputs.arm, y)
    order = np.argsort(-info["recency_score"], kind="stable")
    recency = inputs.pre["recency"].to_numpy()
    for k in (500, 1_000, 1_500):
        assert recency[order[:k]].max() <= recency[order[k:]].min()
        assignment = np.full(inputs.n, "control", dtype=object)
        assignment[order[:k]] = "womens"
        value, _ = policy_value(assignment.astype(str), inputs.arm, y)
        row = curve[(curve["policy"] == "recency_rule") & (curve["n_emailed"] == k)].iloc[0]
        assert row["incremental_visits"] == pytest.approx(inputs.n * (value - nobody)[0])
        assert row["incremental_spend"] == pytest.approx(inputs.n * (value - nobody)[1])
    # Every ranking ends at the same point only when it sends one email to everyone.
    ends = curve[curve["n_emailed"] == inputs.n].set_index("policy")["incremental_visits"]
    assert ends["recency_rule"] == pytest.approx(ends["random"])
    first = curve[curve["n_emailed"] == 0]
    assert (first[["incremental_visits", "incremental_spend"]] == 0).all().all()
    last = curve[(curve["policy"] == "random") & (curve["n_emailed"] == inputs.n)].iloc[0]
    nobody, _ = policy_value(
        np.full(inputs.n, "control"), inputs.arm, inputs.y[["visit", "spend"]].to_numpy()
    )
    everyone, _ = policy_value(
        np.full(inputs.n, "womens"), inputs.arm, inputs.y[["visit", "spend"]].to_numpy()
    )
    assert last["incremental_visits"] == pytest.approx(inputs.n * (everyone - nobody)[0])
    assert last["net_value"] == pytest.approx(last["incremental_spend"] - 0.1 * inputs.n)


def test_calibration_check_matches_direct_computation(rng):
    inputs = _inputs(rng, 3_000)
    table = calibration_check(inputs).set_index("email")
    y = inputs.y["visit"].to_numpy()
    for email in config.TREATMENTS:
        t, c = inputs.arm == email, inputs.arm == "control"
        res = diff_in_proportions(y[t].sum(), t.sum(), y[c].sum(), c.sum())
        r = table.loc[email]
        assert r["observed_ate"] == pytest.approx(res.diff)
        assert r["predicted_mean_uplift"] == pytest.approx(inputs.tau.loc[t | c, email].mean())
        assert r["predicted_within_ci"] == (res.ci_low <= r["predicted_mean_uplift"] <= res.ci_high)


def test_profiles_are_aggregates(rng):
    inputs = _inputs(rng, 2_000)
    _, info = build_policies(inputs, "mens", rng, 300)
    table = profiles(
        inputs, {"targeted_uplift": info["targeted"], "all": np.ones(inputs.n, bool)}, info
    )
    wide = table.pivot_table(index="variable", columns="group", values="value")
    assert wide.loc["n", "targeted_uplift"] == info["targeted"].sum()
    for col in ("zip_code", "channel"):
        shares = wide.loc[[v for v in wide.index if v.startswith(f"share_{col}_")], "all"]
        assert shares.sum() == pytest.approx(1.0)
    assert wide.loc["share_assigned_mens", "all"] + wide.loc[
        "share_assigned_womens", "all"
    ] == pytest.approx(1.0)
    assert "customer_id" not in set(table["variable"])


def test_ab_winner_rule(tmp_path):
    def results(sig_mens, sig_womens, diff_mens=0.7, diff_womens=0.4, recommended="womens"):
        rows = [
            {
                "metric": "spend",
                "family": "primary",
                "treatment": "mens",
                "significant": sig_mens,
                "diff": diff_mens,
            },
            {
                "metric": "spend",
                "family": "primary",
                "treatment": "womens",
                "significant": sig_womens,
                "diff": diff_womens,
            },
            {
                "metric": "visit",
                "family": "primary",
                "treatment": "womens",
                "significant": True,
                "diff": 9.0,
            },
        ]
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                {
                    "frequentist": {"rows": rows},
                    "bayes_spend": {
                        "recommendation": {
                            "recommended_email": recommended,
                            "p_mens_better_than_womens": 0.3,
                        }
                    },
                }
            )
        )
        return path

    assert ab_winner(results(True, True))[0] == "mens"
    assert ab_winner(results(True, True, diff_mens=0.1))[0] == "womens"
    assert ab_winner(results(False, True))[0] == "womens"
    assert ab_winner(results(True, False))[0] == "mens"
    assert ab_winner(results(False, False))[0] == "womens"
    # No email meets either rule: the head-to-head posterior decides and no winner is declared.
    email, basis = ab_winner(results(False, False, recommended=None))
    assert email == "womens" and "no winner declared" in basis


# --- stage ---------------------------------------------------------------------------------------


def test_run_policy_writes_outputs(fixture_db, customers, tmp_path, rng):
    # A synthetic scores file in the layout the uplift stage writes: every customer, per pair.
    rows = []
    for email in config.TREATMENTS:
        in_pair = customers[config.ARM_COL].isin([email, config.CONTROL]).to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "customer_id": customers["customer_id"],
                    "arm": email,
                    "outcome": "visit",
                    "in_pair": in_pair,
                    "x_learner": 0.05 + 0.05 * rng.standard_normal(len(customers)),
                    "response_model": rng.random(len(customers)),
                }
            )
        )
    scores_dir = tmp_path / "processed"
    scores_dir.mkdir()
    pd.concat(rows).to_parquet(scores_dir / SCORES_NAME, index=False)
    budget = 400
    table = run_policy(
        fixture_db, tmp_path, scores_dir, n_resamples=40, budget=budget, winner="mens"
    )
    assert len(table) == len(POLICIES) * 3
    t = table.set_index(["policy", "outcome"])
    for name in BUDGETED:
        assert t.loc[(name, "visit"), "n_emailed"] <= budget
    assert t.loc[("suppression", "visit"), "n_emailed"] == len(customers) - budget
    for name in (
        "policy",
        "policy_comparison",
        "policy_budget_curve",
        "policy_profiles",
        "policy_calibration",
        "policy_suppressed_uplift",
    ):
        assert len(pd.read_csv(tmp_path / "tables" / f"{name}.csv")) > 0
    assert (tmp_path / "figures" / "policy_budget_curve.png").stat().st_size > 0
    text = (tmp_path / "results.json").read_text()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite value {c} in results"))
    section = json.loads(text)[LABEL]
    assert section["label"] == LABEL and "D8" in section["deviation"]
    assert section["ab_winner"]["email"] == "mens" and section["budget"] == budget
    assert set(section["headline"]) == set(config.OUTCOME_COLS)
    assert set(section["headline"]["visit"]) == set(POLICIES)
    assert set(section["uplift_versus"]["spend"]) == set(POLICIES) - {"uplift"}
    assert isinstance(section["uplift_beats_recency_rule"]["visit"], bool)
    assert len(section["calibration"]) == 2 and len(section["suppressed_group"]) == 3
    assert section["exploratory"] == list(EXPLORATORY)
