import json

import numpy as np
import pandas as pd
import pytest

from abkit.eda import SRMError, compute_eda, log_histogram, run_checks
from abkit.metrics import arm_counts, load_sql
from tests.conftest import customers_con

QS = [0.5, 0.9, 0.99, 0.999, 1.0]


def test_load_sql_reads_packaged_file():
    assert "FROM customers" in load_sql("arm_counts")


def test_arm_counts_match_pandas(customers):
    assert (
        arm_counts(customers_con(customers)) == customers.arm.value_counts().sort_index().to_dict()
    )


def test_eda_rates_and_quantiles_match_numpy(customers):
    eda = compute_eda(customers_con(customers))
    r = eda["eda_rates"].iloc[0]
    assert r["visit_rate"] == pytest.approx(customers.visit.mean())
    assert r["conversion_rate"] == pytest.approx(customers.conversion.mean())
    assert r["share_zero_spend"] == pytest.approx((customers.spend == 0).mean())
    dist = eda["eda_distributions"].set_index("variable")
    cols = ["p50", "p90", "p99", "p99_9", "max"]
    np.testing.assert_allclose(
        dist.loc["history", cols].astype(float), np.quantile(customers.history, QS)
    )
    conv = customers.loc[customers.conversion == 1, "spend"]
    np.testing.assert_allclose(dist.loc["spend", cols].astype(float), np.quantile(conv, QS))
    assert dist.loc["spend", "n"] == len(conv)


def test_eda_correlation_matches_numpy(customers):
    corr = compute_eda(customers_con(customers))["eda_correlations"].set_index("outcome")
    for outcome in ("spend", "visit", "conversion"):
        expected = np.corrcoef(customers.history, customers[outcome])[0, 1]
        assert corr.loc[outcome, "pearson_r"] == pytest.approx(expected)


def test_eda_recency_counts(customers):
    rec = compute_eda(customers_con(customers))["eda_recency"].set_index("recency")
    expected = customers.recency.value_counts()
    assert rec["n"].to_dict() == expected.to_dict()
    assert rec["share"].sum() == pytest.approx(1.0)


def test_eda_is_pooled_never_by_arm(customers):
    for table in compute_eda(customers_con(customers)).values():
        assert "arm" not in table.columns


def test_log_histogram_counts_every_value():
    values = np.array([1.0, 10.0, 10.0, 100.0, 1000.0])
    hist = log_histogram(values, n_bins=3)
    np.testing.assert_allclose(hist.bin_left, [1, 10, 100])
    np.testing.assert_allclose(hist.bin_right, [10, 100, 1000])
    assert hist.n.tolist() == [1, 2, 2]
    with pytest.raises(ValueError):
        log_histogram(np.array([0.0, 1.0]))


def test_run_checks_writes_outputs(fixture_db, tmp_path):
    reports = tmp_path / "reports"
    srm = run_checks(fixture_db, reports)
    assert srm.passed
    for name in (
        "srm",
        "balance",
        "balance_chi2",
        "eda_rates",
        "eda_distributions",
        "eda_recency",
        "eda_correlations",
        "eda_spend_converters_hist",
        "eda_history_hist",
    ):
        assert (reports / "tables" / f"{name}.csv").exists(), name
    for name in (
        "srm_arm_counts",
        "balance_smd",
        "eda_spend_converters_log",
        "eda_history_log",
        "eda_recency",
    ):
        assert (reports / "figures" / f"{name}.png").stat().st_size > 10_000, name
    md = (reports / "eda.md").read_text()
    srm_csv = pd.read_csv(reports / "tables" / "srm.csv")
    assert f"{srm_csv.observed.iloc[0]:,}" in md
    results = json.loads((reports / "results.json").read_text())
    assert results["validity"]["srm"]["p_value"] == pytest.approx(srm.p_value)
    assert set(results) == {"validity", "eda"}


def test_run_checks_raises_on_srm_failure(customers, tmp_path):
    df = pd.concat([customers, customers[customers.arm == "mens"]], ignore_index=True)
    df["customer_id"] = np.arange(1, len(df) + 1)
    db = tmp_path / "srm.duckdb"
    con = customers_con(df)
    con.execute(f"ATTACH '{db}' AS out")
    con.execute("CREATE TABLE out.customers AS SELECT * FROM customers")
    con.close()
    with pytest.raises(SRMError):
        run_checks(db, tmp_path / "reports")
    assert (tmp_path / "reports" / "tables" / "srm.csv").exists()
