import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abkit import config
from abkit.frequentist import benjamini_hochberg, holm
from abkit.readout import COMPARISONS, READOUT_METRICS, build_readout, family_of, run_frequentist
from abkit.report import json_records
from tests.conftest import customers_con

N_BOOT = 500


@pytest.fixture
def readout(customers):
    table, cap = build_readout(customers_con(customers), np.random.default_rng(3), N_BOOT)
    return table, cap


def test_readout_has_every_comparison_and_metric(readout):
    table, _ = readout
    assert len(table) == len(COMPARISONS) * len(READOUT_METRICS) == 12
    pairs = set(zip(table.comparison, table.metric, strict=True))
    assert pairs == {(f"{t}_vs_{c}", m) for t, c in COMPARISONS for m in READOUT_METRICS}


def test_families_follow_the_plan(readout):
    table, _ = readout
    sizes = table.family.value_counts().to_dict()
    assert sizes == {"primary": 2, "secondary": 7, "sensitivity": 2, "sensitivity_secondary": 1}
    primary = table[table.family == "primary"]
    assert set(primary.metric) == {"spend"} and set(primary.control) == {"control"}
    assert family_of("spend", "mens", "womens") == "secondary"
    with pytest.raises(ValueError):
        family_of("history", "mens", "control")


def test_adjusted_pvalues_use_family_methods(readout):
    table, _ = readout
    for family, fn in (("primary", holm), ("secondary", benjamini_hochberg), ("sensitivity", holm)):
        part = table[table.family == family]
        np.testing.assert_allclose(part.p_adj, fn(part.p_value.to_numpy()), rtol=1e-12)
    single = table[table.family == "sensitivity_secondary"]
    assert (single.p_adj == single.p_value).all()
    assert (table.significant == (table.p_adj < config.ALPHA)).all()


def test_readout_matches_row_level_calculation(readout, customers):
    table, cap = readout
    conv = customers.loc[customers.conversion == 1, "spend"]
    assert cap == pytest.approx(np.quantile(conv, 0.95), rel=1e-12)
    df = customers.assign(spend_winsorised=customers.spend.clip(upper=cap))
    for _, r in table.iterrows():
        yt = df.loc[df.arm == r.treatment, r.metric].astype(float)
        yc = df.loc[df.arm == r.control, r.metric].astype(float)
        assert r.mean_treatment == pytest.approx(yt.mean(), rel=1e-9)
        assert r.mean_control == pytest.approx(yc.mean(), rel=1e-9)
        assert r["diff"] == pytest.approx(yt.mean() - yc.mean(), rel=1e-9, abs=1e-12)
        assert r.n_treatment == len(yt) and r.n_control == len(yc)
        if r.metric.startswith("spend"):
            ref = stats.ttest_ind(yt, yc, equal_var=False)
            assert r.p_value == pytest.approx(ref.pvalue, rel=1e-8)
            assert r.boot_ci_low < r.boot_ci_high
        else:
            assert math.isnan(r.boot_ci_low)


def test_json_records_are_strict_json():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.inf, np.int64(2)], "c": ["x", "y"]})
    records = json_records(df)
    assert records == [{"a": 1.0, "b": None, "c": "x"}, {"a": None, "b": 2.0, "c": "y"}]
    json.dumps(records, allow_nan=False)


def test_run_frequentist_writes_outputs(fixture_db, tmp_path):
    table = run_frequentist(fixture_db, tmp_path, n_resamples=N_BOOT)
    written = pd.read_csv(tmp_path / "tables" / "readout_frequentist.csv")
    assert len(written) == len(table)
    assert (tmp_path / "figures" / "forest_relative_lift.png").stat().st_size > 0
    section = json.loads((tmp_path / "results.json").read_text())["frequentist"]
    assert len(section["rows"]) == len(table)
    assert section["families"]["secondary"] == {"adjustment": "bh", "n_tests": 7}
    assert section["bootstrap"]["seed"] == config.SEEDS["bootstrap"]
