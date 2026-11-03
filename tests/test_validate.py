import json
import re

import duckdb
import numpy as np
import pandas as pd
import pytest

from abkit import config
from abkit.data import RAW_SCHEMA, RAW_TABLE, load_to_duckdb
from abkit.validate import (
    CLEAN_SCHEMA,
    CLEAN_TABLE,
    ValidationError,
    clean_table,
    render_markdown,
    resolve_label,
    run_validation,
    validate_table,
)
from tests.fixtures.make_fixture import FIXTURE_PATH

N = 3000
SNAKE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")


def fixture_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE_PATH, keep_default_na=False)
    df.insert(0, "customer_id", np.arange(1, len(df) + 1))
    return df


def baseline_extra_copies() -> int:
    df = fixture_df().drop(columns="customer_id")
    return int(df.duplicated(keep="first").sum())


def raw_con(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with a nullable raw_customers table holding ``df``."""
    con = duckdb.connect()
    columns = ", ".join(f"{c} {t}" for c, t in {"customer_id": "INTEGER", **RAW_SCHEMA}.items())
    con.execute(f"CREATE TABLE {RAW_TABLE} ({columns})")
    con.register("df", df)
    con.execute(f"INSERT INTO {RAW_TABLE} SELECT * FROM df")
    return con


def validate(df: pd.DataFrame, expected_rows: int | None = N):
    return validate_table(raw_con(df), expected_rows=expected_rows)


@pytest.fixture
def clean_report():
    return validate(fixture_df())


# --- clean fixture ----------------------------------------------------------------------------


def test_fixture_has_no_critical_failures(clean_report):
    assert clean_report.critical_failures == []


def test_fixture_only_known_warning_is_suburban_spelling(clean_report):
    failed = {c.name for c in clean_report.checks if not c.passed}
    expected = {"categories.zip_code.canonical"}
    if baseline_extra_copies():  # coarse columns make chance duplicates possible
        expected.add("duplicates.exact_rows")
    assert failed == expected
    (fix,) = clean_report.label_fixes
    assert (fix.column, fix.raw, fix.clean, fix.reason) == (
        "zip_code",
        "Surburban",
        "suburban",
        "misspelling",
    )
    assert fix.rows == clean_report.categories["zip_code"]["Surburban"]


def test_category_counts_sum_to_rows(clean_report):
    for col in config.CATEGORICAL_COLS:
        assert sum(clean_report.categories[col].values()) == N


# --- planted problems -------------------------------------------------------------------------


def test_missing_value_detected():
    df = fixture_df()
    df["history"] = df["history"].astype(float)
    df.loc[10, "history"] = np.nan
    df.loc[20, "channel"] = "  "
    report = validate(df)
    check = report.check("missing_values")
    assert not check.passed and check.severity == "critical"
    assert check.observed["history"] == 1
    assert check.observed["channel"] == 1
    assert sum(check.observed.values()) == 2


def test_negative_spend_detected():
    df = fixture_df()
    df.loc[5, "spend"] = -12.5
    check = validate(df).check("range.spend")
    assert not check.passed and check.severity == "critical"
    assert check.observed["n_negative"] == 1
    assert check.observed["min"] == -12.5


def test_negative_history_detected():
    df = fixture_df()
    df.loc[5, "history"] = -1.0
    assert not validate(df).check("range.history").passed


def test_unmappable_misspelt_label_is_critical():
    df = fixture_df()
    df.loc[7, "channel"] = "Phnoe"
    report = validate(df)
    check = report.check("categories.channel.known")
    assert not check.passed and check.severity == "critical"
    assert check.observed["unmapped_labels"] == {"Phnoe": 1}
    with pytest.raises(ValidationError, match="Phnoe"):
        clean_table(raw_con(df))


def test_inconsistent_label_is_fixed_and_reported():
    df = fixture_df()
    df.loc[[3, 4], "channel"] = "web "
    df.loc[8, "zip_code"] = "URBAN"
    report = validate(df)
    assert report.check("categories.channel.known").passed
    assert not report.check("categories.channel.canonical").passed
    fixes = {(f.column, f.raw): (f.clean, f.reason, f.rows) for f in report.label_fixes}
    assert fixes[("channel", "web ")] == ("web", "inconsistent formatting", 2)
    assert fixes[("zip_code", "URBAN")] == ("urban", "inconsistent formatting", 1)


def test_conversion_without_visit_detected():
    df = fixture_df()
    row = df.index[df.conversion == 1][0]
    df.loc[row, "visit"] = 0
    check = validate(df).check("logic.conversion_implies_visit")
    assert not check.passed and check.severity == "warning"
    assert check.observed == {"conversion_without_visit": 1}


def test_spend_without_conversion_and_conversion_without_spend_detected():
    df = fixture_df()
    df.loc[df.index[df.conversion == 0][0], "spend"] = 10.0
    for row in df.index[df.conversion == 1][:2]:
        df.loc[row, "spend"] = 0.0
    check = validate(df).check("logic.spend_iff_conversion")
    assert not check.passed
    assert check.observed == {"spend_without_conversion": 1, "conversion_without_spend": 2}


def test_history_segment_mismatch_detected():
    df = fixture_df()
    df.loc[0, "history"] = 50.0
    df.loc[0, "history_segment"] = "7) $1,000 +"
    check = validate(df).check("logic.history_segment_matches_history")
    assert check.observed == {"mismatches": 1}


def test_history_band_boundary_is_lower_inclusive():
    df = fixture_df()
    df.loc[0, "history"] = 100.0
    df.loc[0, "history_segment"] = "2) $100 - $200"
    assert validate(df).check("logic.history_segment_matches_history").passed


def test_mens_and_womens_zero_counted():
    df = fixture_df()
    before = int(((df.mens == 0) & (df.womens == 0)).sum())
    df.loc[[0, 1, 2], ["mens", "womens"]] = 0
    expected = before + int(
        ((fixture_df().loc[[0, 1, 2], ["mens", "womens"]] != 0).any(axis=1)).sum()
    )
    check = validate(df).check("logic.mens_or_womens_history")
    assert check.observed == {"mens_0_and_womens_0": expected}
    assert not check.passed


def test_non_binary_value_detected():
    df = fixture_df()
    df.loc[0, "newbie"] = 2
    check = validate(df).check("binary.newbie")
    assert not check.passed and check.observed["n_invalid"] == 1


def test_recency_range_reports_observed_values():
    df = fixture_df()
    check = validate(df).check("range.recency")
    assert check.observed["min"] == df.recency.min()
    assert check.observed["max"] == df.recency.max()
    df.loc[0, "recency"] = 13
    assert not validate(df).check("range.recency").passed


def test_duplicates_counted_not_dropped():
    df = fixture_df()
    dupes = df.iloc[[0, 0, 1]].copy()
    dupes["customer_id"] = [N + 1, N + 2, N + 3]
    df = pd.concat([df, dupes], ignore_index=True)
    con = raw_con(df)
    report = validate_table(con, expected_rows=None)
    check = report.check("duplicates.exact_rows")
    expected_extra = baseline_extra_copies() + 3
    assert check.observed["extra_copies"] == expected_extra
    base = fixture_df().drop(columns="customer_id")
    expected_in_groups = int(pd.concat([base, base.iloc[[0, 0, 1]]]).duplicated(keep=False).sum())
    assert check.observed["rows_in_duplicate_groups"] == expected_in_groups
    assert not check.passed
    assert report.duplicates["dropped"] == 0
    assert clean_table(con) == N + 3


def test_schema_row_count_checked():
    check = validate(fixture_df(), expected_rows=64_000).check("schema.row_count")
    assert not check.passed and check.observed == N


def test_schema_types_checked():
    con = raw_con(fixture_df())
    con.execute(f"ALTER TABLE {RAW_TABLE} ALTER spend TYPE VARCHAR")
    report = validate_table(con, expected_rows=N)
    assert not report.check("schema.columns_and_types").passed
    assert "schema.columns_and_types" in [c.name for c in report.critical_failures]
    assert report.check("schema.columns_and_types").observed["spend"] == "VARCHAR"


# --- distributions (known answers) ------------------------------------------------------------


def test_distributions_match_numpy(clean_report):
    df = fixture_df()
    d = clean_report.distributions
    expected_h = np.quantile(df.history, [0.5, 0.9, 0.99, 0.999, 1.0])
    assert list(d["history_quantiles"]) == ["p50", "p90", "p99", "p99.9", "max"]
    np.testing.assert_allclose(list(d["history_quantiles"].values()), expected_h)
    conv = df.loc[df.conversion == 1, "spend"]
    np.testing.assert_allclose(
        list(d["spend_quantiles_converters"].values()),
        np.quantile(conv, [0.5, 0.9, 0.99, 0.999, 1.0]),
    )
    assert d["n_converters"] == len(conv)
    assert d["share_zero_spend"] == pytest.approx((df.spend == 0).mean(), abs=0)


# --- cleaning ---------------------------------------------------------------------------------


def test_resolve_label():
    assert resolve_label("zip_code", "Rural") == ("Rural", None)
    assert resolve_label("zip_code", "Surburban") == ("Suburban", "misspelling")
    assert resolve_label("zip_code", " surburban") == ("Suburban", "misspelling")
    assert resolve_label("channel", "WEB") == ("Web", "inconsistent formatting")
    assert resolve_label("channel", "Fax") is None


def test_clean_table_schema_and_values():
    df = fixture_df()
    con = raw_con(df)
    assert clean_table(con) == N
    described = {r[0]: r[1] for r in con.execute(f"DESCRIBE {CLEAN_TABLE}").fetchall()}
    assert described == CLEAN_SCHEMA
    clean = con.execute(f"SELECT * FROM {CLEAN_TABLE} ORDER BY customer_id").df()
    assert set(clean.arm) == set(config.ARMS)
    expected_arm = df.segment.map(config.RAW_ARM_MAP).to_numpy()
    np.testing.assert_array_equal(clean.arm.to_numpy(), expected_arm)
    for col in ("history_segment", "zip_code", "channel", "arm"):
        assert all(SNAKE.match(v) for v in clean[col].unique()), col
    assert "suburban" in set(clean.zip_code)
    pd.testing.assert_series_equal(clean.spend, df.spend, check_names=False, check_dtype=False)


def test_run_validation_writes_outputs(tmp_path):
    db = tmp_path / "h.duckdb"
    load_to_duckdb(FIXTURE_PATH, db_path=db, expected_rows=N)
    parquet = tmp_path / "processed" / "customers.parquet"
    report = run_validation(db, tmp_path / "reports", parquet, expected_rows=N)
    result = json.loads((tmp_path / "reports" / "validation.json").read_text())
    assert result["summary"]["critical_failures"] == []
    assert result["n_rows"] == report.n_rows == N
    md = (tmp_path / "reports" / "validation.md").read_text()
    assert md == render_markdown(result)
    for check in result["checks"]:
        assert f"`{check['name']}`" in md
    assert len(pd.read_parquet(parquet)) == N


def test_run_validation_raises_on_critical_failure(tmp_path):
    db = tmp_path / "h.duckdb"
    load_to_duckdb(FIXTURE_PATH, db_path=db, expected_rows=N)
    with pytest.raises(ValidationError, match=r"schema\.row_count"):
        run_validation(db, tmp_path / "reports", tmp_path / "c.parquet", expected_rows=64_000)
    result = json.loads((tmp_path / "reports" / "validation.json").read_text())
    assert result["summary"]["critical_failures"] == ["schema.row_count"]
    with duckdb.connect(str(db), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert CLEAN_TABLE not in tables


def test_point_masses_reported(clean_report):
    df = fixture_df()
    d = clean_report.distributions
    top = df.history.value_counts()
    assert d["history_most_common"]["count"] == top.iloc[0]
    conv = df.loc[df.conversion == 1, "spend"]
    assert d["n_converters_at_max_spend"] == int((conv == conv.max()).sum())
    counts = [x["count"] for x in d["spend_converters_most_common"]]
    assert counts == sorted(counts, reverse=True)


def test_duplicates_attributed_to_most_common_history():
    df = fixture_df()
    df.loc[:9, "history"] = 29.99
    dupes = df.iloc[[0, 0]].copy()
    dupes["customer_id"] = [N + 1, N + 2]
    df = pd.concat([df, dupes], ignore_index=True)
    info = validate(df, expected_rows=None).duplicates
    assert info["most_common_history"] == 29.99
    body = df.drop(columns="customer_id")
    expected = int((body.duplicated(keep=False) & (body.history == 29.99)).sum())
    assert expected >= 3
    assert info["rows_in_duplicate_groups_at_most_common_history"] == expected
