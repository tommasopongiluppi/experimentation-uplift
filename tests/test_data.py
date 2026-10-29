import json
import logging

import duckdb
import numpy as np
import pandas as pd
import pytest

from abkit import config
from abkit.data import (
    RAW_SCHEMA,
    RAW_TABLE,
    ChecksumError,
    DataError,
    Source,
    download_raw,
    load_to_duckdb,
    resolve_columns,
    sha256_file,
    verify_checksum,
)
from tests.fixtures.make_fixture import FIXTURE_PATH, make_fixture, true_effects

FIXTURE_ROWS = 3000


def _src(path, filename="raw.csv", sha256=None):
    return Source(path.as_uri(), filename, sha256)


def _download(raw_dir, *sources, **kwargs):
    return download_raw(raw_dir, sources=sources, retries=0, backoff=0.0, **kwargs)


# --- fixture ---------------------------------------------------------------------------------


def test_fixture_file_is_reproducible():
    committed = pd.read_csv(FIXTURE_PATH)
    regenerated = make_fixture(np.random.default_rng(config.SEEDS["fixture"]), n=FIXTURE_ROWS)
    pd.testing.assert_frame_equal(committed, regenerated, check_dtype=False)


def test_fixture_recovers_planted_effects():
    df = make_fixture(np.random.default_rng(1), n=300_000)
    df["arm"] = df["segment"].map(config.RAW_ARM_MAP)
    means = df.groupby("arm")[["visit", "conversion", "spend"]].mean()
    ses = df.groupby("arm")[["visit", "conversion", "spend"]].sem()
    for arm, effects in true_effects().items():
        for outcome, planted in effects.items():
            estimate = means.loc[arm, outcome] - means.loc[config.CONTROL, outcome]
            se = np.hypot(ses.loc[arm, outcome], ses.loc[config.CONTROL, outcome])
            assert abs(estimate - planted) < 4 * se, (arm, outcome)


def test_fixture_conversion_implies_visit_and_spend_only_for_converters():
    df = make_fixture(np.random.default_rng(2), n=5000)
    assert (df.loc[df.conversion == 1, "visit"] == 1).all()
    assert (df.loc[df.conversion == 0, "spend"] == 0).all()
    assert (df.loc[df.conversion == 1, "spend"] > 0).all()


# --- checksums and download -------------------------------------------------------------------


def test_verify_checksum_detects_modified_file(tmp_path):
    path = tmp_path / "f.csv"
    path.write_bytes(b"a,b\r\n1,2\r\n")
    digest = sha256_file(path)
    verify_checksum(path, digest)
    path.write_bytes(b"a,b\r\n1,3\r\n")
    with pytest.raises(ChecksumError):
        verify_checksum(path, digest)


def test_sha256_matches_known_value(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_download_writes_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    path = _download(raw_dir, _src(FIXTURE_PATH))
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    assert path == raw_dir / "raw.csv"
    assert manifest["sha256"] == sha256_file(FIXTURE_PATH)
    assert manifest["bytes"] == FIXTURE_PATH.stat().st_size
    assert manifest["source_url"] == FIXTURE_PATH.as_uri()
    assert manifest["fallback_used"] is False
    assert manifest["column_map"] == {}


def test_second_run_skips_download(tmp_path):
    raw_dir = tmp_path / "raw"
    _download(raw_dir, _src(FIXTURE_PATH))
    unreachable = _src(tmp_path / "does_not_exist.csv")
    assert _download(raw_dir, unreachable) == raw_dir / "raw.csv"


def test_second_run_detects_modified_file(tmp_path):
    raw_dir = tmp_path / "raw"
    path = _download(raw_dir, _src(FIXTURE_PATH))
    with path.open("ab") as fh:
        fh.write(b"1,x\r\n")
    with pytest.raises(ChecksumError):
        _download(raw_dir, _src(FIXTURE_PATH))


def test_pinned_checksum_mismatch_fails_and_removes_file(tmp_path):
    raw_dir = tmp_path / "raw"
    with pytest.raises(ChecksumError):
        _download(raw_dir, _src(FIXTURE_PATH, sha256="0" * 64))
    assert not (raw_dir / "raw.csv").exists()
    assert not (raw_dir / "manifest.json").exists()


def test_fallback_is_used_and_recorded(tmp_path, caplog):
    raw_dir = tmp_path / "raw"
    renamed = tmp_path / "mirror.csv"
    text = FIXTURE_PATH.read_text(newline="")
    renamed.write_text(text.replace("zip_code", "Zip Code", 1), newline="")
    with caplog.at_level(logging.WARNING, logger="abkit.data"):
        path = _download(raw_dir, _src(tmp_path / "missing.csv"), _src(renamed, "mirror.csv"))
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    assert path.name == "mirror.csv"
    assert manifest["fallback_used"] is True
    assert manifest["source_url"] == renamed.as_uri()
    assert manifest["column_map"] == {"Zip Code": "zip_code"}
    assert any("fallback" in r.getMessage() for r in caplog.records)


def test_all_sources_failing_raises(tmp_path):
    with pytest.raises(DataError, match="All sources failed"):
        _download(tmp_path / "raw", _src(tmp_path / "a.csv"), _src(tmp_path / "b.csv"))


# --- column resolution ------------------------------------------------------------------------


def test_resolve_columns_identity():
    assert resolve_columns(list(RAW_SCHEMA)) == {c: c for c in RAW_SCHEMA}


def test_resolve_columns_missing_column_raises():
    with pytest.raises(DataError, match="missing"):
        resolve_columns(list(RAW_SCHEMA)[:-1])


def test_resolve_columns_extra_column_raises():
    with pytest.raises(DataError, match="unexpected"):
        resolve_columns([*RAW_SCHEMA, "customer_id"])


# --- loader -----------------------------------------------------------------------------------


@pytest.fixture
def loaded_db(tmp_path):
    db_path = tmp_path / "test.duckdb"
    n = load_to_duckdb(FIXTURE_PATH, db_path=db_path, expected_rows=FIXTURE_ROWS)
    assert n == FIXTURE_ROWS
    return db_path


def test_loader_types(loaded_db):
    with duckdb.connect(str(loaded_db), read_only=True) as con:
        described = con.execute(f"DESCRIBE {RAW_TABLE}").fetchall()
    types = {row[0]: row[1] for row in described}
    assert list(types) == ["customer_id", *RAW_SCHEMA]
    assert types == {"customer_id": "INTEGER", **RAW_SCHEMA}


def test_loader_row_count_and_customer_id(loaded_db):
    with duckdb.connect(str(loaded_db), read_only=True) as con:
        n, lo, hi, distinct = con.execute(
            f"SELECT count(*), min(customer_id), max(customer_id), count(DISTINCT customer_id) "
            f"FROM {RAW_TABLE}"
        ).fetchone()
    assert (n, lo, hi, distinct) == (FIXTURE_ROWS, 1, FIXTURE_ROWS, FIXTURE_ROWS)


def test_loader_preserves_file_order(loaded_db):
    expected = pd.read_csv(FIXTURE_PATH)
    with duckdb.connect(str(loaded_db), read_only=True) as con:
        loaded = con.execute(f"SELECT * FROM {RAW_TABLE} ORDER BY customer_id").df()
    pd.testing.assert_frame_equal(loaded.drop(columns="customer_id"), expected, check_dtype=False)


def test_loader_wrong_row_count_raises(tmp_path):
    with pytest.raises(DataError, match="Expected 64000 rows"):
        load_to_duckdb(FIXTURE_PATH, db_path=tmp_path / "x.duckdb", expected_rows=64_000)


def test_loader_blank_numeric_value_raises(tmp_path):
    lines = FIXTURE_PATH.read_text(newline="").split("\r\n")
    fields = lines[1].split(",")
    fields[0] = ""
    lines[1] = ",".join(fields)
    bad = tmp_path / "bad.csv"
    bad.write_text("\r\n".join(lines), newline="")
    with pytest.raises(DataError, match="expected types"):
        load_to_duckdb(bad, db_path=tmp_path / "x.duckdb", expected_rows=None)


def test_loader_wrong_columns_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\r\n1,2\r\n")
    with pytest.raises(DataError, match="Unexpected columns"):
        load_to_duckdb(bad, db_path=tmp_path / "x.duckdb", expected_rows=None)


# --- real download ----------------------------------------------------------------------------


@pytest.mark.network
def test_real_download_and_load(tmp_path):
    raw_dir = tmp_path / "raw"
    path = download_raw(raw_dir, retries=1, backoff=1.0)
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    if manifest["source_url"] == config.DATA_URL:
        assert manifest["sha256"] == config.KNOWN_SHA256[config.DATA_URL]
    assert load_to_duckdb(path, db_path=tmp_path / "h.duckdb") == config.EXPECTED_ROWS
