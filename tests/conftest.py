"""Shared pytest fixtures."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from abkit.data import load_to_duckdb
from abkit.validate import CLEAN_SCHEMA, CLEAN_TABLE, clean_table
from tests.fixtures.make_fixture import FIXTURE_PATH


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    """DuckDB file holding raw_customers and the cleaned customers table from the fixture."""
    db = tmp_path / "fixture.duckdb"
    load_to_duckdb(FIXTURE_PATH, db_path=db, expected_rows=None)
    with duckdb.connect(str(db)) as con:
        clean_table(con)
    return db


@pytest.fixture
def customers(fixture_db: Path) -> pd.DataFrame:
    with duckdb.connect(str(fixture_db), read_only=True) as con:
        return con.execute(f"SELECT * FROM {CLEAN_TABLE} ORDER BY customer_id").df()


def customers_con(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """In-memory connection with a ``customers`` table built from a cleaned-schema DataFrame."""
    con = duckdb.connect()
    columns = ", ".join(f"{c} {t}" for c, t in CLEAN_SCHEMA.items())
    con.execute(f"CREATE TABLE {CLEAN_TABLE} ({columns})")
    con.register("df", df[list(CLEAN_SCHEMA)])
    con.execute(f"INSERT INTO {CLEAN_TABLE} SELECT * FROM df")
    return con
