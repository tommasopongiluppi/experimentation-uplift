"""Run the SQL metric layer and return per-arm sufficient statistics."""

import logging
from importlib import resources

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def load_sql(name: str) -> str:
    """Read a query from the package's ``sql/`` directory.

    Parameters
    ----------
    name : str
        File name without the ``.sql`` suffix, e.g. ``"arm_counts"``.

    Returns
    -------
    str
        The SQL text.
    """
    return resources.files("abkit").joinpath("sql", f"{name}.sql").read_text()


def run_sql(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """Run a packaged query against a connection holding the ``customers`` table.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.
    name : str
        Query name (file name in ``sql/`` without suffix).

    Returns
    -------
    pandas.DataFrame
        Query result.
    """
    logger.debug("Running sql/%s.sql", name)
    return con.execute(load_sql(name)).df()


def arm_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Count the customers in each arm.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with a ``customers`` table.

    Returns
    -------
    dict
        Arm name to customer count, ordered by arm name.
    """
    df = run_sql(con, "arm_counts")
    return {str(a): int(n) for a, n in zip(df["arm"], df["n"], strict=True)}
