"""Write results.json and tables, and update generated README sections."""

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abkit import config

logger = logging.getLogger(__name__)


def write_table(df: pd.DataFrame, name: str, tables_dir: Path = config.TABLES) -> Path:
    """Write a table as CSV under ``tables_dir``.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to write (the index is not written).
    name : str
        File name without the ``.csv`` suffix.
    tables_dir : Path
        Output directory.

    Returns
    -------
    Path
        The path written.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def read_table(name: str, tables_dir: Path = config.TABLES) -> pd.DataFrame:
    """Read a table written by :func:`write_table`."""
    return pd.read_csv(tables_dir / f"{name}.csv", keep_default_na=False)


def update_results(section: str, data: dict[str, Any], path: Path = config.RESULTS_JSON) -> Path:
    """Replace one top-level section of ``results.json``, keeping the others.

    Parameters
    ----------
    section : str
        Top-level key, e.g. ``"validity"``.
    data : dict
        JSON-serialisable content for that key.
    path : Path
        Location of results.json.

    Returns
    -------
    Path
        The path written.
    """
    results = json.loads(path.read_text()) if path.exists() else {}
    results[section] = data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(results.items())), indent=2) + "\n")
    logger.info("Updated section %r of %s.", section, path)
    return path


def _json_value(x: Any) -> Any:
    """Convert numpy scalars to Python types, and NaN or infinity to None (JSON null)."""
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Turn a table into JSON-safe records for a results.json section.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to convert.

    Returns
    -------
    list of dict
        One dict per row; numpy types become Python types and NaN or infinity becomes None.
    """
    return [
        {str(k): _json_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")
    ]
