"""Write results.json and tables, and update the generated blocks of the README.

``python -m abkit.report update-readme`` rewrites every block of ``README.md`` that sits
between ``<!-- RESULTS:name -->`` and ``<!-- /RESULTS:name -->`` from ``results.json`` and
the tables it names; ``--check`` only reports whether the file is up to date. The text of
each block is built in :mod:`abkit.readme`.
"""

import argparse
import json
import logging
import math
import re
import sys
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


# ---------------------------------------------------------------------------------------------
# Generated README blocks
# ---------------------------------------------------------------------------------------------

_BLOCK = re.compile(
    r"(?P<open><!-- RESULTS:(?P<name>[a-z0-9_]+) -->\n)(?P<body>.*?)"
    r"(?P<close><!-- /RESULTS:(?P=name) -->)",
    re.DOTALL,
)


class ReadmeError(ValueError):
    """The README's markers and the generated blocks do not match."""


def block_names(text: str) -> list[str]:
    """Names of the ``<!-- RESULTS:name -->`` blocks in ``text``, in order of appearance."""
    return [m.group("name") for m in _BLOCK.finditer(text)]


def render_blocks(text: str, blocks: dict[str, str]) -> str:
    """Replace the body of every marked block in ``text`` with its generated content.

    Parameters
    ----------
    text : str
        Markdown with ``<!-- RESULTS:name -->`` ... ``<!-- /RESULTS:name -->`` pairs.
    blocks : dict of str to str
        Generated markdown per block name.

    Returns
    -------
    str
        The text with every block body replaced; everything outside the markers is kept.

    Raises
    ------
    ReadmeError
        If a marker has no generated block, or a generated block has no marker.
    """
    present = block_names(text)
    unknown = sorted(set(present) - set(blocks))
    unused = sorted(set(blocks) - set(present))
    if unknown or unused:
        raise ReadmeError(f"markers without a block: {unknown}; blocks without a marker: {unused}")
    return _BLOCK.sub(
        lambda m: f"{m.group('open')}{blocks[m.group('name')].strip()}\n{m.group('close')}", text
    )


def update_readme(
    readme_path: Path = config.README,
    results_path: Path = config.RESULTS_JSON,
    check: bool = False,
) -> bool:
    """Regenerate the marked blocks of the README from ``results.json``.

    Parameters
    ----------
    readme_path : Path
        The README to update.
    results_path : Path
        ``results.json``; the tables it names are read from its directory.
    check : bool
        Only compare; do not write.

    Returns
    -------
    bool
        True if the README was already up to date.
    """
    from abkit.readme import build_blocks  # imported here: abkit.readme imports this module

    results = json.loads(results_path.read_text())
    current = readme_path.read_text()
    updated = render_blocks(current, build_blocks(results, results_path.parent))
    up_to_date = updated == current
    if not up_to_date and not check:
        readme_path.write_text(updated)
        logger.info("Updated the generated blocks of %s.", readme_path)
    return up_to_date


def main(argv: list[str] | None = None) -> int:
    """Run the command line: ``python -m abkit.report update-readme [--check]``.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        0 on success; 1 if ``--check`` finds the README out of date.
    """
    parser = argparse.ArgumentParser(prog="python -m abkit.report", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    readme = commands.add_parser(
        "update-readme", help="Regenerate the RESULTS blocks of README.md from results.json."
    )
    readme.add_argument("--readme", type=Path, default=config.README)
    readme.add_argument("--results", type=Path, default=config.RESULTS_JSON)
    readme.add_argument(
        "--check", action="store_true", help="Exit with 1 if the README is out of date."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    up_to_date = update_readme(args.readme, args.results, check=args.check)
    if args.check and not up_to_date:
        logger.error("%s is out of date; run python -m abkit.report update-readme.", args.readme)
        return 1
    if up_to_date:
        logger.info("%s is up to date.", args.readme)
    return 0


if __name__ == "__main__":
    sys.exit(main())
