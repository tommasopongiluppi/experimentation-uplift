"""Download the Hillstrom CSV, verify its checksum and load it into DuckDB."""

import datetime as dt
import gzip
import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.csv as pacsv

from abkit import config

logger = logging.getLogger(__name__)

RAW_TABLE = "raw_customers"

#: Expected raw columns in file order, with their DuckDB types (no type inference).
RAW_SCHEMA: dict[str, str] = {
    "recency": "INTEGER",
    "history_segment": "VARCHAR",
    "history": "DOUBLE",
    "mens": "INTEGER",
    "womens": "INTEGER",
    "zip_code": "VARCHAR",
    "newbie": "INTEGER",
    "channel": "VARCHAR",
    "segment": "VARCHAR",
    "visit": "INTEGER",
    "conversion": "INTEGER",
    "spend": "DOUBLE",
}

#: Alternative column spellings seen in mirrors, mapped to the original names.
COLUMN_ALIASES: dict[str, str] = {
    "history_seg": "history_segment",
    "zipcode": "zip_code",
    "treatment": "segment",
}

_ARROW_TYPES: dict[str, pa.DataType] = {
    "INTEGER": pa.int32(),
    "DOUBLE": pa.float64(),
    "VARCHAR": pa.string(),
}

_USER_AGENT = "abkit-data-download/0.1"


class DataError(RuntimeError):
    """Raised when the raw data is missing, malformed or fails a check."""


class ChecksumError(DataError):
    """Raised when a file's SHA-256 does not match the recorded value."""


@dataclass(frozen=True)
class Source:
    """A download location for the raw file.

    Attributes
    ----------
    url : str
        URL to download (``http``, ``https`` or ``file``).
    filename : str
        Name under which the file is saved in the raw-data directory.
    sha256 : str or None
        Known SHA-256 of the file, if pinned in config.
    """

    url: str
    filename: str
    sha256: str | None = None


DEFAULT_SOURCES: tuple[Source, ...] = (
    Source(
        config.DATA_URL,
        config.DATA_URL.rsplit("/", 1)[-1],
        config.KNOWN_SHA256.get(config.DATA_URL),
    ),
    Source(
        config.MIRROR_URL,
        config.MIRROR_URL.rsplit("/", 1)[-1],
        config.KNOWN_SHA256.get(config.MIRROR_URL),
    ),
)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Lower-case hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str) -> None:
    """Check that a file's SHA-256 equals the expected digest.

    Parameters
    ----------
    path : Path
        File to check.
    expected : str
        Expected hex digest.

    Raises
    ------
    ChecksumError
        If the digests differ.
    """
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ChecksumError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}. "
            "The file was modified or corrupted; delete it and its manifest to re-download."
        )


def read_manifest(raw_dir: Path = config.DATA_RAW) -> dict[str, Any] | None:
    """Read the download manifest, if one exists.

    Parameters
    ----------
    raw_dir : Path
        Raw-data directory holding ``manifest.json``.

    Returns
    -------
    dict or None
        Manifest contents, or None if there is no manifest.
    """
    path = raw_dir / config.MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def resolve_columns(header: list[str]) -> dict[str, str]:
    """Map a file header to the original Hillstrom column names.

    Names are compared after stripping whitespace, lower-casing and replacing spaces with
    underscores; known mirror spellings in :data:`COLUMN_ALIASES` are also accepted.

    Parameters
    ----------
    header : list of str
        Column names as they appear in the file.

    Returns
    -------
    dict
        Mapping from each file column name to its original name, in file order.

    Raises
    ------
    DataError
        If any expected column is missing, or unexpected columns are present.
    """
    mapping: dict[str, str] = {}
    for name in header:
        key = name.strip().lower().replace(" ", "_")
        mapping[name] = COLUMN_ALIASES.get(key, key)
    resolved = list(mapping.values())
    missing = [c for c in RAW_SCHEMA if c not in resolved]
    unexpected = [c for c in resolved if c not in RAW_SCHEMA]
    if missing or unexpected or len(header) != len(RAW_SCHEMA):
        raise DataError(
            f"Unexpected columns: expected the {len(RAW_SCHEMA)} columns {list(RAW_SCHEMA)}, "
            f"got {header} (missing={missing}, unexpected={unexpected})."
        )
    return mapping


def read_header(path: Path) -> list[str]:
    """Read the header line of a CSV file (plain or gzip-compressed).

    Parameters
    ----------
    path : Path
        CSV file; a ``.gz`` suffix means gzip compression.

    Returns
    -------
    list of str
        Column names.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as fh:
        return fh.readline().rstrip("\r\n").split(",")


def _fetch(url: str, dest: Path, timeout: float, retries: int, backoff: float) -> None:
    """Download ``url`` to ``dest`` atomically, retrying on network errors."""
    tmp = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            tmp.write_bytes(payload)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            logger.warning(
                "Download attempt %d/%d failed for %s: %s", attempt + 1, retries + 1, url, exc
            )
            if attempt < retries:
                time.sleep(backoff * 2**attempt)
    tmp.unlink(missing_ok=True)
    raise DataError(f"Could not download {url} after {retries + 1} attempts: {last_error}")


def download_raw(
    raw_dir: Path = config.DATA_RAW,
    sources: tuple[Source, ...] = DEFAULT_SOURCES,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> Path:
    """Download the raw Hillstrom file once, then verify it on later runs.

    If ``raw_dir`` already holds a manifest and the file it names, the file's checksum is
    verified and no download happens. Otherwise each source is tried in order; using any
    source but the first logs a warning. The manifest records the SHA-256, size, source URL,
    download date, whether the fallback was used and any column renaming needed.

    Parameters
    ----------
    raw_dir : Path
        Directory for the raw file and ``manifest.json``.
    sources : tuple of Source
        Download locations in order of preference.
    timeout : float
        Socket timeout in seconds for each attempt.
    retries : int
        Retries per source after the first failed attempt.
    backoff : float
        Base delay in seconds between attempts (doubled after each failure).

    Returns
    -------
    Path
        Path to the verified raw file.

    Raises
    ------
    ChecksumError
        If an existing file, or a freshly downloaded one, fails its checksum.
    DataError
        If every source fails or the file does not have the expected columns.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(raw_dir)
    if manifest is not None:
        path = raw_dir / manifest["filename"]
        if path.exists():
            verify_checksum(path, manifest["sha256"])
            logger.info("Raw file %s present and checksum verified; skipping download.", path)
            return path
        logger.warning("Manifest found but %s is missing; downloading again.", path)

    for index, source in enumerate(sources):
        dest = raw_dir / source.filename
        try:
            _fetch(source.url, dest, timeout=timeout, retries=retries, backoff=backoff)
        except DataError as exc:
            logger.warning("Source failed: %s", exc)
            continue

        expected = source.sha256
        if manifest is not None and manifest.get("source_url") == source.url:
            expected = manifest["sha256"]
        if expected is not None:
            try:
                verify_checksum(dest, expected)
            except ChecksumError:
                dest.unlink()
                raise

        column_map = {k: v for k, v in resolve_columns(read_header(dest)).items() if k != v}
        if index > 0:
            logger.warning(
                "Primary source unavailable; used fallback %s (column renames: %s).",
                source.url,
                column_map or "none",
            )
        record = {
            "filename": source.filename,
            "source_url": source.url,
            "fallback_used": index > 0,
            "sha256": sha256_file(dest),
            "bytes": dest.stat().st_size,
            "compression": "gzip" if dest.suffix == ".gz" else "none",
            "column_map": column_map,
            "downloaded_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        }
        (raw_dir / config.MANIFEST_NAME).write_text(json.dumps(record, indent=2) + "\n")
        logger.info("Downloaded %s (%d bytes, sha256 %s).", dest, record["bytes"], record["sha256"])
        return dest

    raise DataError(f"All sources failed: {[s.url for s in sources]}")


def _create_table_sql() -> str:
    columns = ",\n    ".join(f"{name} {dtype} NOT NULL" for name, dtype in RAW_SCHEMA.items())
    return (
        f"CREATE OR REPLACE TABLE {RAW_TABLE} (\n"
        f"    customer_id INTEGER PRIMARY KEY,\n    {columns}\n)"
    )


def load_to_duckdb(
    csv_path: Path | None = None,
    db_path: Path = config.DUCKDB_PATH,
    expected_rows: int | None = config.EXPECTED_ROWS,
) -> int:
    """Load the raw file into the DuckDB table ``raw_customers`` with explicit types.

    Columns are parsed with the fixed types in :data:`RAW_SCHEMA` (empty or malformed values
    raise), and ``customer_id`` is added as the 1-based row number in file order.

    Parameters
    ----------
    csv_path : Path, optional
        Raw CSV (plain or ``.gz``). Defaults to the file named in the raw-data manifest.
    db_path : Path
        DuckDB database file; created if missing. The table is replaced if it exists.
    expected_rows : int or None
        Required row count; None disables the check (used for test fixtures).

    Returns
    -------
    int
        Number of rows loaded.

    Raises
    ------
    DataError
        If the file is missing, has the wrong columns, a value that does not parse as its
        column type, or the wrong number of rows.
    """
    if csv_path is None:
        manifest = read_manifest(config.DATA_RAW)
        if manifest is None:
            raise DataError("No raw-data manifest found; run download_raw() first.")
        csv_path = config.DATA_RAW / manifest["filename"]
    if not csv_path.exists():
        raise DataError(f"Raw file not found: {csv_path}")

    names = list(resolve_columns(read_header(csv_path)).values())
    try:
        table = pacsv.read_csv(
            csv_path,
            read_options=pacsv.ReadOptions(column_names=names, skip_rows=1),
            convert_options=pacsv.ConvertOptions(
                column_types={name: _ARROW_TYPES[dtype] for name, dtype in RAW_SCHEMA.items()},
                null_values=[],
                strings_can_be_null=False,
            ),
        ).select(list(RAW_SCHEMA))
    except pa.ArrowInvalid as exc:
        raise DataError(f"Could not parse {csv_path} with the expected types: {exc}") from exc

    n_rows = table.num_rows
    if expected_rows is not None and n_rows != expected_rows:
        raise DataError(f"Expected {expected_rows} rows in {csv_path}, found {n_rows}.")
    table = table.add_column(0, "customer_id", pa.array(range(1, n_rows + 1), type=pa.int32()))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.execute(_create_table_sql())
        con.register("raw_arrow", table)
        con.execute(f"INSERT INTO {RAW_TABLE} SELECT * FROM raw_arrow")
        con.unregister("raw_arrow")
        (loaded,) = con.execute(f"SELECT count(*) FROM {RAW_TABLE}").fetchone()
    if loaded != n_rows:
        raise DataError(f"Loaded {loaded} rows into {RAW_TABLE}, expected {n_rows}.")
    logger.info("Loaded %d rows from %s into %s:%s.", n_rows, csv_path, db_path, RAW_TABLE)
    return n_rows
