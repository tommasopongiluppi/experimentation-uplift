"""Validate the raw Hillstrom table and clean it into the ``customers`` table.

Validation never compares outcomes between arms (analysis-plan rule): it checks schema, missing
values, ranges, category labels, logical consistency between columns, duplicates and pooled
distributions. Cleaning renames the assignment column, recodes labels to lower_snake_case and
keeps every row.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
import pandas as pd

from abkit import config
from abkit.data import RAW_SCHEMA, RAW_TABLE, DataError

logger = logging.getLogger(__name__)

CLEAN_TABLE = "customers"
VALIDATION_JSON = config.REPORTS / "validation.json"
VALIDATION_MD = config.REPORTS / "validation.md"
CUSTOMERS_PARQUET = config.DATA_PROCESSED / "customers.parquet"
QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999, 1.0)

#: Column order and DuckDB types of the cleaned table.
CLEAN_SCHEMA: dict[str, str] = {
    "customer_id": "INTEGER",
    config.ARM_COL: "VARCHAR",
    **{c: RAW_SCHEMA[c] for c in config.PRE_TREATMENT_COLS},
    **{c: RAW_SCHEMA[c] for c in config.OUTCOME_COLS},
}

DUPLICATE_NOTE = (
    "Exact duplicate rows are counted but kept. The file has no customer identifier and its "
    "columns are coarse (several are binary or binned), so two different customers can have "
    "identical rows; dropping them would remove real customers and bias the arm sizes."
)

Severity = Literal["critical", "warning"]


class ValidationError(DataError):
    """Raised when at least one critical validation check fails."""


@dataclass
class Check:
    """Result of one validation check.

    Attributes
    ----------
    name : str
        Dotted check identifier, e.g. ``"range.spend"``.
    severity : {"critical", "warning"}
        Critical failures stop the pipeline; warnings are reported only.
    passed : bool
        Whether the check passed.
    observed : Any
        JSON-serialisable observed value.
    expected : str
        Plain-language statement of what passing means.
    """

    name: str
    severity: Severity
    passed: bool
    observed: Any
    expected: str


@dataclass
class LabelFix:
    """A raw category label that was corrected during cleaning."""

    column: str
    raw: str
    canonical: str
    clean: str
    reason: str
    rows: int


@dataclass
class ValidationReport:
    """All validation results for one table."""

    table: str
    n_rows: int
    checks: list[Check] = field(default_factory=list)
    categories: dict[str, dict[str, int]] = field(default_factory=dict)
    label_fixes: list[LabelFix] = field(default_factory=list)
    duplicates: dict[str, Any] = field(default_factory=dict)
    distributions: dict[str, Any] = field(default_factory=dict)

    @property
    def critical_failures(self) -> list[Check]:
        """Critical checks that failed."""
        return [c for c in self.checks if c.severity == "critical" and not c.passed]

    def check(self, name: str) -> Check:
        """Return the check with the given name."""
        return next(c for c in self.checks if c.name == name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        n_failed = sum(not c.passed for c in self.checks)
        return {
            "table": self.table,
            "n_rows": self.n_rows,
            "summary": {
                "n_checks": len(self.checks),
                "n_passed": len(self.checks) - n_failed,
                "n_failed": n_failed,
                "critical_failures": [c.name for c in self.critical_failures],
            },
            "checks": [asdict(c) for c in self.checks],
            "categories": self.categories,
            "label_fixes": [asdict(f) for f in self.label_fixes],
            "duplicates": self.duplicates,
            "distributions": self.distributions,
        }


def _normalise(label: str) -> str:
    return " ".join(label.split()).casefold()


def resolve_label(column: str, label: str) -> tuple[str, str | None] | None:
    """Map a raw category label to its canonical raw label.

    Parameters
    ----------
    column : str
        Categorical column name (a key of ``config.CATEGORY_MAPS``).
    label : str
        Raw label as found in the data.

    Returns
    -------
    tuple or None
        ``(canonical_label, reason)`` where ``reason`` is None for an exact canonical label,
        ``"misspelling"`` or ``"inconsistent formatting"`` for a fixed label; None if the label
        cannot be mapped.
    """
    canonical = config.CATEGORY_MAPS[column]
    misspellings = config.LABEL_MISSPELLINGS.get(column, {})
    if label in canonical:
        return label, None
    if label in misspellings:
        return misspellings[label], "misspelling"
    key = _normalise(label)
    for known in canonical:
        if _normalise(known) == key:
            return known, "inconsistent formatting"
    for known, target in misspellings.items():
        if _normalise(known) == key:
            return target, "misspelling"
    return None


def _json_number(x: Any) -> Any:
    if isinstance(x, np.integer | int):
        return int(x)
    return float(x)


def _schema_checks(
    con: duckdb.DuckDBPyConnection, table: str, expected_rows: int | None
) -> list[Check]:
    described = {row[0]: row[1] for row in con.execute(f"DESCRIBE {table}").fetchall()}
    expected_types = {"customer_id": "INTEGER", **RAW_SCHEMA}
    n_rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    checks = [
        Check(
            "schema.columns_and_types",
            "critical",
            described == expected_types,
            described,
            f"exactly these columns and types: {expected_types}",
        )
    ]
    if expected_rows is not None:
        checks.append(
            Check(
                "schema.row_count",
                "critical",
                n_rows == expected_rows,
                n_rows,
                f"{expected_rows} rows",
            )
        )
    return checks


def _missing_check(df: pd.DataFrame) -> Check:
    missing = {}
    for col in df.columns:
        n = df[col].isna()
        if not pd.api.types.is_numeric_dtype(df[col]):
            n = n | (df[col].astype(str).str.strip() == "")
        missing[col] = int(n.sum())
    return Check(
        "missing_values",
        "critical",
        sum(missing.values()) == 0,
        missing,
        "no NULL or blank values in any column",
    )


def _range_checks(df: pd.DataFrame) -> list[Check]:
    lo, hi = config.RECENCY_RANGE
    rec = df["recency"].dropna()
    checks = [
        Check(
            "range.recency",
            "warning",
            bool(((rec >= lo) & (rec <= hi) & (rec == rec.round())).all()),
            {
                "min": _json_number(rec.min()),
                "max": _json_number(rec.max()),
                "n_distinct": int(rec.nunique()),
            },
            f"whole months in [{lo}, {hi}] (population: purchased in the last 12 months)",
        )
    ]
    for col in ("history", "spend"):
        values = df[col].dropna()
        n_negative = int((values < 0).sum())
        checks.append(
            Check(
                f"range.{col}",
                "critical",
                n_negative == 0,
                {
                    "min": _json_number(values.min()),
                    "max": _json_number(values.max()),
                    "n_negative": n_negative,
                },
                f"{col} >= 0",
            )
        )
    for col in config.BINARY_COLS:
        values = df[col].dropna()
        bad = values[~values.isin([0, 1])]
        checks.append(
            Check(
                f"binary.{col}",
                "critical",
                bad.empty,
                {
                    "distinct": sorted(_json_number(v) for v in values.unique()),
                    "n_invalid": len(bad),
                },
                f"{col} in {{0, 1}}",
            )
        )
    return checks


def _category_checks(
    df: pd.DataFrame,
) -> tuple[list[Check], dict[str, dict[str, int]], list[LabelFix]]:
    checks: list[Check] = []
    categories: dict[str, dict[str, int]] = {}
    fixes: list[LabelFix] = []
    for col in config.CATEGORICAL_COLS:
        counts = df[col].dropna().value_counts().sort_index()
        categories[col] = {str(k): int(v) for k, v in counts.items()}
        unknown: dict[str, int] = {}
        for label, n in categories[col].items():
            resolved = resolve_label(col, label)
            if resolved is None:
                unknown[label] = n
                continue
            canonical, reason = resolved
            if reason is not None:
                clean = config.CATEGORY_MAPS[col][canonical]
                fixes.append(LabelFix(col, label, canonical, clean, reason, n))
        col_fixes = [f for f in fixes if f.column == col]
        checks.append(
            Check(
                f"categories.{col}.known",
                "critical",
                not unknown,
                {"unmapped_labels": unknown},
                f"every label is one of {list(config.CATEGORY_MAPS[col])} "
                "or a known fixable variant",
            )
        )
        checks.append(
            Check(
                f"categories.{col}.canonical",
                "warning",
                not col_fixes,
                {"labels_fixed": {f.raw: f.canonical for f in col_fixes}},
                "every label is already spelled canonically (no fixes needed)",
            )
        )
    return checks, categories, fixes


def _history_band(history: pd.Series) -> pd.Series:
    bins = [lo for lo, _, _ in config.HISTORY_BANDS] + [config.HISTORY_BANDS[-1][1]]
    labels = [name for _, _, name in config.HISTORY_BANDS]
    return pd.cut(history, bins=bins, labels=labels, right=False).astype(object)


def _logic_checks(df: pd.DataFrame) -> list[Check]:
    conv, visit, spend = df["conversion"], df["visit"], df["spend"]
    conv_no_visit = int(((conv == 1) & (visit == 0)).sum())
    spend_no_conv = int(((spend > 0) & (conv == 0)).sum())
    conv_no_spend = int(((conv == 1) & (spend <= 0)).sum())

    def clean_band(label: Any) -> Any:
        resolved = resolve_label("history_segment", label) if isinstance(label, str) else None
        return config.CATEGORY_MAPS["history_segment"][resolved[0]] if resolved else None

    stated = df["history_segment"].map(clean_band)
    implied = _history_band(df["history"])
    comparable = stated.notna() & implied.notna()
    band_mismatch = int((stated[comparable] != implied[comparable]).sum())
    neither = int(((df["mens"] == 0) & (df["womens"] == 0)).sum())
    return [
        Check(
            "logic.conversion_implies_visit",
            "warning",
            conv_no_visit == 0,
            {"conversion_without_visit": conv_no_visit},
            "every converter also visited",
        ),
        Check(
            "logic.spend_iff_conversion",
            "warning",
            spend_no_conv == 0 and conv_no_spend == 0,
            {"spend_without_conversion": spend_no_conv, "conversion_without_spend": conv_no_spend},
            "spend > 0 exactly when conversion = 1",
        ),
        Check(
            "logic.history_segment_matches_history",
            "warning",
            band_mismatch == 0,
            {"mismatches": band_mismatch},
            "history_segment is the dollar band of history (lower bound inclusive)",
        ),
        Check(
            "logic.mens_or_womens_history",
            "warning",
            neither == 0,
            {"mens_0_and_womens_0": neither},
            "every customer bought men's or women's merchandise in the past year",
        ),
    ]


def _duplicates(df: pd.DataFrame) -> tuple[Check, dict[str, Any]]:
    cols = [c for c in df.columns if c != "customer_id"]
    is_dup = df.duplicated(subset=cols, keep=False)
    in_groups = int(is_dup.sum())
    extra = int(df.duplicated(subset=cols, keep="first").sum())
    top_history = _json_number(df["history"].mode().iloc[0]) if len(df) else None
    info = {
        "rows_in_duplicate_groups": in_groups,
        "extra_copies": extra,
        "dropped": 0,
        "most_common_history": top_history,
        "rows_in_duplicate_groups_at_most_common_history": int(
            (is_dup & (df["history"] == top_history)).sum()
        ),
        "note": DUPLICATE_NOTE,
    }
    check = Check(
        "duplicates.exact_rows",
        "warning",
        extra == 0,
        {"rows_in_duplicate_groups": in_groups, "extra_copies": extra},
        "no exact duplicate rows (duplicates are kept either way)",
    )
    return check, info


def _quantiles(values: pd.Series) -> dict[str, float]:
    qs = np.quantile(values.to_numpy(dtype=float), QUANTILES)
    return {
        ("max" if q == 1.0 else f"p{q * 100:g}"): float(v)
        for q, v in zip(QUANTILES, qs, strict=True)
    }


def _top_values(values: pd.Series, k: int) -> list[dict[str, Any]]:
    counts = (
        values.value_counts()
        .sort_index(ascending=False)
        .sort_values(ascending=False, kind="stable")
    )
    return [{"value": _json_number(v), "count": int(n)} for v, n in counts.head(k).items()]


def _distributions(df: pd.DataFrame) -> dict[str, Any]:
    spend = df["spend"].dropna()
    converters = df.loc[df["conversion"] == 1, "spend"].dropna()
    return {
        "history_quantiles": _quantiles(df["history"].dropna()),
        "spend_quantiles_converters": _quantiles(converters) if len(converters) else {},
        "n_converters": len(converters),
        "share_zero_spend": float((spend == 0).mean()),
        "history_most_common": _top_values(df["history"].dropna(), 1)[0],
        "spend_converters_most_common": _top_values(converters, 3),
        "n_converters_at_max_spend": int((converters == converters.max()).sum())
        if len(converters)
        else 0,
        "quantile_method": "numpy linear interpolation",
    }


def validate_table(
    con: duckdb.DuckDBPyConnection,
    table: str = RAW_TABLE,
    expected_rows: int | None = config.EXPECTED_ROWS,
) -> ValidationReport:
    """Run every validation check on a raw table.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection holding the table.
    table : str
        Name of the raw table (columns as in :data:`abkit.data.RAW_SCHEMA` plus ``customer_id``).
    expected_rows : int or None
        Required row count; None skips the row-count check.

    Returns
    -------
    ValidationReport
        Checks, category counts, label fixes, duplicate counts and pooled distributions.
    """
    df = con.execute(f"SELECT * FROM {table} ORDER BY customer_id").df()
    report = ValidationReport(table=table, n_rows=len(df))
    report.checks += _schema_checks(con, table, expected_rows)
    if not report.check("schema.columns_and_types").passed:
        logger.error("Schema check failed; skipping content checks.")
        return report
    report.checks.append(_missing_check(df))
    report.checks += _range_checks(df)
    category_checks, report.categories, report.label_fixes = _category_checks(df)
    report.checks += category_checks
    report.checks += _logic_checks(df)
    dup_check, report.duplicates = _duplicates(df)
    report.checks.append(dup_check)
    report.distributions = _distributions(df)

    for fix in report.label_fixes:
        logger.warning(
            "Label fix in %s: %r -> %r (%s, %d rows).",
            fix.column,
            fix.raw,
            fix.clean,
            fix.reason,
            fix.rows,
        )
    for check in report.checks:
        if not check.passed:
            log = logger.error if check.severity == "critical" else logger.warning
            log("Check %s failed (%s): %s", check.name, check.severity, check.observed)
    return report


def clean_table(
    con: duckdb.DuckDBPyConnection,
    source: str = RAW_TABLE,
    target: str = CLEAN_TABLE,
) -> int:
    """Build the cleaned table from the raw table, keeping every row.

    ``segment`` is renamed to ``arm`` (values from ``config.RAW_ARM_MAP``), known label fixes
    are applied and categorical values are recoded to lower_snake_case via
    ``config.CATEGORY_MAPS``. No rows are dropped and no values are capped.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection holding the raw table.
    source, target : str
        Raw and cleaned table names; the target is replaced if it exists.

    Returns
    -------
    int
        Number of rows in the cleaned table.

    Raises
    ------
    ValidationError
        If a label cannot be mapped (run :func:`validate_table` first).
    """
    df = con.execute(f"SELECT * FROM {source} ORDER BY customer_id").df()
    for col in config.CATEGORICAL_COLS:
        mapping = {}
        for label in df[col].unique():
            resolved = resolve_label(col, label)
            if resolved is None:
                raise ValidationError(f"Cannot clean unmapped label {label!r} in column {col}.")
            mapping[label] = config.CATEGORY_MAPS[col][resolved[0]]
        df[col] = df[col].map(mapping)
    df = df.rename(columns={config.RAW_ARM_COL: config.ARM_COL})[list(CLEAN_SCHEMA)]

    columns = ",\n    ".join(f"{name} {dtype} NOT NULL" for name, dtype in CLEAN_SCHEMA.items())
    con.execute(
        f"CREATE OR REPLACE TABLE {target} (\n    {columns},\n    PRIMARY KEY (customer_id)\n)"
    )
    con.register("clean_df", df)
    con.execute(f"INSERT INTO {target} SELECT * FROM clean_df")
    con.unregister("clean_df")
    n_clean = con.execute(f"SELECT count(*) FROM {target}").fetchone()[0]
    if n_clean != len(df):
        raise ValidationError(f"Cleaned table has {n_clean} rows, expected {len(df)}.")
    return n_clean


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4g}" if abs(value) < 1 else f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_fmt(v)}" for k, v in value.items()) or "none"
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    """Render a validation report (as produced by :meth:`ValidationReport.to_dict`) as Markdown.

    Parameters
    ----------
    result : dict
        Parsed contents of ``validation.json``.

    Returns
    -------
    str
        Markdown document.
    """
    s = result["summary"]
    status = "FAILED" if s["critical_failures"] else "PASSED"
    lines = [
        "# Data Validation Report",
        "",
        "Generated by `abkit.validate` from `reports/validation.json`; do not edit by hand.",
        "",
        f"- Table: `{result['table']}`, {_fmt(result['n_rows'])} rows",
        f"- Checks: {s['n_checks']} run, {s['n_passed']} passed, {s['n_failed']} failed",
        f"- Critical failures: {', '.join(s['critical_failures']) or 'none'} (**{status}**)",
        "- No outcome is compared between arms here; that waits for the analysis plan.",
        "",
        "## Checks",
        "",
        "| Check | Severity | Result | Observed | Expected |",
        "|---|---|---|---|---|",
    ]
    for c in result["checks"]:
        observed = (
            "see below"
            if c["name"] == "schema.columns_and_types" and c["passed"]
            else _fmt(c["observed"])
        )
        mark = "pass" if c["passed"] else "**FAIL**"
        lines.append(f"| `{c['name']}` | {c['severity']} | {mark} | {observed} | {c['expected']} |")

    lines += ["", "## Category labels (raw values and counts)", ""]
    for col, counts in result["categories"].items():
        lines += [f"**{col}**", "", "| Raw label | Rows |", "|---|---:|"]
        lines += [f"| `{label}` | {_fmt(n)} |" for label, n in counts.items()]
        lines.append("")

    dup = result["duplicates"]
    lines += ["## Label fixes applied in cleaning", ""]
    if result["label_fixes"]:
        lines += [
            "| Column | Raw label | Canonical | Cleaned value | Reason | Rows |",
            "|---|---|---|---|---|---:|",
        ]
        lines += [
            f"| {f['column']} | `{f['raw']}` | `{f['canonical']}` | `{f['clean']}` "
            f"| {f['reason']} | {_fmt(f['rows'])} |"
            for f in result["label_fixes"]
        ]
    else:
        lines.append("None.")
    lines += [
        "",
        "All other labels are recoded to lower_snake_case with the maps in "
        "`abkit.config.CATEGORY_MAPS`; `segment` is renamed to `arm`.",
        "",
        "## Duplicate rows",
        "",
        f"- Rows in groups of identical rows: {_fmt(dup.get('rows_in_duplicate_groups', 0))}",
        f"- Extra copies beyond the first: {_fmt(dup.get('extra_copies', 0))}",
        f"- Rows dropped: {_fmt(dup.get('dropped', 0))}",
        f"- Of the rows in duplicate groups, "
        f"{_fmt(dup.get('rows_in_duplicate_groups_at_most_common_history', 0))} have the most "
        f"common history value (${_fmt(dup.get('most_common_history'))}).",
        "",
        dup.get("note", ""),
        "",
        "## Distributions (all arms pooled)",
        "",
    ]
    d = result["distributions"]
    if d:
        hq, sq = d["history_quantiles"], d["spend_quantiles_converters"]
        lines += ["| Quantile | history ($) | spend among converters ($) |", "|---|---:|---:|"]
        lines += [f"| {q} | {_fmt(hq[q])} | {_fmt(sq[q]) if sq else 'n/a'} |" for q in hq]
        lines += [
            "",
            f"- Converters: {_fmt(d['n_converters'])}",
            f"- Share of customers with zero spend: {d['share_zero_spend']:.2%}",
            f"- Most common history value: ${_fmt(d['history_most_common']['value'])} "
            f"({_fmt(d['history_most_common']['count'])} customers)",
            "- Most common spend values among converters: "
            + ", ".join(
                f"${_fmt(x['value'])} ({_fmt(x['count'])})"
                for x in d["spend_converters_most_common"]
            ),
            f"- Converters whose spend equals the maximum: {_fmt(d['n_converters_at_max_spend'])} "
            "(a heap at the maximum is consistent with spend being capped; outlier handling is "
            "decided in the analysis plan)",
            f"- Quantile method: {d['quantile_method']}",
        ]
    return "\n".join(lines) + "\n"


def write_report(
    report: ValidationReport, json_path: Path = VALIDATION_JSON, md_path: Path = VALIDATION_MD
) -> None:
    """Write ``validation.json`` and the Markdown report generated from it.

    Parameters
    ----------
    report : ValidationReport
        Validation results.
    json_path, md_path : Path
        Output paths.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    md_path.write_text(render_markdown(json.loads(json_path.read_text())))


def run_validation(
    db_path: Path = config.DUCKDB_PATH,
    reports_dir: Path = config.REPORTS,
    parquet_path: Path = CUSTOMERS_PARQUET,
    expected_rows: int | None = config.EXPECTED_ROWS,
) -> ValidationReport:
    """Validate ``raw_customers``, write the reports and, if no critical check fails, clean.

    Parameters
    ----------
    db_path : Path
        DuckDB database holding ``raw_customers``; ``customers`` is written there.
    reports_dir : Path
        Directory for ``validation.json`` and ``validation.md``.
    parquet_path : Path
        Output path of the cleaned table as Parquet.
    expected_rows : int or None
        Required row count; None skips the check.

    Returns
    -------
    ValidationReport
        The validation results.

    Raises
    ------
    ValidationError
        If any critical check fails (the reports are still written).
    """
    with duckdb.connect(str(db_path)) as con:
        report = validate_table(con, RAW_TABLE, expected_rows)
        write_report(report, reports_dir / VALIDATION_JSON.name, reports_dir / VALIDATION_MD.name)
        if report.critical_failures:
            names = [c.name for c in report.critical_failures]
            raise ValidationError(f"Critical validation checks failed: {names}")
        n_rows = clean_table(con)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY {CLEAN_TABLE} TO '{parquet_path}' (FORMAT parquet)")
    logger.info("Validation passed; wrote %d rows to %s and %s.", n_rows, CLEAN_TABLE, parquet_path)
    return report
