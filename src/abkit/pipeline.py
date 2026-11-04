"""Command-line entry point: ``python -m abkit.pipeline [<stage> | run-all] [options]``.

Stages run in dependency order, from the raw download to the validated ``results.json``::

    data -> validate -> checks -> frequentist -> cuped -> bayes -> power -> sequential
         -> segments -> uplift -> policy -> [robustness] -> report

Every stage caches its outputs: a fingerprint of its input files, the ``results.json``
sections it reads, the source files it depends on and its settings is stored in
``data/pipeline_cache.json``, and the stage is skipped when the fingerprint is unchanged and
its outputs still exist. ``--force`` ignores the cache. ``--fast`` replaces the simulation,
bootstrap and seed counts with :data:`abkit.config.FAST_SIZES`; ``--data`` runs the whole
pipeline on a CSV such as the synthetic fixture instead of the download; ``--out`` moves
every output (database, processed data, reports) under another directory.
"""

import argparse
import datetime as dt
import fnmatch
import functools
import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from abkit import __version__, bayes, config, readout, segments, sequential
from abkit.bayes import run_bayes_rates, run_bayes_spend
from abkit.data import DEFAULT_SOURCES, DataError, download_raw, load_to_duckdb, read_manifest
from abkit.eda import EDA_MD, SRMError, run_checks
from abkit.power import run_power
from abkit.readout import run_cuped, run_frequentist
from abkit.report import update_results
from abkit.results_schema import OPTIONAL_SECTIONS, SCHEMA, ResultsSchemaError, check_results
from abkit.segments import run_segments
from abkit.sequential import run_sequential
from abkit.uplift import models as uplift_models
from abkit.uplift import policy as uplift_policy
from abkit.uplift import robustness as uplift_robustness
from abkit.uplift.learners import GRID, Grid
from abkit.uplift.models import SCORES_NAME, run_uplift_models, score_pair
from abkit.uplift.policy import run_policy
from abkit.uplift.robustness import run_seed_robustness
from abkit.validate import (
    CLEAN_TABLE,
    VALIDATION_JSON,
    VALIDATION_MD,
    ValidationError,
    run_validation,
)

logger = logging.getLogger(__name__)

PACKAGE_DIR: Path = Path(__file__).resolve().parent
#: Source files every stage depends on (hashed into every fingerprint).
CORE_MODULES: tuple[str, ...] = ("config.py", "metrics.py", "report.py", "sql/*.sql")
#: The full-size simulation, bootstrap and seed counts, read from the modules that own them.
FULL_SIZES: dict[str, int] = {
    "frequentist_resamples": readout.N_RESAMPLES,
    "bayes_draws": bayes.N_DRAWS,
    "bayes_ppc_replicates": bayes.N_PPC_REPLICATES,
    "sequential_runs": sequential.N_RUNS,
    "sequential_runs_looks": sequential.N_RUNS_LOOKS,
    "sequential_paths": sequential.N_PATHS,
    "segments_sim_runs": segments.N_SIM_RUNS,
    "uplift_resamples": uplift_models.N_RESAMPLES,
    "policy_resamples": uplift_policy.N_RESAMPLES,
    "robustness_seeds": uplift_robustness.N_SEEDS,
}
#: Fast mode also models visit only and searches a single hyperparameter point.
FAST_UPLIFT_OUTCOMES: tuple[str, ...] = ("visit",)
FAST_GRID: Grid = GRID[:1]
ROBUSTNESS_MINUTES: int = 24


@dataclass(frozen=True)
class Settings:
    """Options of one pipeline invocation.

    Attributes
    ----------
    fast : bool
        Use :data:`abkit.config.FAST_SIZES` instead of the full sizes.
    data : Path or None
        CSV to load instead of downloading the Hillstrom file (row count unchecked).
    out : Path
        Root under which ``data/`` and ``reports/`` are written (the project root by default).
    with_robustness : bool
        Include the seed-robustness stage in ``run-all``.
    force : bool
        Ignore the stage cache and run every requested stage.
    """

    fast: bool = False
    data: Path | None = None
    out: Path = config.ROOT
    with_robustness: bool = False
    force: bool = False

    @property
    def data_dir(self) -> Path:
        """Directory holding the raw and processed data and the DuckDB file."""
        return self.out / config.DATA_DIR.name

    @property
    def raw_dir(self) -> Path:
        """Directory of the downloaded file and its manifest."""
        return self.data_dir / config.DATA_RAW.name

    @property
    def processed_dir(self) -> Path:
        """Directory of the cleaned parquet and the uplift scores."""
        return self.data_dir / config.DATA_PROCESSED.name

    @property
    def db_path(self) -> Path:
        """DuckDB database file."""
        return self.data_dir / config.DUCKDB_PATH.name

    @property
    def reports_dir(self) -> Path:
        """Reports directory (tables, figures, results.json)."""
        return self.out / config.REPORTS.name

    @property
    def tables_dir(self) -> Path:
        """Tables directory."""
        return self.reports_dir / config.TABLES.name

    @property
    def figures_dir(self) -> Path:
        """Figures directory."""
        return self.reports_dir / config.FIGURES.name

    @property
    def results_path(self) -> Path:
        """Location of results.json."""
        return self.reports_dir / config.RESULTS_JSON.name

    @property
    def cache_path(self) -> Path:
        """Location of the stage cache."""
        return self.data_dir / config.PIPELINE_CACHE.name

    @property
    def parquet_path(self) -> Path:
        """Cleaned ``customers`` table as parquet, the input of every analysis stage."""
        return self.processed_dir / "customers.parquet"

    @property
    def scores_path(self) -> Path:
        """Out-of-fold uplift scores written by the uplift stage."""
        return self.processed_dir / SCORES_NAME

    @property
    def raw_path(self) -> Path:
        """The raw CSV: ``--data`` if given, else the manifest's file or the default name."""
        if self.data is not None:
            return self.data
        manifest = read_manifest(self.raw_dir)
        name = manifest["filename"] if manifest else DEFAULT_SOURCES[0].filename
        return self.raw_dir / name

    @property
    def mode(self) -> str:
        """``"fast"`` or ``"full"``."""
        return "fast" if self.fast else "full"

    @property
    def sizes(self) -> dict[str, int]:
        """Simulation, bootstrap and seed counts for this mode."""
        return dict(config.FAST_SIZES) if self.fast else dict(FULL_SIZES)

    @property
    def uplift_outcomes(self) -> tuple[str, ...]:
        """Outcomes the uplift stage models."""
        return FAST_UPLIFT_OUTCOMES if self.fast else uplift_models.OUTCOMES

    @property
    def grid(self) -> Grid:
        """Hyperparameter grid of the uplift learners."""
        return FAST_GRID if self.fast else GRID

    @property
    def data_label(self) -> str:
        """``"hillstrom"`` for the download, else the name of the CSV given."""
        return "hillstrom" if self.data is None else self.data.name

    def ensure_dirs(self) -> None:
        """Create the data and report directories."""
        for path in (self.raw_dir, self.processed_dir, self.tables_dir, self.figures_dir):
            path.mkdir(parents=True, exist_ok=True)


def budget_for(settings: Settings) -> int:
    """Email budget: the configured one on the real data, scaled by row count on a fixture.

    Parameters
    ----------
    settings : Settings
        Pipeline settings; the cleaned table must exist.

    Returns
    -------
    int
        ``config.BUDGET`` when the table has the expected row count, else the same share
        of the rows (at least one).
    """
    with duckdb.connect(str(settings.db_path), read_only=True) as con:
        (n,) = con.execute(f"SELECT count(*) FROM {CLEAN_TABLE}").fetchone()
    if n == config.EXPECTED_ROWS:
        return config.BUDGET
    return max(1, round(config.BUDGET * n / config.EXPECTED_ROWS))


# ---------------------------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """One pipeline stage and what its cache fingerprint covers.

    Attributes
    ----------
    name : str
        Subcommand name.
    help : str
        One-line description for the command-line help.
    run : callable
        ``run(settings)`` performs the stage.
    inputs : callable
        ``inputs(settings)`` lists the files whose content is hashed.
    outputs : callable
        ``outputs(settings)`` lists the files that must exist for a cache hit.
    sections : tuple of str
        ``results.json`` sections the stage writes (must exist for a cache hit).
    reads_sections : tuple of str
        ``results.json`` sections the stage reads (hashed as inputs).
    modules : tuple of str
        Source files under the package directory the stage depends on (globs allowed).
    params : callable
        ``params(settings)`` returns the settings that affect the outputs.
    optional : bool
        Left out of ``run-all`` unless ``--with-robustness`` is given.
    cache : bool
        Whether the stage is ever skipped.
    """

    name: str
    help: str
    run: Callable[[Settings], None]
    inputs: Callable[[Settings], list[Path]]
    outputs: Callable[[Settings], list[Path]]
    sections: tuple[str, ...] = ()
    reads_sections: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    params: Callable[[Settings], dict[str, Any]] = lambda s: {}
    optional: bool = False
    cache: bool = True


def _tables(settings: Settings, *names: str) -> list[Path]:
    return [settings.tables_dir / f"{n}.csv" for n in names]


def _figures(settings: Settings, *names: str) -> list[Path]:
    return [settings.figures_dir / n for n in names]


def _run_data(s: Settings) -> None:
    if s.data is not None:
        load_to_duckdb(s.data, s.db_path, expected_rows=None)
    else:
        load_to_duckdb(download_raw(s.raw_dir), s.db_path)


def _run_validate(s: Settings) -> None:
    run_validation(
        s.db_path,
        s.reports_dir,
        s.parquet_path,
        expected_rows=None if s.data is not None else config.EXPECTED_ROWS,
    )


def _run_bayes(s: Settings) -> None:
    run_bayes_rates(s.db_path, s.reports_dir, n_draws=s.sizes["bayes_draws"])
    run_bayes_spend(
        s.db_path,
        s.reports_dir,
        n_draws=s.sizes["bayes_draws"],
        n_replicates=s.sizes["bayes_ppc_replicates"],
    )


def _run_sequential(s: Settings) -> None:
    run_sequential(
        s.db_path,
        s.reports_dir,
        n_runs=s.sizes["sequential_runs"],
        n_runs_looks=s.sizes["sequential_runs_looks"],
        n_paths=s.sizes["sequential_paths"],
    )


def _run_uplift(s: Settings) -> None:
    run_uplift_models(
        s.db_path,
        s.reports_dir,
        s.processed_dir,
        n_resamples=s.sizes["uplift_resamples"],
        outcomes=s.uplift_outcomes,
        grid=s.grid,
    )


def _log_budget(s: Settings) -> int:
    budget = budget_for(s)
    if budget != config.BUDGET:
        logger.info("Budget scaled from %d to %d emails for the row count.", config.BUDGET, budget)
    return budget


def _run_policy(s: Settings) -> None:
    run_policy(
        s.db_path,
        s.reports_dir,
        s.processed_dir,
        n_resamples=s.sizes["policy_resamples"],
        budget=_log_budget(s),
    )


def _run_robustness(s: Settings) -> None:
    run_seed_robustness(
        s.db_path,
        s.reports_dir,
        s.processed_dir,
        n_seeds=s.sizes["robustness_seeds"],
        budget=_log_budget(s),
        scorer=functools.partial(score_pair, grid=s.grid),
    )


def stages_in_run(settings: Settings) -> list[str]:
    """Names of the stages ``run-all`` executes under ``settings``, in order."""
    return [n for n, st in STAGES.items() if not st.optional or settings.with_robustness]


def _run_report(s: Settings) -> None:
    """Write the ``pipeline`` section and validate ``results.json`` against the schema."""
    update_results(
        "pipeline",
        {
            "mode": s.mode,
            "data": s.data_label,
            "with_robustness": s.with_robustness,
            "stages": stages_in_run(s),
            "sizes": s.sizes,
            "abkit_version": __version__,
        },
        s.results_path,
    )
    required = frozenset(SCHEMA) - OPTIONAL_SECTIONS
    if s.with_robustness:
        required = required | {"seed_robustness"}
    check_results(s.results_path, required)


def _uplift_figures(s: Settings) -> list[Path]:
    return _figures(
        s,
        *(
            f"{kind}_{arm}_{outcome}.png"
            for arm in config.TREATMENTS
            for outcome in s.uplift_outcomes
            for kind in ("qini", "calibration")
        ),
    )


_SEEDS = config.SEEDS
STAGES: dict[str, Stage] = {
    st.name: st
    for st in (
        Stage(
            "data",
            "Download (or verify) the raw file, or load --data, into DuckDB.",
            _run_data,
            inputs=lambda s: [s.raw_path],
            outputs=lambda s: [s.db_path],
            modules=("data.py",),
            params=lambda s: {"data": s.data_label},
        ),
        Stage(
            "validate",
            "Validate the raw table, write validation.json/.md and build the cleaned table.",
            _run_validate,
            inputs=lambda s: [s.raw_path],
            outputs=lambda s: [
                s.db_path,
                s.parquet_path,
                s.reports_dir / VALIDATION_JSON.name,
                s.reports_dir / VALIDATION_MD.name,
            ],
            modules=("data.py", "validate.py"),
            params=lambda s: {"expected_rows": None if s.data else config.EXPECTED_ROWS},
        ),
        Stage(
            "checks",
            "SRM and balance checks plus pooled EDA.",
            lambda s: run_checks(s.db_path, s.reports_dir),
            inputs=lambda s: [s.parquet_path],
            outputs=lambda s: [
                *_tables(
                    s,
                    "srm",
                    "balance",
                    "balance_chi2",
                    "eda_rates",
                    "eda_distributions",
                    "eda_correlations",
                    "eda_recency",
                    "eda_history_hist",
                    "eda_spend_converters_hist",
                ),
                *_figures(
                    s,
                    "srm_arm_counts.png",
                    "balance_smd.png",
                    "eda_spend_converters_log.png",
                    "eda_history_log.png",
                    "eda_recency.png",
                ),
                s.reports_dir / EDA_MD,
            ],
            sections=("validity", "eda"),
            modules=("eda.py", "balance.py", "frequentist.py", "plots.py"),
            params=lambda s: {"srm_alpha": config.SRM_ALPHA, "smd": config.SMD_THRESHOLD},
        ),
        Stage(
            "frequentist",
            "Frequentist readout of every comparison and metric.",
            lambda s: run_frequentist(
                s.db_path, s.reports_dir, n_resamples=s.sizes["frequentist_resamples"]
            ),
            inputs=lambda s: [s.parquet_path],
            outputs=lambda s: [
                *_tables(s, readout.TABLE_NAME),
                *_figures(s, readout.FIGURE_NAME),
            ],
            sections=("frequentist",),
            modules=("readout.py", "frequentist.py", "cuped.py", "eda.py", "plots.py"),
            params=lambda s: {
                "n_resamples": s.sizes["frequentist_resamples"],
                "seed": _SEEDS["bootstrap"],
                "alpha": config.ALPHA,
            },
        ),
        Stage(
            "cuped",
            "CUPED and Lin regression-adjusted estimates.",
            lambda s: run_cuped(s.db_path, s.reports_dir),
            inputs=lambda s: [s.parquet_path],
            outputs=lambda s: [
                *_tables(s, readout.CUPED_TABLE_NAME),
                *_figures(s, readout.CUPED_FIGURE_NAME),
            ],
            sections=("cuped",),
            modules=("readout.py", "frequentist.py", "cuped.py", "eda.py", "plots.py"),
            params=lambda s: {"alpha": config.ALPHA},
        ),
        Stage(
            "bayes",
            "Bayesian readouts: Beta-Binomial rates and bootstrap spend.",
            _run_bayes,
            inputs=lambda s: [s.parquet_path],
            outputs=lambda s: [
                *_tables(
                    s,
                    bayes.TABLE_NAME,
                    bayes.SPEND_TABLE_NAME,
                    bayes.SPEND_PPC_TABLE_NAME,
                    bayes.SPEND_HIST_TABLE_NAME,
                ),
                *_figures(s, bayes.FIGURE_NAME, bayes.SPEND_FIGURE_NAME),
            ],
            sections=("bayes_rates", "bayes_spend"),
            modules=("bayes.py", "frequentist.py", "eda.py", "plots.py"),
            params=lambda s: {
                "n_draws": s.sizes["bayes_draws"],
                "n_replicates": s.sizes["bayes_ppc_replicates"],
                "seed": _SEEDS["bayes"],
                "p_better": config.BAYES_P_BETTER,
                "loss_fraction": config.BAYES_LOSS_FRACTION,
                "alpha": config.ALPHA,
            },
        ),
        Stage(
            "power",
            "Minimum detectable effects, sample sizes and power curves.",
            lambda s: run_power(s.db_path, s.reports_dir),
            inputs=lambda s: [s.parquet_path],
            outputs=lambda s: [
                *_tables(s, "power", "power_sample_size", "power_curve"),
                *_figures(s, "power_curve.png"),
            ],
            sections=("power",),
            modules=("power.py", "cuped.py", "plots.py"),
            params=lambda s: {"alpha": config.ALPHA},
        ),
        Stage(
            "sequential",
            "Peeking simulation: naive, Bayesian, Pocock, O'Brien-Fleming, mSPRT.",
            _run_sequential,
            inputs=lambda s: [s.parquet_path],
            outputs=lambda s: [
                *_tables(
                    s,
                    sequential.TABLE_NAME,
                    sequential.LOOKS_TABLE_NAME,
                    sequential.BY_DAY_TABLE_NAME,
                    sequential.BOUNDARIES_TABLE_NAME,
                ),
                *_figures(
                    s,
                    sequential.FPR_FIGURE_NAME,
                    sequential.CUMULATIVE_FIGURE_NAME,
                    sequential.BOUNDARIES_FIGURE_NAME,
                ),
            ],
            sections=("sequential",),
            modules=("sequential.py", "power.py", "frequentist.py", "plots.py"),
            params=lambda s: {
                "n_runs": s.sizes["sequential_runs"],
                "n_runs_looks": s.sizes["sequential_runs_looks"],
                "n_paths": s.sizes["sequential_paths"],
                "seed": _SEEDS["simulation"],
                "alpha": config.ALPHA,
            },
        ),
        Stage(
            "segments",
            "Exploratory segment effects, interaction tests and BH.",
            lambda s: run_segments(
                s.db_path, s.reports_dir, n_sim_runs=s.sizes["segments_sim_runs"]
            ),
            inputs=lambda s: [s.parquet_path],
            outputs=lambda s: [
                *_tables(
                    s,
                    segments.TABLE_NAME,
                    segments.INTERACTIONS_TABLE_NAME,
                    segments.FDR_TABLE_NAME,
                ),
                *_figures(s, segments.FIGURE_NAME),
            ],
            sections=("segments",),
            modules=("segments.py", "frequentist.py", "plots.py"),
            params=lambda s: {
                "n_sim_runs": s.sizes["segments_sim_runs"],
                "seed": _SEEDS["simulation"] + 1,
                "alpha": config.ALPHA,
            },
        ),
        Stage(
            "uplift",
            "Cross-fitted uplift learners and baselines with evaluation.",
            _run_uplift,
            inputs=lambda s: [s.parquet_path],
            outputs=lambda s: [
                *_tables(s, uplift_models.TABLE_NAME, uplift_models.COMPARISON_TABLE_NAME),
                s.scores_path,
                *_uplift_figures(s),
            ],
            sections=("uplift_models",),
            modules=("uplift/*.py", "plots.py"),
            params=lambda s: {
                "n_resamples": s.sizes["uplift_resamples"],
                "outcomes": list(s.uplift_outcomes),
                "grid": list(s.grid),
                "seeds": {"uplift": _SEEDS["uplift"], "bootstrap": _SEEDS["bootstrap"]},
            },
        ),
        Stage(
            "policy",
            "Budgeted targeting policies, IPW values and the budget curve.",
            _run_policy,
            inputs=lambda s: [s.parquet_path, s.scores_path],
            outputs=lambda s: [
                *_tables(
                    s,
                    uplift_policy.TABLE_NAME,
                    uplift_policy.COMPARISON_TABLE_NAME,
                    uplift_policy.CURVE_TABLE_NAME,
                    uplift_policy.PROFILE_TABLE_NAME,
                    uplift_policy.CALIBRATION_TABLE_NAME,
                    uplift_policy.SUPPRESSED_TABLE_NAME,
                ),
                *_figures(s, uplift_policy.FIGURE_NAME),
            ],
            sections=("policy",),
            reads_sections=("frequentist", "bayes_spend"),
            modules=("uplift/*.py", "plots.py"),
            params=lambda s: {
                "n_resamples": s.sizes["policy_resamples"],
                "budget": budget_for(s),
                "cost_per_email": config.COST_PER_EMAIL,
                "arm_probability": config.ARM_PROBABILITY,
                "seeds": {"uplift": _SEEDS["uplift"], "bootstrap": _SEEDS["bootstrap"]},
            },
        ),
        Stage(
            "robustness",
            f"Repeat the visit cross-fitting over fold seeds (about {ROBUSTNESS_MINUTES} minutes "
            "on the real data; run-all includes it only with --with-robustness).",
            _run_robustness,
            inputs=lambda s: [s.parquet_path, s.scores_path],
            outputs=lambda s: _tables(s, uplift_robustness.TABLE_NAME),
            sections=("seed_robustness",),
            reads_sections=("frequentist", "bayes_spend"),
            modules=("uplift/*.py",),
            params=lambda s: {
                "n_seeds": s.sizes["robustness_seeds"],
                "budget": budget_for(s),
                "grid": list(s.grid),
                "seed": _SEEDS["uplift"],
            },
            optional=True,
        ),
        Stage(
            "report",
            "Write the pipeline section of results.json and validate the file against "
            "docs/results_schema.md (never cached).",
            _run_report,
            inputs=lambda s: [],
            outputs=lambda s: [s.results_path],
            sections=("pipeline",),
            cache=False,
        ),
    )
}


# ---------------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    """SHA-256 of a file's content, or ``"missing"``."""
    if not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _module_files(patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in (*CORE_MODULES, *patterns):
        if any(ch in pattern for ch in "*?["):
            files.extend(
                p
                for p in PACKAGE_DIR.rglob("*")
                if fnmatch.fnmatch(str(p.relative_to(PACKAGE_DIR)), pattern)
            )
        else:
            files.append(PACKAGE_DIR / pattern)
    return sorted(set(files))


def load_results(path: Path) -> dict[str, Any]:
    """Return the parsed ``results.json`` at ``path``, or an empty dict if it does not exist."""
    return json.loads(path.read_text()) if path.exists() else {}


def fingerprint(stage: Stage, settings: Settings) -> str:
    """Hash of everything that determines a stage's outputs.

    Parameters
    ----------
    stage : Stage
        The stage.
    settings : Settings
        Pipeline settings.

    Returns
    -------
    str
        SHA-256 over the package version, the content of the input files and the source
        files, the ``results.json`` sections the stage reads and its parameters.
    """
    results = load_results(settings.results_path)
    payload = {
        "abkit": __version__,
        "inputs": {p.name: _file_hash(p) for p in stage.inputs(settings)},
        "sections": {n: _json_hash(results.get(n)) for n in stage.reads_sections},
        "modules": {
            str(p.relative_to(PACKAGE_DIR)): _file_hash(p) for p in _module_files(stage.modules)
        },
        "params": stage.params(settings),
    }
    return _json_hash(payload)


def load_cache(path: Path) -> dict[str, Any]:
    """Return the stage cache at ``path`` (empty if missing or unreadable)."""
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        logger.warning("Ignoring unreadable cache %s.", path)
        return {}


def save_cache(cache: dict[str, Any], path: Path) -> None:
    """Write the stage cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def is_cached(stage: Stage, settings: Settings, cache: dict[str, Any], fp: str) -> bool:
    """Whether a stage can be skipped: same fingerprint, outputs present, sections present."""
    entry = cache.get(stage.name)
    if not stage.cache or settings.force or entry is None or entry.get("fingerprint") != fp:
        return False
    if not all(p.exists() for p in stage.outputs(settings)):
        return False
    results = load_results(settings.results_path)
    return all(section in results for section in stage.sections)


@dataclass(frozen=True)
class StageOutcome:
    """What happened to one stage in a run."""

    name: str
    status: str  # "ran" or "cached"
    seconds: float


def run_stage(stage: Stage, settings: Settings, cache: dict[str, Any]) -> StageOutcome:
    """Run one stage unless its cache entry is valid, and record the result in ``cache``.

    Parameters
    ----------
    stage : Stage
        The stage.
    settings : Settings
        Pipeline settings.
    cache : dict
        The loaded stage cache; updated in place and written after the stage.

    Returns
    -------
    StageOutcome
        Status and runtime (the recorded runtime for a cached stage).
    """
    fp = fingerprint(stage, settings)
    if is_cached(stage, settings, cache, fp):
        previous = float(cache[stage.name].get("seconds", 0.0))
        logger.info("Stage %-12s cached: inputs, code and settings unchanged.", stage.name)
        return StageOutcome(stage.name, "cached", previous)
    logger.info("Stage %-12s running (%s mode)", stage.name, settings.mode)
    start = time.perf_counter()
    stage.run(settings)
    seconds = time.perf_counter() - start
    if stage.cache:
        cache[stage.name] = {
            "fingerprint": fp,
            "mode": settings.mode,
            "seconds": round(seconds, 2),
            "finished_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        save_cache(cache, settings.cache_path)
    logger.info("Stage %-12s done in %.1f s", stage.name, seconds)
    return StageOutcome(stage.name, "ran", seconds)


def run(names: list[str], settings: Settings) -> list[StageOutcome]:
    """Run the named stages in order and log a summary table.

    Parameters
    ----------
    names : list of str
        Stage names (``run-all`` expands to :func:`stages_in_run`).
    settings : Settings
        Pipeline settings.

    Returns
    -------
    list of StageOutcome
        One outcome per stage, in order.
    """
    settings.ensure_dirs()
    cache = load_cache(settings.cache_path)
    start = time.perf_counter()
    outcomes = [run_stage(STAGES[name], settings, cache) for name in names]
    total = time.perf_counter() - start
    lines = [f"  {o.name:<12} {o.status:<7} {o.seconds:8.1f} s" for o in outcomes]
    ran = sum(o.seconds for o in outcomes if o.status == "ran")
    logger.info(
        "Pipeline summary (%s mode, data: %s):\n%s\n  total wall time %.1f s, of which %.1f s "
        "in stages that ran (%d ran, %d cached)",
        settings.mode,
        settings.data_label,
        "\n".join(lines),
        total,
        ran,
        sum(o.status == "ran" for o in outcomes),
        sum(o.status == "cached" for o in outcomes),
    )
    return outcomes


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        One positional command (a stage name or ``run-all``) plus the options.
    """
    stage_help = "\n".join(f"  {name:<12} {st.help}" for name, st in STAGES.items())
    parser = argparse.ArgumentParser(
        prog="python -m abkit.pipeline",
        description=(
            "Run the Hillstrom experimentation pipeline, one stage or all of them.\n\n"
            "Stages, in dependency order:\n"
            f"{stage_help}\n"
            "  run-all      Every stage above from data to report, skipping stages whose\n"
            "               inputs, code and settings are unchanged since their last run.\n"
            f"               The robustness stage (about {ROBUSTNESS_MINUTES} minutes) is left "
            "out unless\n               --with-robustness is given."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=[*STAGES, "run-all"],
        metavar="<stage> | run-all",
        help="stage to run, or run-all",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "fewer simulations, bootstrap draws and seeds (config.FAST_SIZES), visit only and "
            "one hyperparameter point in the uplift stage; for CI and quick checks"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        metavar="PATH",
        help=(
            "CSV to load instead of downloading the Hillstrom file, e.g. the synthetic fixture "
            "tests/fixtures/hillstrom_fixture.csv (the row count is not checked and the email "
            "budget is scaled to the rows)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.ROOT,
        metavar="DIR",
        help="write data/ and reports/ under this directory (default: the project root)",
    )
    parser.add_argument(
        "--with-robustness",
        action="store_true",
        help=f"include the seed-robustness stage in run-all (about {ROBUSTNESS_MINUTES} minutes)",
    )
    parser.add_argument(
        "--force", action="store_true", help="ignore the stage cache and rerun every stage asked"
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Build :class:`Settings` from parsed arguments."""
    return Settings(
        fast=args.fast,
        data=args.data.resolve() if args.data is not None else None,
        out=args.out.resolve(),
        with_robustness=args.with_robustness,
        force=args.force,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested command.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 when a stage fails one of its checks.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)
    if settings.data is not None and not settings.data.is_file():
        logger.error("--data file not found: %s", settings.data)
        return 2
    names = stages_in_run(settings) if args.command == "run-all" else [args.command]
    try:
        run(names, settings)
    except (DataError, ValidationError, SRMError, ResultsSchemaError) as exc:
        logger.error("Pipeline stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
