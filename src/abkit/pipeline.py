"""Command-line entry point: ``python -m abkit.pipeline run-all``."""

import argparse
import logging
import sys
from collections.abc import Callable

from abkit import config
from abkit.bayes import run_bayes_rates, run_bayes_spend
from abkit.data import download_raw, load_to_duckdb
from abkit.eda import run_checks
from abkit.power import run_power
from abkit.readout import run_cuped, run_frequentist
from abkit.segments import run_segments
from abkit.sequential import run_sequential
from abkit.validate import run_validation

logger = logging.getLogger(__name__)


def run_data() -> None:
    """Download (or verify) the raw file and load it into DuckDB."""
    load_to_duckdb(download_raw())


def run_validate() -> None:
    """Validate ``raw_customers``, write the reports and build ``customers``.

    Raises
    ------
    abkit.validate.ValidationError
        If any critical check fails.
    """
    run_validation()


def run_validity() -> None:
    """Run the SRM and balance checks and the pooled EDA, writing tables, figures and eda.md.

    Raises
    ------
    abkit.eda.SRMError
        If the sample ratio mismatch check fails.
    """
    run_checks()


def run_readout() -> None:
    """Run the frequentist readout, writing its table, results.json section and forest plot.

    Raises
    ------
    abkit.eda.SRMError
        If the sample ratio mismatch check fails.
    """
    run_frequentist()


def run_adjusted() -> None:
    """Run CUPED and the Lin regression adjustment, writing their table, section and figure.

    Raises
    ------
    abkit.eda.SRMError
        If the sample ratio mismatch check fails.
    """
    run_cuped()


def run_bayesian() -> None:
    """Run the Bayesian readouts of the rates and of spend, writing tables, sections and figures.

    Raises
    ------
    abkit.eda.SRMError
        If the sample ratio mismatch check fails.
    """
    run_bayes_rates()
    run_bayes_spend()


def run_design() -> None:
    """Run the MDE, sample-size and power-curve calculations and write their outputs."""
    run_power()


def run_peeking() -> None:
    """Run the peeking simulation study and write its tables, figures and results section."""
    run_sequential()


def run_subgroups() -> None:
    """Run the exploratory segment analysis and write its tables, figure and results section."""
    run_segments()


def run_all() -> None:
    """Run every pipeline stage end to end."""
    for name, step in STEPS.items():
        logger.info("Running step: %s", name)
        step()


STEPS: dict[str, Callable[[], None]] = {
    "data": run_data,
    "validate": run_validate,
    "checks": run_validity,
    "frequentist": run_readout,
    "cuped": run_adjusted,
    "bayes": run_bayesian,
    "power": run_design,
    "sequential": run_peeking,
    "segments": run_subgroups,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with one subcommand per pipeline stage plus ``run-all``.
    """
    parser = argparse.ArgumentParser(prog="abkit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="Download or verify the raw file and load it into DuckDB.")
    sub.add_parser("validate", help="Validate the raw table and build the cleaned table.")
    sub.add_parser("checks", help="SRM and balance checks plus pooled EDA.")
    sub.add_parser("frequentist", help="Frequentist readout of every comparison and metric.")
    sub.add_parser("cuped", help="CUPED and Lin regression-adjusted estimates.")
    sub.add_parser("bayes", help="Bayesian readouts: Beta-Binomial rates and bootstrap spend.")
    sub.add_parser("power", help="Minimum detectable effects, sample sizes and power curves.")
    sub.add_parser(
        "sequential", help="Peeking simulation: naive, Bayesian, Pocock, O'Brien-Fleming, mSPRT."
    )
    sub.add_parser("segments", help="Exploratory segment effects, interaction tests and BH.")
    sub.add_parser("run-all", help="Run every pipeline stage end to end.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested command.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    config.ensure_dirs()
    if args.command == "run-all":
        run_all()
    else:
        STEPS[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
