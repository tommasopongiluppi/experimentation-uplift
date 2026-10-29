"""Command-line entry point: ``python -m abkit.pipeline run-all``."""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with one subcommand per pipeline stage.
    """
    parser = argparse.ArgumentParser(prog="abkit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run-all", help="Run every pipeline stage end to end.")
    return parser


def run_all() -> None:
    """Run every pipeline stage end to end."""
    logger.info("No pipeline stages implemented yet.")


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
    if args.command == "run-all":
        run_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
