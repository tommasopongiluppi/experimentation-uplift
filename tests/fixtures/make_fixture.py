"""Generate a synthetic Hillstrom-shaped dataset with planted treatment effects.

No real rows are used. Every value is drawn from simple distributions that only mimic the
schema and value domains of the real file (same 12 columns, same category labels, CRLF line
endings). Pre-treatment columns are drawn independently of the arm, as under randomisation.

Planted effects are set on the unconditional per-arm rates, so the true average treatment
effects are known exactly:

    P(conversion = 1 | arm) = conversion_rate[arm]
    P(visit = 1 | arm)      = visit_rate[arm]
    E[spend | arm]          = conversion_rate[arm] * spend_per_conversion[arm]

Generation, per customer in arm a (requires visit_rate[a] >= conversion_rate[a]):

    conversion ~ Bernoulli(c_a)
    visit      = 1 if conversion else Bernoulli((v_a - c_a) / (1 - c_a))
    spend      = conversion * Gamma(shape=2, scale=m_a / 2), rounded to cents

so every converter also visited and only converters spend. The default effects are larger
than in the real experiment so they are detectable in about 3,000 rows.

Usage::

    python tests/fixtures/make_fixture.py              # writes hillstrom_fixture.csv
    python tests/fixtures/make_fixture.py --n 500 --seed 7 --out /tmp/small.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from abkit import config

FIXTURE_PATH = Path(__file__).with_name("hillstrom_fixture.csv")

RAW_ARM_LABELS: dict[str, str] = {v: k for k, v in config.RAW_ARM_MAP.items()}

VISIT_RATE: dict[str, float] = {"control": 0.10, "mens": 0.20, "womens": 0.16}
CONVERSION_RATE: dict[str, float] = {"control": 0.010, "mens": 0.030, "womens": 0.020}
SPEND_PER_CONVERSION: dict[str, float] = {"control": 110.0, "mens": 120.0, "womens": 115.0}

HISTORY_BINS = [0.0, 100.0, 200.0, 350.0, 500.0, 750.0, 1000.0, np.inf]
HISTORY_LABELS = [
    "1) $0 - $100",
    "2) $100 - $200",
    "3) $200 - $350",
    "4) $350 - $500",
    "5) $500 - $750",
    "6) $750 - $1,000",
    "7) $1,000 +",
]
SPEND_GAMMA_SHAPE = 2.0


def true_effects(
    visit_rate: dict[str, float] = VISIT_RATE,
    conversion_rate: dict[str, float] = CONVERSION_RATE,
    spend_per_conversion: dict[str, float] = SPEND_PER_CONVERSION,
) -> dict[str, dict[str, float]]:
    """Return the planted average treatment effect of each email vs control, per outcome."""
    mean_spend = {a: conversion_rate[a] * spend_per_conversion[a] for a in config.ARMS}
    return {
        arm: {
            "visit": visit_rate[arm] - visit_rate[config.CONTROL],
            "conversion": conversion_rate[arm] - conversion_rate[config.CONTROL],
            "spend": mean_spend[arm] - mean_spend[config.CONTROL],
        }
        for arm in config.TREATMENTS
    }


def make_fixture(
    rng: np.random.Generator,
    n: int = 3000,
    visit_rate: dict[str, float] = VISIT_RATE,
    conversion_rate: dict[str, float] = CONVERSION_RATE,
    spend_per_conversion: dict[str, float] = SPEND_PER_CONVERSION,
) -> pd.DataFrame:
    """Draw a synthetic dataset with the raw Hillstrom schema and planted effects.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    n : int
        Number of rows.
    visit_rate, conversion_rate, spend_per_conversion : dict
        Per-arm planted values, keyed by "mens", "womens", "control".

    Returns
    -------
    pandas.DataFrame
        Columns in the raw file's order, with raw arm labels in ``segment``.
    """
    for arm in config.ARMS:
        if not 0 <= conversion_rate[arm] <= visit_rate[arm] < 1:
            raise ValueError(f"Need 0 <= conversion_rate <= visit_rate < 1 for arm {arm!r}.")

    arm = rng.choice(np.array(config.ARMS), size=n)
    history = np.round(29.99 + rng.lognormal(mean=np.log(150.0), sigma=0.9, size=n), 2)
    purchase_type = rng.choice(3, size=n, p=[0.45, 0.45, 0.10])  # mens only, womens only, both

    v = np.array([visit_rate[a] for a in arm])
    c = np.array([conversion_rate[a] for a in arm])
    m = np.array([spend_per_conversion[a] for a in arm])
    conversion = rng.random(n) < c
    visit = conversion | (rng.random(n) < (v - c) / (1 - c))
    spend = np.where(
        conversion, np.round(rng.gamma(SPEND_GAMMA_SHAPE, m / SPEND_GAMMA_SHAPE), 2), 0.0
    )

    return pd.DataFrame(
        {
            "recency": rng.integers(1, 13, size=n),
            "history_segment": pd.cut(history, HISTORY_BINS, labels=HISTORY_LABELS).astype(str),
            "history": history,
            "mens": (purchase_type != 1).astype(int),
            "womens": (purchase_type != 0).astype(int),
            "zip_code": rng.choice(["Surburban", "Urban", "Rural"], size=n, p=[0.45, 0.40, 0.15]),
            "newbie": rng.integers(0, 2, size=n),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], size=n, p=[0.44, 0.44, 0.12]),
            "segment": [RAW_ARM_LABELS[a] for a in arm],
            "visit": visit.astype(int),
            "conversion": conversion.astype(int),
            "spend": spend,
        }
    )


def write_fixture(df: pd.DataFrame, path: Path = FIXTURE_PATH) -> Path:
    """Write the fixture as CSV with CRLF line endings, like the original file."""
    df.to_csv(path, index=False, lineterminator="\r\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=config.SEEDS["fixture"])
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args()
    write_fixture(make_fixture(np.random.default_rng(args.seed), n=args.n), args.out)


if __name__ == "__main__":
    main()
