"""Project-wide constants: paths, arm labels, significance level, seeds and budget."""

from pathlib import Path

# Paths
ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DUCKDB_PATH: Path = DATA_PROCESSED / "hillstrom.duckdb"
REPORTS: Path = ROOT / "reports"
TABLES: Path = REPORTS / "tables"
FIGURES: Path = REPORTS / "figures"
RESULTS_JSON: Path = REPORTS / "results.json"

# Data source
DATA_URL: str = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
RAW_CSV: Path = DATA_RAW / "hillstrom.csv"
DATA_SHA256: str | None = None  # pinned after the first verified download

# Treatment arms (raw column `segment` is renamed to `arm` during cleaning)
RAW_ARM_COL: str = "segment"
ARM_COL: str = "arm"
RAW_ARM_MAP: dict[str, str] = {
    "Mens E-Mail": "mens",
    "Womens E-Mail": "womens",
    "No E-Mail": "control",
}
CONTROL: str = "control"
TREATMENTS: tuple[str, ...] = ("mens", "womens")
ARMS: tuple[str, ...] = (*TREATMENTS, CONTROL)

# Columns
PRE_TREATMENT_COLS: tuple[str, ...] = (
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
)
OUTCOME_COLS: tuple[str, ...] = ("visit", "conversion", "spend")

# Decision settings
ALPHA: float = 0.05
BUDGET: int = 10_000

# Seeds (all randomness flows through numpy.random.Generator built from these)
SEED: int = 20080320
SEEDS: dict[str, int] = {
    "split": SEED + 1,
    "bootstrap": SEED + 2,
    "simulation": SEED + 3,
    "uplift": SEED + 4,
    "bayes": SEED + 5,
}


def ensure_dirs() -> None:
    """Create the local data and report directories if they do not exist."""
    for path in (DATA_RAW, DATA_PROCESSED, TABLES, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
