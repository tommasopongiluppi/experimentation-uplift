"""Project-wide constants: paths, arm labels, significance level, seeds and budget."""

from pathlib import Path

# Paths
ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DUCKDB_PATH: Path = DATA_DIR / "hillstrom.duckdb"
REPORTS: Path = ROOT / "reports"
TABLES: Path = REPORTS / "tables"
FIGURES: Path = REPORTS / "figures"
RESULTS_JSON: Path = REPORTS / "results.json"

# Data source (plain http: the original server redirects https to http)
DATA_URL: str = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
MIRROR_URL: str = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"
# SHA-256 of files verified at first download; a mismatch on re-download fails loudly.
KNOWN_SHA256: dict[str, str] = {
    DATA_URL: "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece",
    MIRROR_URL: "bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f",
}
MANIFEST_NAME: str = "manifest.json"
EXPECTED_ROWS: int = 64_000

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
    "fixture": SEED + 6,
}


def ensure_dirs() -> None:
    """Create the local data and report directories if they do not exist."""
    for path in (DATA_RAW, DATA_PROCESSED, TABLES, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
