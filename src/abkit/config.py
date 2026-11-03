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
BINARY_COLS: tuple[str, ...] = ("mens", "womens", "newbie", "visit", "conversion")
CATEGORICAL_COLS: tuple[str, ...] = ("history_segment", "zip_code", "channel", RAW_ARM_COL)

# Cleaning: canonical raw label -> lower_snake_case value, per categorical column
CATEGORY_MAPS: dict[str, dict[str, str]] = {
    "history_segment": {
        "1) $0 - $100": "usd_0_100",
        "2) $100 - $200": "usd_100_200",
        "3) $200 - $350": "usd_200_350",
        "4) $350 - $500": "usd_350_500",
        "5) $500 - $750": "usd_500_750",
        "6) $750 - $1,000": "usd_750_1000",
        "7) $1,000 +": "usd_1000_plus",
    },
    "zip_code": {"Urban": "urban", "Suburban": "suburban", "Rural": "rural"},
    "channel": {"Phone": "phone", "Web": "web", "Multichannel": "multichannel"},
    RAW_ARM_COL: RAW_ARM_MAP,
}
# Known misspellings in the source file -> canonical raw label
LABEL_MISSPELLINGS: dict[str, dict[str, str]] = {
    "zip_code": {"Surburban": "Suburban"},
}
# history_segment dollar bands, lower bound inclusive and upper bound exclusive (as in the data)
HISTORY_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 100.0, "usd_0_100"),
    (100.0, 200.0, "usd_100_200"),
    (200.0, 350.0, "usd_200_350"),
    (350.0, 500.0, "usd_350_500"),
    (500.0, 750.0, "usd_500_750"),
    (750.0, 1000.0, "usd_750_1000"),
    (1000.0, float("inf"), "usd_1000_plus"),
)
# Population: customers who purchased in the last 12 months, so recency should lie in [1, 12]
RECENCY_RANGE: tuple[int, int] = (1, 12)

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
