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
SRM_ALPHA: float = 0.001  # analysis plan §5: SRM p below this blocks the readout
SMD_THRESHOLD: float = 0.1  # analysis plan §5: |SMD| above this is flagged
BUDGET: int = 10_000
# Design assignment probability per arm (three equal arms); the policy value uses it as the
# known propensity in inverse propensity weighting, never an estimated one.
ARM_PROBABILITY: float = 1.0 / 3.0
# Cost of sending one email, in the same currency as spend. Zero by default because the
# challenge gives no cost; set it to see each policy's net value (incremental spend minus
# cost x emails sent).
COST_PER_EMAIL: float = 0.0
# Bayesian decision rule (analysis plan §9): ship if P(better) >= BAYES_P_BETTER and the expected
# loss of shipping is below BAYES_LOSS_FRACTION of the control arm's mean.
BAYES_P_BETTER: float = 0.95
BAYES_LOSS_FRACTION: float = 0.01

# Figures: categorical slots 1-3 of the reference palette, fixed per arm across the project
ARM_COLORS: dict[str, str] = {"mens": "#2a78d6", "womens": "#eb6834", "control": "#1baf7a"}

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


# Pipeline: --fast replaces the full simulation sizes (the module constants N_DRAWS, N_RUNS and
# so on) with these, for CI and quick checks. The sizes used are recorded in results.json.
FAST_SIZES: dict[str, int] = {
    "frequentist_resamples": 1_000,
    "bayes_draws": 20_000,
    "bayes_ppc_replicates": 500,
    "sequential_runs": 200,
    "sequential_runs_looks": 100,
    "sequential_paths": 2_000,
    "segments_sim_runs": 200,
    "uplift_resamples": 30,
    "policy_resamples": 40,
    "robustness_seeds": 3,
}
# Stage cache: fingerprints of each stage's inputs, code and settings (gitignored with data/).
PIPELINE_CACHE: Path = DATA_DIR / "pipeline_cache.json"
