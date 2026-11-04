import abkit
from abkit import config
from abkit.pipeline import STEPS, build_parser


def test_version():
    assert abkit.__version__ == "0.1.0"


def test_arm_labels_consistent():
    assert set(config.RAW_ARM_MAP.values()) == set(config.ARMS)
    assert config.CONTROL in config.ARMS
    assert config.CONTROL not in config.TREATMENTS


def test_outcomes_are_not_pre_treatment():
    assert set(config.PRE_TREATMENT_COLS).isdisjoint(config.OUTCOME_COLS)


def test_budget():
    assert config.BUDGET == 10_000


def test_pipeline_commands():
    parser = build_parser()
    steps = ["data", "validate", "checks", "frequentist", "cuped", "bayes", "power"]
    for command in (*steps, "run-all"):
        assert parser.parse_args([command]).command == command
    assert list(STEPS) == steps
