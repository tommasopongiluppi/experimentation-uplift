"""Pipeline CLI: stage registry, fast sizes, fingerprints, cache and the end-to-end smoke run."""

import json
import time

import pytest

from abkit import config
from abkit.pipeline import (
    FULL_SIZES,
    STAGES,
    Settings,
    StageOutcome,
    budget_for,
    build_parser,
    fingerprint,
    is_cached,
    main,
    run,
    settings_from_args,
    stages_in_run,
)
from abkit.results_schema import OPTIONAL_SECTIONS, SCHEMA, check_results
from tests.fixtures.make_fixture import FIXTURE_PATH

EXPECTED_STAGES = [
    "data",
    "validate",
    "checks",
    "frequentist",
    "cuped",
    "bayes",
    "power",
    "sequential",
    "segments",
    "uplift",
    "policy",
    "robustness",
    "report",
]


def test_stage_registry_order_and_optional_flag():
    assert list(STAGES) == EXPECTED_STAGES
    assert STAGES["robustness"].optional and not STAGES["report"].cache
    assert stages_in_run(Settings()) == [s for s in EXPECTED_STAGES if s != "robustness"]
    assert stages_in_run(Settings(with_robustness=True)) == EXPECTED_STAGES
    # Every section of the schema is written by exactly one stage.
    written = [sec for st in STAGES.values() for sec in st.sections]
    assert sorted(written) == sorted(SCHEMA)


def test_fast_sizes_cover_every_full_size_and_are_smaller():
    assert set(config.FAST_SIZES) == set(FULL_SIZES)
    for key, fast in config.FAST_SIZES.items():
        assert 0 < fast < FULL_SIZES[key], key
    fast, full = Settings(fast=True), Settings()
    assert fast.sizes == config.FAST_SIZES and full.sizes == FULL_SIZES
    assert fast.uplift_outcomes == ("visit",) and len(fast.grid) == 1


def test_parser_and_settings(tmp_path):
    parser = build_parser()
    for command in (*EXPECTED_STAGES, "run-all"):
        assert parser.parse_args([command]).command == command
    args = parser.parse_args(
        ["run-all", "--fast", "--data", str(FIXTURE_PATH), "--out", str(tmp_path), "--force"]
    )
    s = settings_from_args(args)
    assert s.fast and s.force and not s.with_robustness
    assert s.data == FIXTURE_PATH.resolve() and s.out == tmp_path.resolve()
    assert s.db_path == tmp_path.resolve() / "data" / config.DUCKDB_PATH.name
    assert s.results_path == tmp_path.resolve() / "reports" / "results.json"
    assert s.data_label == FIXTURE_PATH.name and Settings().data_label == "hillstrom"
    with pytest.raises(SystemExit):
        parser.parse_args(["not-a-stage"])


def test_missing_data_file_is_an_error(tmp_path):
    assert main(["data", "--data", str(tmp_path / "nope.csv"), "--out", str(tmp_path)]) == 2


def test_fingerprint_tracks_inputs_params_and_code(tmp_path):
    settings = Settings(data=FIXTURE_PATH, out=tmp_path)
    stage = STAGES["data"]
    before = fingerprint(stage, settings)
    assert before == fingerprint(stage, settings)  # deterministic
    # Another input file with other content.
    other = tmp_path / "other.csv"
    other.write_text(FIXTURE_PATH.read_text().replace("Mens E-Mail", "Womens E-Mail", 1))
    assert fingerprint(stage, Settings(data=other, out=tmp_path)) != before
    # Settings that matter change the fingerprint of the stages they feed.
    assert fingerprint(STAGES["bayes"], Settings(out=tmp_path)) != fingerprint(
        STAGES["bayes"], Settings(out=tmp_path, fast=True)
    )
    # A missing output or section is never a cache hit, even with a matching fingerprint.
    cache = {"data": {"fingerprint": before}}
    assert not is_cached(stage, settings, cache, before)
    forced = Settings(data=FIXTURE_PATH, out=tmp_path, force=True)
    assert not is_cached(stage, forced, cache, before)


def _assert_outputs_exist(settings: Settings, names: list[str]) -> None:
    for name in names:
        for path in STAGES[name].outputs(settings):
            assert path.exists(), f"{name}: {path} missing"


def test_run_all_fast_on_fixture_end_to_end(tmp_path):
    """The CI smoke test: every stage but robustness, on the fixture, in fast mode."""
    start = time.perf_counter()
    assert main(["run-all", "--fast", "--data", str(FIXTURE_PATH), "--out", str(tmp_path)]) == 0
    first = time.perf_counter() - start
    settings = Settings(fast=True, data=FIXTURE_PATH.resolve(), out=tmp_path.resolve())
    names = stages_in_run(settings)
    _assert_outputs_exist(settings, names)
    results = check_results(settings.results_path, frozenset(SCHEMA) - OPTIONAL_SECTIONS)
    assert "seed_robustness" not in results
    assert results["pipeline"]["mode"] == "fast"
    assert results["pipeline"]["data"] == FIXTURE_PATH.name
    assert results["pipeline"]["stages"] == names
    assert results["pipeline"]["sizes"] == config.FAST_SIZES
    assert (
        results["frequentist"]["bootstrap"]["n_resamples"]
        == config.FAST_SIZES["frequentist_resamples"]
    )
    assert results["uplift_models"]["outcomes"] == ["visit"]
    assert len(results["uplift_models"]["hyperparameter_grid"]) == 1
    # The email budget scales with the row count of the fixture.
    expected_budget = round(config.BUDGET * 3000 / config.EXPECTED_ROWS)
    assert budget_for(settings) == expected_budget
    assert results["policy"]["budget"] == expected_budget
    # Every cached stage has an entry; the report stage is never cached.
    cache = json.loads(settings.cache_path.read_text())
    assert set(cache) == set(names) - {"report"}
    assert all(entry["mode"] == "fast" for entry in cache.values())

    # Second run: nothing changed, so every stage but report is skipped.
    start = time.perf_counter()
    outcomes = run(names, settings)
    second = time.perf_counter() - start
    assert [o.status for o in outcomes] == ["cached"] * (len(names) - 1) + ["ran"]
    assert isinstance(outcomes[0], StageOutcome)
    assert second < first / 4

    # Removing an output invalidates only that stage; later stages whose inputs are the
    # cleaned table stay cached.
    STAGES["power"].outputs(settings)[0].unlink()
    outcomes = {o.name: o.status for o in run(names, settings)}
    assert outcomes["power"] == "ran" and outcomes["bayes"] == "cached"
    # --force reruns a single stage on request.
    forced = run(["cuped"], Settings(fast=True, data=settings.data, out=settings.out, force=True))
    assert forced[0].status == "ran"


@pytest.mark.slow
def test_run_all_fast_with_robustness_on_fixture(tmp_path):
    argv = ["run-all", "--fast", "--with-robustness", "--data", str(FIXTURE_PATH)]
    assert main([*argv, "--out", str(tmp_path)]) == 0
    settings = Settings(
        fast=True, data=FIXTURE_PATH.resolve(), out=tmp_path.resolve(), with_robustness=True
    )
    results = check_results(settings.results_path, frozenset(SCHEMA))
    assert results["seed_robustness"]["n_seeds"] == config.FAST_SIZES["robustness_seeds"]
    assert results["pipeline"]["with_robustness"] is True
