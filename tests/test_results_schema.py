"""results.json schema: the committed file matches it, the doc is in sync, problems are named."""

import json

import pytest

from abkit import config
from abkit.results_schema import (
    DOC_PATH,
    OPTIONAL_SECTIONS,
    SCHEMA,
    ResultsSchemaError,
    _check_type,
    check_results,
    render_markdown,
    validate_results,
)


def test_committed_results_match_schema():
    results = check_results(config.RESULTS_JSON)
    assert set(results) >= set(SCHEMA) - OPTIONAL_SECTIONS


def test_doc_is_rendered_from_the_schema():
    assert DOC_PATH.read_text() == render_markdown()


@pytest.mark.parametrize(
    ("value", "type_", "ok"),
    [
        (1, "int", True),
        (True, "int", False),
        (1, "float", True),
        (1.5, "int", False),
        (None, "str|null", True),
        (None, "str", False),
        ("x", "str|null", True),
        ([1.0, 2], "list[float]", True),
        ([1, "a"], "list[int]", False),
        ({"a": 0.1}, "dict[str, float]", True),
        ({"a": "b"}, "dict[str, bool]", False),
        ([{"a": 1}], "list[dict]", True),
    ],
)
def test_check_type(value, type_, ok):
    assert _check_type(value, type_) is ok


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown schema type"):
        _check_type(1, "complex")


def test_validate_results_names_every_problem(tmp_path):
    results = json.loads(config.RESULTS_JSON.read_text())
    results["validity"]["srm"]["p_value"] = "0.9"  # wrong type
    del results["eda"]["pooled_rates"]  # missing key
    results["frequentist"]["figure"] = "figures/does_not_exist.png"  # dangling path
    results["mystery"] = {}  # unknown section
    del results["power"]  # missing section
    problems = validate_results(results, config.REPORTS)
    assert any(p.startswith("validity.srm.p_value: expected float") for p in problems)
    assert "eda.pooled_rates: missing" in problems
    assert any(p.startswith("frequentist.figure: file") for p in problems)
    assert "mystery: unknown section" in problems
    assert "power: section missing" in problems
    assert len(problems) == 5
    # check_results resolves paths from the file's directory, so every table is missing too.
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))
    with pytest.raises(ResultsSchemaError, match=r"does not match docs/results_schema\.md"):
        check_results(path)


def test_segment_families_are_validated_record_by_record():
    results = json.loads(config.RESULTS_JSON.read_text())
    families = results["segments"]["families"]
    assert set(families) == {"segments_spend", "segments_visit", "segments_conversion"}
    for counts in families.values():
        assert counts["n_tests"] >= counts["n_significant_raw"] >= counts["n_significant_fdr"]
    del families["segments_spend"]["n_significant_raw"]  # missing key inside one family
    families["segments_visit"]["n_significant_fdr"] = 47.5  # wrong type inside another
    families["segments_conversion"] = 36  # a family that is not a record
    assert validate_results(results, config.REPORTS) == [
        "segments.families.segments_spend.n_significant_raw: missing",
        "segments.families.segments_visit.n_significant_fdr: expected int, got float",
        "segments.families.segments_conversion: expected dict, got int",
    ]


def test_optional_section_is_validated_only_when_present():
    results = json.loads(config.RESULTS_JSON.read_text())
    results.pop("seed_robustness", None)
    assert validate_results(results, config.REPORTS) == []
    results["seed_robustness"] = {"label": 1}
    problems = validate_results(results, config.REPORTS)
    assert "seed_robustness.label: expected str, got int" in problems
    assert validate_results(results, config.REPORTS, frozenset({"seed_robustness"}))
