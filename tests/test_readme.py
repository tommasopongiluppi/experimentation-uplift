"""The README's generated blocks: marker handling, freshness against results.json, dollar signs."""

import json
import re
from pathlib import Path

import pytest

from abkit import config
from abkit.readme import build_blocks, fixed, join_words, p_value, pp, probability, usd
from abkit.report import ReadmeError, block_names, main, render_blocks, update_readme

DOCS = [
    config.ROOT / "docs" / name
    for name in ("analysis_plan.md", "data_card.md", "methodology.md", "results_schema.md")
]


def test_render_blocks_replaces_only_marked_text():
    text = (
        "Intro.\n<!-- RESULTS:a -->\nold a\n<!-- /RESULTS:a -->\nBetween.\n"
        "<!-- RESULTS:b -->\n<!-- /RESULTS:b -->\nEnd.\n"
    )
    assert block_names(text) == ["a", "b"]
    out = render_blocks(text, {"a": "new a\n", "b": "line 1\nline 2"})
    assert out == (
        "Intro.\n<!-- RESULTS:a -->\nnew a\n<!-- /RESULTS:a -->\nBetween.\n"
        "<!-- RESULTS:b -->\nline 1\nline 2\n<!-- /RESULTS:b -->\nEnd.\n"
    )
    assert render_blocks(out, {"a": "new a", "b": "line 1\nline 2"}) == out  # idempotent


def test_render_blocks_rejects_unmatched_markers_and_blocks():
    text = "<!-- RESULTS:a -->\n<!-- /RESULTS:a -->\n"
    with pytest.raises(ReadmeError, match="markers without a block"):
        render_blocks(text, {})
    with pytest.raises(ReadmeError, match="blocks without a marker"):
        render_blocks(text, {"a": "x", "b": "y"})


def test_formatting_helpers():
    assert usd(0.7698, signed=True) == "+\\$0.77"
    assert usd(-1234.5, 0) == "-\\$1,234"
    assert pp(0.0766, signed=True) == "+7.7 pp"
    assert p_value(0.0004) == "<0.001" and p_value(0.0305) == "0.030"
    assert fixed(-0.0004) == "0.000" and fixed(-0.0006) == "-0.001"
    assert join_words(["a"]) == "a" and join_words(["a", "b", "c"]) == "a, b and c"
    # A simulated probability is never printed as certainty.
    assert probability(1.0) == "> 0.999" and probability(0.99947) == "> 0.999"
    assert probability(0.985175) == "= 0.985" and probability(0.0) == "< 0.001"


def test_readme_is_up_to_date_with_results_json():
    """Fails when README.md differs from what results.json generates."""
    assert update_readme(config.README, config.RESULTS_JSON, check=True), (
        "README.md is out of date: run python -m abkit.report update-readme"
    )
    assert main(["update-readme", "--check"]) == 0


def test_stale_readme_is_detected_and_rewritten(tmp_path):
    results = json.loads(config.RESULTS_JSON.read_text())
    blocks = build_blocks(results, config.REPORTS)
    current = config.README.read_text()
    assert set(block_names(current)) == set(blocks)
    # A changed result changes the generated text, so the committed README no longer matches.
    row = next(
        r
        for r in results["frequentist"]["rows"]
        if r["metric"] == "spend" and r["comparison"] == "mens_vs_control"
    )
    row["diff"] += 0.25
    changed = build_blocks(results, config.REPORTS)
    assert changed["summary"] != blocks["summary"]
    stale = tmp_path / "README.md"
    stale.write_text(render_blocks(current, changed))
    assert main(["update-readme", "--check", "--readme", str(stale)]) == 1
    assert stale.read_text() != current  # --check does not write
    assert main(["update-readme", "--readme", str(stale)]) == 0
    assert stale.read_text() == current


def _outside_code(text: str) -> str:
    """Markdown with fenced blocks and inline code removed."""
    text = re.sub(r"^```.*?^```", "", text, flags=re.DOTALL | re.MULTILINE)
    return re.sub(r"`[^`\n]*`", "", text)


@pytest.mark.parametrize("path", [config.README, *DOCS], ids=lambda p: p.name)
def test_dollar_signs_are_escaped(path: Path):
    """GitHub renders text between two bare dollar signs as mathematics."""
    bare = re.findall(r"(?<!\\)\$.{0,20}", _outside_code(path.read_text()))
    assert not bare, f"{path.name}: unescaped dollar signs near {bare[:5]}"


def test_readme_never_prints_certainty_or_a_zero_standard_error():
    text = config.README.read_text()
    assert "1.000" not in text and "±0.0%" not in text and "(±0.00%)" not in text
