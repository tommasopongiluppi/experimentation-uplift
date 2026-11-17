"""The walkthrough notebook is committed executed and shows aggregates only."""

import json

from abkit import config

NOTEBOOK = config.ROOT / "notebooks" / "walkthrough.ipynb"
MAX_TABLE_ROWS = 20  # summary tables only; a dump of customers would be far longer


def _cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def test_notebook_is_short_and_executed_in_order():
    cells = _cells()
    assert 10 <= len(cells) <= 15
    code = [c for c in cells if c["cell_type"] == "code"]
    assert [c["execution_count"] for c in code] == list(range(1, len(code) + 1))
    assert all(c["outputs"] for c in code)
    assert not any(o["output_type"] == "error" for c in code for o in c["outputs"])


def test_notebook_outputs_hold_no_customer_rows_or_local_paths():
    for cell in _cells():
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            data = output.get("data", {})
            text += "".join(data.get("text/plain", [])) + "".join(data.get("text/html", []))
            assert "customer_id" not in text
            assert "/Users/" not in text and "/home/" not in text
            assert "".join(data.get("text/plain", [])).count("\n") < MAX_TABLE_ROWS
            assert "".join(data.get("text/html", [])).count("<tr") <= MAX_TABLE_ROWS
