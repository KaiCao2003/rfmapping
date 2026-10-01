import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Utils.Sessions import Session


def test_session_rejects_non_adc_last_stream(tmp_path):
    probe = {
        "folder_name": "OneBox-1.ProbeA/",
        "sample_rate": 30000,
        "num_channels": 384,
        "recorded_processor_id": 103,
    }
    adc = dict(probe, folder_name="OneBox-1.OneBox-ADC/", num_channels=12)
    descriptor = tmp_path / "Record Node 103" / "experiment1" / "recording1" / "structure.oebin"
    descriptor.parent.mkdir(parents=True)
    data = {"continuous": [probe, adc], "events": [probe, adc]}
    descriptor.write_text(json.dumps(data))
    assert Session(descriptor).continuous_ADC_folder == "OneBox-1.OneBox-ADC"

    data["continuous"] = [adc, probe]
    descriptor.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="Last continuous stream must be ADC"):
        Session(descriptor)


@pytest.mark.parametrize(
    "start,end,expected", [(0, -2, [(0, -2)]), (3, None, [(3, None)]), (0, None, [])]
)
def test_notebook_persists_tail_trim_without_recording_noop(start, end, expected):
    notebook = json.loads((Path(__file__).resolve().parents[1] / "matlab.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "sessionEdit.deleteByRange(" in "".join(cell.get("source", []))
    )
    branch = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "deleteByRange"
            for statement in node.body
        )
    )
    calls = []

    def delete_by_range(start_frame, end_frame):
        calls.append((start_frame, end_frame))
        return True, None

    namespace = {
        "deleteStartIndex": start,
        "deleteEndIndex": end,
        "sessionEdit": SimpleNamespace(deleteByRange=delete_by_range),
    }
    exec(compile(ast.Module(body=[branch], type_ignores=[]), "matlab.ipynb", "exec"), namespace)
    assert calls == expected
