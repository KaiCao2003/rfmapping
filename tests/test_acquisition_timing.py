import ast
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from Utils.Sessions import Session
from Utils.session_edits import (
    SessionEditStore,
    apply_session_edit,
    check_session_edits,
    open_session_edit_store,
)


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


def test_session_edit_open_does_not_create_or_initialize_database(tmp_path):
    missing = tmp_path / "new-mouse" / "session_slices.sqlite3"
    with pytest.raises(FileNotFoundError):
        check_session_edits(missing, "260922", 3)
    assert not missing.parent.exists()

    uninitialized = tmp_path / "empty.sqlite3"
    uninitialized.touch()
    with pytest.raises(RuntimeError, match="not initialized"):
        open_session_edit_store(uninitialized)
    assert uninitialized.read_bytes() == b""


def test_saved_session_edits_are_read_and_applied_separately(tmp_path):
    database = tmp_path / "session_slices.sqlite3"
    assert SessionEditStore.create_database(database)
    session = check_session_edits(database, "260922", 3)
    assert session.deleteByFrame([1, -1]) == (True, None)
    assert session.interp(1, 2, 2) == (True, None)
    stored_bytes = database.read_bytes()

    edit = session.get_session_edit()
    assert session.getSessionInfo() == (True, edit)
    target = np.array([0., 99., 10., 40., 50., 999.])
    edited = apply_session_edit(target, edit)
    np.testing.assert_array_equal(edited, [0., 10., 20., 30., 40., 50.])
    np.testing.assert_array_equal(target, [0., 99., 10., 40., 50., 999.])
    assert database.read_bytes() == stored_bytes
    with pytest.raises(TypeError):
        session.getSessionInfo(target=target)

    # The getter's connection cannot create or change a database, even if a
    # future query is accidentally changed to a write.
    with SessionEditStore._connect(database, read_only=True) as connection:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM session_edits")


@pytest.mark.parametrize("target", [[1., 2.], np.array([1., 2.])])
def test_missing_session_edit_returns_independent_input_copy(target):
    edited = apply_session_edit(target, None)
    edited[0] = 99.
    assert target[0] == 1.


def test_interval_generation_rejects_ambiguous_session_before_writing(tmp_path):
    from Utils.recording import gen_recording_interval_table

    for name in ("recording1", "recording2"):
        descriptor = tmp_path / name / "structure.oebin"
        descriptor.parent.mkdir()
        descriptor.write_text("{}")

    with pytest.raises(ValueError, match="Multiple session descriptor"):
        gen_recording_interval_table(str(tmp_path))
    assert not (tmp_path / "data").exists()


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
