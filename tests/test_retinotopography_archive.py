"""New ADN annotations retain the fixed cohort of archived source recordings."""

import hashlib
import json

import pytest

from Utils.retinotopography import load_adn_archive


@pytest.fixture
def adn_archive(tmp_path):
    source_units = tmp_path / "source_unit_angles_depth.csv"
    source_units.write_text(
        "mouse,date,probe,unit_id,y_um\n"
        "m14,260609,A,7,300\n"
        "m14,260609,A,8,200\n"
    )
    source_recordings = tmp_path / "source_recordings.csv"
    source_recordings.write_text("mouse,date,probe\nm14,260609,A\n")
    source_pdf = tmp_path / "selection.pdf"
    source_pdf.write_bytes(b"archived annotation PDF")
    replacement_image = tmp_path / "replacement.png"
    replacement_image.write_bytes(b"archived replacement image")
    config = {
        "recordings": [
            dict(mouse="m14", date="260609", probe="A", unit_ids=[7],
                 boundary_unit_ids=[7]),
            dict(mouse="m21", date="261006", probe="A", unit_ids=[140],
                 boundary_unit_ids=[], replacement_image="m21_annotation.png"),
        ],
    }
    for name, path in [("source_unit_table", source_units), ("source_pdf", source_pdf),
                       ("replacement_image", replacement_image)]:
        config[name] = str(path)
        config[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_path = tmp_path / "adn_units.json"
    config_path.write_text(json.dumps(config))
    return config_path, source_recordings


def test_new_recording_remains_in_config_without_joining_archived_cohort(adn_archive):
    config_path, source_recordings = adn_archive
    expected_config = json.loads(config_path.read_text())

    config, source_units, sources, audit, selected = load_adn_archive(
        config_path, source_recordings,
    )

    assert config == expected_config
    assert config["recordings"][1]["unit_ids"] == [140]
    assert sources[["mouse", "date", "probe"]].values.tolist() == [
        ["m14", "260609", "A"],
    ]
    assert source_units.unit_id.tolist() == [7, 8]
    assert audit.adn_selected.tolist() == [True, False]
    assert selected[["mouse", "date", "probe", "unit_id"]].values.tolist() == [
        ["m14", "260609", "A", 7],
    ]
    assert selected.boundary_included.tolist() == [True]


def test_missing_selected_unit_in_archived_recording_still_raises(adn_archive):
    config_path, source_recordings = adn_archive
    config = json.loads(config_path.read_text())
    config["recordings"][0]["unit_ids"].append(9)
    config_path.write_text(json.dumps(config))

    with pytest.raises(AssertionError):
        load_adn_archive(config_path, source_recordings)
