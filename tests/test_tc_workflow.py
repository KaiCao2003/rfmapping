import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from Utils import direction_comparison as comparison


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def source_readers(monkeypatch):
    calls = []

    def reader(kind):
        def load(path, *, probe, **options):
            calls.append((kind, Path(path), probe, options))
            bins = 60 if kind == "EBC" else 30
            table = comparison.profile_table(
                [np.arange(1., bins + 1)], [7], np.arange(bins) * 360 / bins + 180 / bins, probe=probe,
            )
            return table
        return load

    for kind in ("HD", "RF", "EBC"):
        monkeypatch.setattr(comparison, f"load_{kind.lower()}_profiles", reader(kind))
    return calls


@pytest.fixture
def classified_tc(tmp_path):
    path = tmp_path / "tuning_curves.tc"
    path.write_text(json.dumps({
        "unit_id": [17, 9, 31, 5, 23, 2],
        "unit_data": {
            "hd_class": [2, 1, 0, None, 1, 2],
            "rayleigh_p": [.001, .001, .2, None, .2, .001],
            "shuffle_p": [.001, .2, .2, None, .001, .001],
        },
        "spike_counts": [[2 * rate] * 180 for rate in [5, 3, 8, 1, 6, 2]],
        "occupancy_time_s": [2] * 180,
        "angle_bin_edges_deg": list(range(0, 361, 2)),
    }))
    return path


@pytest.mark.parametrize("hd_class, unit_ids, rates", [
    (1, [9, 23], [3, 6]),
    (2, [17, 2], [5, 2]),
    ((1, 2), [17, 9, 23, 2], [5, 3, 6, 2]),
    ([2, 1, 2], [17, 9, 23, 2], [5, 3, 6, 2]),
])
def test_load_tc_selects_classes_without_reordering_or_duplicating_units(classified_tc, hd_class, unit_ids, rates):
    options = dict(mouse="m14", date=260609, probe="A")
    table = comparison.load_tc(classified_tc, hd_class=hd_class, **options)
    assert table.index.tolist() == [("m14", "260609", "A", unit_id) for unit_id in unit_ids]
    assert table.index.is_unique
    np.testing.assert_allclose(table.to_numpy(), np.repeat(np.array(rates)[:, None], 30, axis=1))
    all_units = comparison.load_tc(classified_tc, **options)
    pd.testing.assert_frame_equal(comparison.hd_pick(all_units, hd_class=hd_class), table)
    assert set(table.attrs["unit_info"]) == set(table.index)


def test_hd_default_loading_keeps_zero_missing_and_unknown_units_without_writing(classified_tc):
    data = json.loads(classified_tc.read_text())
    data["spike_counts"][0] = [0] * 180
    data["spike_counts"][3] = [None] * 180
    classified_tc.write_text(json.dumps(data))
    sentinel = classified_tc.parent / "hd_cells_2.npy"
    sentinel.write_bytes(b"existing saved classification")
    before = {path.name: path.read_bytes() for path in classified_tc.parent.iterdir()}

    table = comparison.load_tc(classified_tc, mouse="m14", date=260609)

    assert table.index.get_level_values("unit_id").tolist() == data["unit_id"]
    np.testing.assert_array_equal(table.iloc[0], np.zeros(30))
    assert table.iloc[3].isna().all()
    assert table.attrs["unit_info"] == {
        ("m14", "260609", "A", unit): {"hd_class": value}
        for unit, value in zip(data["unit_id"], [2, 1, 0, None, 1, 2], strict=True)
    }
    assert {path.name: path.read_bytes() for path in classified_tc.parent.iterdir()} == before
    original = table.copy(deep=True)
    picked = comparison.hd_pick(table)
    assert picked.index.get_level_values("unit_id").tolist() == [17, 2]
    np.testing.assert_array_equal(picked.iloc[0], np.zeros(30))
    picked.attrs["unit_info"][picked.index[0]]["hd_class"] = 0
    pd.testing.assert_frame_equal(table, original)
    assert table.attrs == original.attrs
    pd.testing.assert_frame_equal(comparison.hd_pick(table, hd_class=None), table)


def test_hd_selection_commutes_with_combine_and_preserves_all_metadata(classified_tc):
    tables = [comparison.load_tc(classified_tc, mouse=mouse, date=date, probe=probe)
              for mouse, date, probe in [("m14", 260609, "A"), ("m14", 260609, "B"),
                                        ("m14", 260610, "A"), ("m15", 260609, "A")]]
    pooled = comparison.combine(*tables)
    assert pooled.attrs["unit_info"] == {
        key: info for table in tables for key, info in table.attrs["unit_info"].items()
    }
    selected_after = comparison.hd_pick(pooled, hd_class=(1, 2))
    selected_before = comparison.combine(*(comparison.hd_pick(table, hd_class=(1, 2)) for table in tables))
    pd.testing.assert_frame_equal(selected_after, selected_before)
    assert selected_after.attrs == selected_before.attrs
    subset = tables[0].iloc[:1]
    combined_subset = comparison.combine(subset, tables[1])
    assert set(combined_subset.attrs["unit_info"]) == set(combined_subset.index)


def test_rf_selection_uses_native_missing_counts_after_resampling_and_combine(tmp_path):
    path = tmp_path / "native_60_bins.rfmap"
    rates = np.ones((4, 2, 60, 1))
    rates[0] = 0
    rates[1, :, :2] = np.nan
    rates[2, :, :3] = np.nan
    rates[3] = np.nan
    stored = rates.astype(object)
    stored[np.isnan(rates)] = None
    path.write_text(json.dumps({
        "unitsSpikeCounts": stored.tolist(), "unitsSpikeCountsSize": list(rates.shape),
        "unitPool": [7, 9, 11, 13], "xPositions": np.arange(3., 360., 6.).tolist(),
        "yPositions": [-12., 12.], "timeBinEdges": [0., .2], "responseUnits": "Hz",
    }))
    before = path.read_bytes()
    tables = [comparison.load_rf(path, mouse="m14", date=260609, probe=probe) for probe in ("A", "B")]
    for table, probe in zip(tables, ("A", "B"), strict=True):
        assert table.shape == (4, 30)
        assert table.attrs["unit_info"] == {
            ("m14", "260609", probe, unit): {"missing_bins": count}
            for unit, count in zip([7, 9, 11, 13], [0, 2, 3, 60], strict=True)
        }
        assert table.iloc[1].isna().sum() != table.attrs["unit_info"][table.index[1]]["missing_bins"]
        np.testing.assert_array_equal(table.iloc[0], np.zeros(30))
        assert table.iloc[3].isna().all()
        selected = comparison.rf_pick(table)
        assert selected.index.get_level_values("unit_id").tolist() == [7, 9]
        during_load = comparison.load_rf(path, mouse="m14", date=260609, probe=probe, max_missing_bins=2)
        pd.testing.assert_frame_equal(selected, during_load)
        assert selected.attrs == during_load.attrs
        original = table.copy(deep=True)
        selected.attrs["unit_info"][selected.index[0]]["missing_bins"] = 99
        pd.testing.assert_frame_equal(table, original)
        assert table.attrs == original.attrs
        pd.testing.assert_frame_equal(comparison.rf_pick(table, max_missing_bins=None), table)
    pooled = comparison.combine(*tables)
    selected_after = comparison.rf_pick(pooled)
    selected_before = comparison.combine(*(comparison.rf_pick(table) for table in tables))
    pd.testing.assert_frame_equal(selected_after, selected_before)
    assert selected_after.attrs == selected_before.attrs
    assert set(selected_after.attrs["unit_info"]) == set(selected_after.index)
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path}


def test_profile_transformations_preserve_and_subset_unit_metadata(classified_tc):
    source = comparison.load_hd_profiles(classified_tc, bins=60)
    source.attrs["provenance"] = "native recording"
    original = source.copy(deep=True)
    collection = comparison.TuningCurveCollection()
    table = collection.add(source, mouse="m14", date=260609, prefix="HD")
    assert table.shape == (6, 30)
    assert table.attrs["provenance"] == "native recording"
    assert table.attrs["unit_info"] == {
        ("m14", "260609", *key): value for key, value in source.attrs["unit_info"].items()
    }
    converted = comparison.convert_profile_coordinates(table, is_ego_deg=False)
    smoothed = comparison.smooth_profiles(converted, 1)
    assert smoothed.attrs == converted.attrs
    units = table.index[[2, 0]].tolist()
    aligned = comparison.align_profiles(smoothed, units, [0, 90])
    assert aligned.index.tolist() == units
    assert aligned.attrs["unit_info"] == {key: table.attrs["unit_info"][key] for key in units}
    assert aligned.attrs["provenance"] == "native recording"
    assert aligned.attrs["is_ego_deg"] is False
    pd.testing.assert_frame_equal(source, original)
    assert source.attrs == original.attrs


@pytest.mark.parametrize("loader, kind, options", [
    ("load_tc", "HD", {}),
    ("load_rf", "RF", {"window": (0., .1)}), ("load_ebc", "EBC", {}),
])
def test_loaders_return_same_table_contract_from_literal_paths(tmp_path, source_readers, capsys, loader, kind, options):
    path = tmp_path / "exports/curve_{date}_{session}.data"
    defaults = {
        "HD": {"bins": 30, "smoothing_deg": 0, "hd_class": None},
        "RF": {"window": (0., .2), "smoothing_bins": 0, "max_missing_bins": None},
        "EBC": {},
    }[kind]
    table = getattr(comparison, loader)(path, mouse="m19", date=260918, **options)
    assert source_readers == [(kind, path, "A", defaults | options)]
    assert isinstance(table, pd.DataFrame) and table.shape == (1, 30)
    assert table.index.names == ["mouse", "date", "probe", "unit_id"]
    assert table.index.tolist() == [("m19", "260918", "A", 7)]
    assert table.attrs["label"] == kind
    assert table.attrs["is_ego_deg"] is (kind != "HD")
    np.testing.assert_array_equal(np.sort(table.columns), np.arange(-174., 180., 12.))
    custom = getattr(comparison, loader)(path, mouse="m19", date=260918, probe="B",
                                        label="Custom", is_ego_deg=True)
    assert custom.index.tolist() == [("m19", "260918", "B", 7)]
    assert custom.attrs["label"] == "Custom" and custom.attrs["is_ego_deg"] is True
    assert source_readers[-1] == (kind, path, "B", defaults)
    assert capsys.readouterr().out == ""


def test_combine_preserves_identity_and_rejects_repeated_units(source_readers, capsys):
    tables = [comparison.load_tc("literal.tc", mouse=mouse, date=date, probe=probe)
              for mouse, date, probe in [("m19", 260918, "A"), ("m19", 260918, "B"),
                                         ("m19", 260921, "A"), ("m20", 260918, "A")]]
    pooled = comparison.combine(*tables, label="Pooled HD")
    assert isinstance(pooled, pd.DataFrame) and pooled.shape == (4, 30)
    assert pooled.index.tolist() == [table.index[0] for table in tables]
    assert pooled.attrs["label"] == "Pooled HD"
    pd.testing.assert_frame_equal(comparison.combine(comparison.combine(*tables[:2]), tables[2]),
                                  comparison.combine(*tables[:3]))
    for table in tables:
        np.testing.assert_array_equal(pooled.loc[table.index], table)
    with pytest.raises(ValueError, match="overlapping"):
        comparison.combine(tables[0], tables[0].copy())
    result = comparison.plot_sort(tables[0], pooled.loc[tables[0].index], show=False)
    assert result["order"] == tables[0].index.tolist()
    assert capsys.readouterr().out == ""


def test_independent_plots_share_prepared_inputs_without_label_collisions(capsys):
    angles = np.arange(-174., 180., 12.)
    tables = []
    for ids, peaks in (([7, 9, 11], [54., -18., 126.]), ([7, 9, 13], [-66., 30., 90.])):
        table = comparison.profile_table(2 + np.cos(np.deg2rad(angles[None, :] - np.array(peaks)[:, None])),
                                         ids, angles, probe="A")
        table = pd.concat({("m19", "260918"): table}, names=["mouse", "date"])
        table.attrs.update(label="HD", is_ego_deg=False)
        tables.append(table)
    units = tables[0].index.intersection(tables[1].index)
    tables = [comparison.smooth_profiles(table.loc[units], 1) for table in tables]
    expected = comparison.paired_peak_angles(*tables)
    with plt.style.context("dark_background"):
        for plot in (comparison.plot_sort, comparison.plot_align, comparison.plot_sum):
            result = plot(*tables, show=False)
            assert len(result["figures"]) == 2
            pd.testing.assert_frame_equal(result["peaks"], expected)
            assert result["order"] == expected.index.tolist()
            for figure, axis in result["figures"]:
                assert figure.get_facecolor() == axis.get_facecolor() == (1., 1., 1., 1.)
                assert axis.xaxis.label.get_color() == "black"
    reversed_expected = comparison.paired_peak_angles(*reversed(tables))
    reversed_result = comparison.plot_sort(*reversed(tables), show=False)
    assert len(reversed_result["figures"]) == 2
    pd.testing.assert_frame_equal(reversed_result["peaks"], reversed_expected)
    result = comparison.plot_sort(*tables, labels=("Before", "After"), show=False)
    assert "Before" in result["figures"][0][1].get_title()
    assert "After" in result["figures"][1][1].get_title()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("plot_name, align, mode", [
    ("plot_sort", False, "native"), ("plot_align", True, "aligned"), ("plot_sum", "sum", "sum"),
])
def test_pair_and_named_plots_preserve_labels_exports_and_show_lifecycle(tmp_path, monkeypatch, plot_name, align, mode):
    angles = np.arange(-174., 180., 12.)
    hd = comparison.profile_table([2 + np.cos(np.deg2rad(angles - 54))], [7], angles)
    rf = comparison.profile_table([2 + np.cos(np.deg2rad(angles + 18))], [7], angles)
    hd.attrs["label"], rf.attrs["label"] = "HD label", "RF label"
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf().number))
    direct = getattr(comparison, plot_name)(hd, rf, save_dir=tmp_path / "direct")
    assert shown == [figure.number for figure, _ in direct["figures"]]
    assert all(not plt.fignum_exists(figure.number) for figure, _ in direct["figures"])
    assert {path.name for path in (tmp_path / "direct").iterdir()} == {
        f"0_1_0_{mode}.png", f"0_1_1_{mode}.png",
    }
    named = comparison.plot_comparison_heatmaps(
        {"HD": hd, "RF": rf}, sorted_by="HD", apply_to="RF", align=align,
        save_dir=tmp_path / "named", show=False,
    )
    assert len(shown) == 2
    assert all(plt.fignum_exists(figure.number) for figure, _ in named["figures"])
    assert {path.name for path in (tmp_path / "named").iterdir()} == {
        f"HD_RF_HD_{mode}.png", f"HD_RF_RF_{mode}.png",
    }
    pd.testing.assert_frame_equal(named["peaks"], direct["peaks"])
    suffixes = ("", "") if not align else (" | peak aligned", " | peak sum" if align == "sum" else " | peak aligned")
    for (_, direct_axis), (_, named_axis), label, suffix in zip(direct["figures"], named["figures"], ("HD", "RF"), suffixes, strict=True):
        np.testing.assert_array_equal(direct_axis.images[0].get_array(), named_axis.images[0].get_array())
        assert direct_axis.get_title() == f"{label} label | sorted by HD label{suffix}"
        assert named_axis.get_title() == f"{label} | sorted by HD{suffix}"
