# RF Mapping Analysis

This repository contains the scientific RF mapping model, raw-trial loader,
cluster-permutation detector, notebooks, and MATLAB-related analysis sources.
The Python/Tk, SwiftUI, and Web viewers now live in the sibling
`../rfmapping_gui` repository and are not runtime dependencies of this package.

## Python RFMap API

`Utils/rfmap.py` loads one JSON file into an ordered `RFMapList` containing one
`RFMap` per recorded unit. Array-index lookup and recorded unit-ID lookup are
deliberately separate.

```python
import numpy as np

from Utils.rfmap import asrfmap, load_rf_maps
from Utils.rf_trials import load_regular_rf_trials

session = "/mnt/senzailab/Kai/#Recording/m15/260630/260630_3"
rf_json = (
    f"{session}/data/rfmapping/good/-100_400_1ms/ProbeA/"
    "regular_unitsSpikeCounts_260630_3.json"
)

raw = load_rf_maps(rf_json)
summed = raw.sum(0.0, 0.2, show_progress=True)
trials = load_regular_rf_trials(session, "A", summed)
result = summed._detect_rf(
    trials,
    is_shuffle=True,
    cluster_forming_z=1.5,
    alpha=0.05,
    n_permutations=10_000,
    random_seed=0,
    wrap_x=True,
    n_jobs=None,
    show_progress=True,
)

rf_masks_2d = result["final_mask"]
rf_masks_x = np.any(rf_masks_2d, axis=1).astype(np.uint8)

unit_by_source_index = summed.by_index(5)
the_same_unit = summed.by_unit_id(unit_by_source_index.unit_id)

zero_unit_offsets = np.unique(summed.where(0)[0])
zero_unit_ids = np.asarray(summed.unit_ids)[zero_unit_offsets]

array_map = asrfmap(np.zeros((7, 30)), start_time=0.0, end_time=0.2)
```

`sum(earlier, later)` uses seconds and the half-open interval
`[earlier, later)`. Both values must resolve to actual `timeBinEdges` entries
within `1e-12` seconds. Equal edges produce a valid zero-valued singleton time
axis; reversed intervals are invalid.

Formal `detect_rf()`, `rf_2d()`, and `rf_1d()` operate on a single-bin summed
object and matching trial data. A pooled JSON does not contain a trial axis and
is insufficient for label permutation. The regular-data loader reconstructs
per-trial responses from the authoritative MAT, onset, spike-time, cluster, and
good-unit files. Permutations keep responses fixed and shuffle the joint
`(x, y)` label within verified exchangeability blocks after ON/OFF filtering.

Candidate pixels use the configured cluster-forming z threshold. Significance
comes from the null distribution of the maximum 4-connected cluster mass. The
1-D RF is a projection of the final 2-D mask, not a separate test. Correction is
within a unit and does not correct across units, polarities, or separately run
analyses.

Batch work can show `Sum`, `Detecting RF`, and `Center` progress bars. Calling
`rf_2d(..., return_center=True)` reduces each non-empty mask to one
response-weighted RF bin. Pass `show_progress=False` for silent library use.

See [docs/rfmap.md](docs/rfmap.md) for the complete data contract, array shapes,
permutation semantics, and troubleshooting guide.

## Install and validate

Project code is run only on `hhw9l84` with the existing remote virtualenv:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && \
  ~/.virtualenvs/rfmapping/bin/pip install -e ".[test]"'

ssh hhw9l84 'cd ~/Developer/rfmapping && \
  PYTHONDONTWRITEBYTECODE=1 ~/.virtualenvs/rfmapping/bin/python -m pytest -q \
  tests/test_rfmap.py tests/test_rf_detection.py tests/test_rf_trials.py'
```

The optional `analysis` dependency group covers plotting/tuning helpers:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && \
  ~/.virtualenvs/rfmapping/bin/pip install -e ".[analysis,test]"'
```

## MATLAB pipeline

The active data-generation pipeline remains:

```text
~/Developer/sync/matlab.ipynb
-> /mnt/ssd4.1/Matlab/RFmapping.m
-> generated RF JSON
-> ~/Developer/rfmapping_gui/{python,swift,web}
```

The authoritative `RFmapping.m` and `RFmapping_core.m` sources are on
`hhw9l84:/mnt/ssd4.1/Matlab`; they are intentionally not duplicated into this
checkout. `MasterFile_EB_VisualStimuli.m` and the MATLAB notebooks here are
supporting/legacy analysis material. `vs.py` is also legacy translation work
and is not part of the standalone RF package validation gate.

With stimulus spacing near 100 ms, negative bins can overlap the previous
stimulus response and late positive bins can overlap the next stimulus. Those
spikes are assigned to the current trial's x/y by `RFmapping_core.m`, so JSON
activity before zero is not by itself a plotting bug or a pre-stimulus visual
response.
