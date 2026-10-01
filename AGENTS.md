# Agent Instructions

- This repository is the RF analysis/Matlab side of the project. GUI work
  belongs in the sibling `../rfmapping_gui` repository.
- Run project code only on the remote host reached with `ssh hhw9l84`.
- Use the remote virtual environment at `~/.virtualenvs/rfmapping`.
- Do not run project Python from the local checkout. For example:

  ```sh
  ssh hhw9l84 'cd ~/Developer/rfmapping && \
    ~/.virtualenvs/rfmapping/bin/python -m pytest -q \
    tests/test_rfmap.py tests/test_rf_detection.py tests/test_rf_trials.py'
  ```

- Ignore MATLAB `.m` files when resolving Python dependencies or validating
  the Python runtime.
- Original MATLAB data-generation sources live on the Linux remote host under
  `/mnt/ssd4.1/Matlab`. When investigating JSON generation, inspect those files
  through `ssh hhw9l84`; do not treat local legacy `.m` copies as authoritative.
- `vs.py` is legacy translation work with remote-only helper modules and is not
  part of the standalone RF package validation gate.

## RFmapping Pipeline Notes

The active pipeline is:

```text
~/Developer/sync/matlab.ipynb
-> /mnt/ssd4.1/Matlab/RFmapping.m
-> generated RF JSON
-> ~/Developer/rfmapping_gui/{python,swift,web}
```

- Treat remote raw/session data as source of truth. For timing or JSON
  generation questions, inspect `/mnt/senzailab/Kai/#Recording` and
  `/mnt/ssd4.1/Matlab`; do not rely on stale viewer fixtures.
- When investigating "response before VS", inspect the full `timeBinEdges`,
  per-bin timeline, and, when needed, recompute from `on_list_times.npy`,
  `trials.mat`, `adc_spike_time.npy`, `spike_clusters.npy`, and good-unit labels.
- With windows such as `VSTimeWindow = [-0.1 0.2]` and stimuli spaced about
  100 ms apart, negative bins mostly overlap the prior stimulus response and
  late positive bins can overlap the next stimulus. `RFmapping_core.m` assigns
  those spikes to the current trial's x/y.
- Plot range is a 2-D display control only. Timeline views should show the full
  time axis unless a dedicated timeline control explicitly filters it.
