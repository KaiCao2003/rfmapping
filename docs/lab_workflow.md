# Lab deployment example: raw recordings to `.rfmap` on hhw9l84

This optional worked example uses the existing **`hhw9l84`** lab deployment,
including its installed software, launcher, and fixed directory layout. Follow
the [main guide](../README.md) for the data structures and a workflow with
configurable code and data paths. The commands below assume this lab setup.

Keep the current Linux/macOS SQLite connection for this deployment. For native
Windows, use the [platform and database instructions](platforms_and_data_safety.md)
before opening the manual timing notebook. Preserve the existing mouse-level
correction database and onset files before rerunning save cells; a different
computer is not a reason to recreate or overwrite the database.

The main example is **mouse `m20`, date `260918`, session `2`, Probe `A`**,
with regular square stimuli. Replace those values with your recording's values.
The variations section covers OFF maps, vertical bars, and the additional
inputs for free-moving RF mapping.

```text
Raw Open Ephys recording                  Stimulus experiment's <date>.mat
          |                                           |
          v                                           |
run_pipeline.sh: sorting + session splitting           |
          |                                           |
          +--> spike_clusters.npy + cluster_KSLabel.tsv|
          +--> data/probeA/adc_spike_time.npy           |
                                                      |
Raw ADC photodiode --> timing notebook --> on_list_times.npy
          |                                           |
          +--------------------+----------------------+
                               |
                               v
                         RFmapping.m
                               |
                               v
              data/rfmapping/good/-100_400_1ms/ProbeA/
                  regular_unitsSpikeCounts_260918_2.rfmap
```

## 1. Choose your starting point

| What you already have | Start at |
| --- | --- |
| Raw Open Ephys files and the matching stimulus `.mat` | [Prepare and sort the raw recording](#4-prepare-and-sort-the-raw-recording) |
| Per-session Kilosort output and `adc_spike_time.npy` | [Create the stimulus onset file](#5-create-the-stimulus-onset-file) |
| Sorted spikes and a checked `on_list_times.npy` | [Check the MATLAB inputs](#6-check-the-matlab-inputs) |
| A generated `.rfmap` | [Check the generated file](#9-check-the-generated-file) |

The output contains a response timeline for each included unit at each spatial
position. Its axes are **unit, y, x, time**. Generating it does not require
`locate_rf.ipynb`, a shuffle test, HD tuning, EBC analysis, or the GUI. Those
can use the file afterward.

## 2. Use the correct machine and environment

### 2.1 Open a remote terminal

Run this on your own computer:

```sh
ssh hhw9l84
```

All later shell commands run in that remote terminal unless a step explicitly
says **local computer**. The `/mnt/...` paths are on `hhw9l84`.

| Location on `hhw9l84` | Purpose |
| --- | --- |
| `~/scripts/run_pipeline.sh` | Raw processing, spike sorting, and session splitting |
| `~/pipeline` | Upstream processing implementation and its environment |
| `~/spikeinterface` | Sorting analysis, splitting, and spike-time export |
| `~/Developer/rfmapping` | RF timing notebooks and Python RF reader |
| `~/.virtualenvs/rfmapping` | Python environment for this repository |
| `/mnt/ssd4.1/Matlab` | Authoritative MATLAB RF generator |
| `/mnt/senzailab/Kai/#Recording` | Archived recordings and session results |

Use the remote MATLAB files. Legacy MATLAB files in this Python checkout are
not the generator used here. Use `~/Developer/rfmapping/matlab.ipynb` for the
current stored-edit timing workflow; the older copy under
`~/Developer/sync` has different settings and logic.

Check the installed tools:

```sh
ls -l ~/scripts/run_pipeline.sh
ls -l ~/.virtualenvs/rfmapping/bin/python
ls -l ~/pipeline/.venv/bin/pipe
ls -l ~/spikeinterface/venv/bin/python
command -v matlab
```

For RF Python work:

```sh
cd ~/Developer/rfmapping
source ~/.virtualenvs/rfmapping/bin/activate
python -c 'import sys; print(sys.executable)'
```

The executable should be under `/home/kai/.virtualenvs/rfmapping/`. The
upstream launcher selects its own environments for its own stages; leave
those paths as supplied by the script.

If this checkout has not been installed into the RF environment:

```sh
python -m pip install -e '.[analysis]'
```

The upstream processing described below is configured for the lab's current
OneBox/Neuropixels setup. Its session splitter currently assumes 384 channels,
int16 samples, and a nominal 30 kHz probe rate; Kilosort uses CUDA. A different
recording format or probe layout requires checking that upstream configuration
before following these commands.

### 2.2 Open the remote notebooks

If your editor already runs notebooks with the remote RF Python environment,
use that setup. Otherwise, use Jupyter through an SSH tunnel.

In a terminal on your **local computer**:

```sh
ssh -L 8888:127.0.0.1:8888 hhw9l84
```

In the remote terminal opened by that command:

```sh
cd ~/Developer/rfmapping
source ~/.virtualenvs/rfmapping/bin/activate
python -m jupyterlab --no-browser --ip=127.0.0.1 --port=8888
```

Open the printed `http://127.0.0.1:8888/...` URL, including its token, in your
local browser. Keep the terminal open. Check the notebook kernel with:

```python
import sys
print(sys.executable)
```

It must point to the remote RF environment. Select a cell and press
**Shift+Enter** to run it. Restart the kernel when switching recordings, then
rerun the required setup cells so values from the previous recording are not
reused.

## 3. Identify the recording and collect its files

Write down these values first:

| Setting | Example | Meaning |
| --- | --- | --- |
| Mouse | `m20` | Animal directory |
| Date | `260918` | Six-character recording-date label |
| Session | `2` | One recording within that day |
| Probe | `A` | Probe whose units enter the map |
| Stimulus | Regular squares | Determines the MATLAB geometry settings |
| Polarity | ON | White stimuli: `Square_Luminance == 1` |

The example session directory is:

```text
/mnt/senzailab/Kai/#Recording/m20/260918/260918_2/
```

The repeated date is intentional. A complete session has this structure:

```text
260918_2/
├── 260918.mat                         stimulus log from this exact session
├── 260918/
│   └── Record Node 102/
│       ├── settings.xml               acquisition settings and probe geometry
│       └── experiment1/recording1/
│           ├── structure.oebin
│           ├── sync_messages.txt
│           ├── continuous/
│           │   ├── OneBox-107.ProbeA/
│           │   │   ├── continuous.dat
│           │   │   ├── timestamps.npy
│           │   │   └── sample_numbers.npy
│           │   └── OneBox-107.OneBox-ADC/
│           │       ├── continuous.dat
│           │       ├── timestamps.npy
│           │       └── sample_numbers.npy
│           └── events/                retain the original event files
├── kilosort/                          created by upstream processing
│   └── ProbeA/kilosort_2/
│       ├── spike_times.npy
│       ├── spike_clusters.npy
│       └── cluster_KSLabel.tsv
└── data/
    ├── probeA/adc_spike_time.npy       created after session splitting
    └── on_list_times.npy              created by the RF timing notebook
```

Record Node numbers and stream names vary. Read them from your recording's
`structure.oebin`; do not rename raw stream folders to match the example.
Linux paths are case-sensitive: **`kilosort/ProbeA`** and **`data/probeA`**
use different capitalization.

The original stimulus log is required. `260918.mat` must contain the
`trials` recorded for **session 2**, in presentation order. Different sessions
can use the same date-based filename, so check the enclosing session directory.
The neural recording alone cannot recover which square was displayed on
each trial.

| Required field in `trials` | Meaning |
| --- | --- |
| `Square_PositionX` | Horizontal stimulus position in screen/RF degrees |
| `Square_PositionY` | Vertical stimulus position in screen/RF degrees |
| `Square_Size` | Square size, or bar width, in degrees |
| `Square_Luminance` | `1` for white/ON; `0` for black/OFF |

The recorded photodiode supplies actual display timing. The `trials` array
supplies the matching stimulus identity and location.

## 4. Prepare and sort the raw recording

Skip this section if the per-session outputs in Section 4.4 already exist
and belong to the intended sorting run. Reprocessing can replace split
sorting outputs; inspecting them does not require rerunning the pipeline.

### 4.1 Put the raw recording in the SSD input location

For every session being sorted, the upstream input is:

```text
/mnt/ssd4.1/260918_2/Record Node 102/experiment1/recording1/...
```

Put the complete Open Ephys tree there: the Record Node's `settings.xml`,
descriptor, continuous streams, timestamps, sample numbers, and events.
The upstream reader uses `settings.xml` to obtain the probe geometry.
There should not be an extra
`260918` directory between `260918_2` and `Record Node ...` at this SSD
location.

If the raw data are already on the lab share, copy that session's inner date
directory back to the SSD:

```sh
mkdir -p /mnt/ssd4.1/260918_2
rsync -rlt --info=progress2 \
  '/mnt/senzailab/Kai/#Recording/m20/260918/260918_2/260918/' \
  /mnt/ssd4.1/260918_2/
```

Repeat for every session in the sorting group, then inspect each input:

```sh
find /mnt/ssd4.1/260918_2 -name structure.oebin
```

The result should identify the intended recording. Resolve extra Record Nodes
or recordings before proceeding; the RF notebooks below expect the selected
session's `experiment1/recording1` layout.

### 4.2 Configure the sorting group

Make a run-specific copy of the launcher:

```sh
cp ~/scripts/run_pipeline.sh ~/scripts/run_rf_260918.sh
nano ~/scripts/run_rf_260918.sh
```

Change the top configuration block. To reproduce the sorting group used by
the worked example:

```bash
MOUSE_ID="m20"
DATE="260918"
SESSION_LIST=(2 3 1 4 5 6 7 8 9 10 11)
FREE_MOVING_SESSIONS=()
BASLER_SESSIONS=()
```

This group includes recordings used to compare the same units across
sessions. Every listed session must have its own SSD input directory.
For a **new experiment intended to sort only session 2 independently**, use:

```bash
SESSION_LIST=(2)
```

Use the group and order planned for your experiment. To reproduce an existing
jointly sorted experiment, retain its group rather than replacing one session
with independently sorted results. The launcher passes the same ordered list
to concatenation and splitting. Equal numerical cluster IDs from independent
sorts do not establish that they are the same neuron.

Empty `FREE_MOVING_SESSIONS` and `BASLER_SESSIONS` skip optional behavior,
tuning, spatial-cell, and video analyses. They do not remove sessions from
sorting. Regular screen-coordinate RF generation needs none of those optional
analyses.

In `nano`, press **Ctrl+O**, **Enter**, then **Ctrl+X** to save and exit.

### 4.3 Run upstream processing

```sh
bash ~/scripts/run_rf_260918.sh
```

The launcher:

1. Copies raw recordings to their session directories on the lab share.
2. Generates `~/pipeline/config_result.yaml` for the selected sessions.
3. Runs concatenation and Kilosort processing.
4. Runs SpikeInterface analysis and splits sorting results by session.
5. Exports each session's `data/probeA/adc_spike_time.npy` and corresponding
   files for the other detected probes.
6. Finishes the pipeline archive transfers.

Wait for successful completion. This script does not call `RFmapping.m`
or create the RF stimulus-onset file. It uses a shared
`~/pipeline/config_result.yaml`; run one configured group at a time.

### 4.4 Check the per-session outputs

```sh
ls -lh \
  '/mnt/senzailab/Kai/#Recording/m20/260918/260918_2/kilosort/ProbeA/kilosort_2/spike_times.npy' \
  '/mnt/senzailab/Kai/#Recording/m20/260918/260918_2/kilosort/ProbeA/kilosort_2/spike_clusters.npy' \
  '/mnt/senzailab/Kai/#Recording/m20/260918/260918_2/kilosort/ProbeA/kilosort_2/cluster_KSLabel.tsv' \
  '/mnt/senzailab/Kai/#Recording/m20/260918/260918_2/data/probeA/adc_spike_time.npy'
```

Use the **per-session** sorting directory. Splitting preserves cluster IDs
and subtracts each session's concatenation offset from `spike_times.npy`.
A sample index from the concatenated run cannot directly index a single
session's timestamp array.

The RF generator selects `good` labels from `cluster_KSLabel.tsv`. Editing
only `cluster_group.tsv` in a separate curation workflow does not change
this selection.

### 4.5 If only the spike timestamp file is missing

After the raw archive and per-session sorting files exist, regenerate the
timestamp file without rerunning sorting:

```sh
~/.virtualenvs/rfmapping/bin/python \
  ~/spikeinterface/generate_adc_spike_time.py \
  '/mnt/senzailab/Kai/#Recording/m20' 260918 2 A
```

Use `AB` instead of `A` when both probes are present and should be exported.
This command writes the selected session's `data/probeA/adc_spike_time.npy`.

The exporter indexes the original probe `timestamps.npy` with the
session-local Kilosort sample numbers. It preserves their recorded timestamp
origin by default; these seconds are not necessarily Unix/UTC timestamps.
**Do not add `--convert-to-zero` for this workflow.**
The onset notebook also keeps its original timestamp origin.

The concatenated upstream pipeline also produces a file named
`adc_spike_times.npy`, with a plural `times`. That has different timing
provenance. MATLAB expects the singular
**`<session>/data/probeA/adc_spike_time.npy`** described here.

## 5. Create the stimulus onset file

The required result is:

```text
<session>/data/on_list_times.npy
```

For **N trials**, the current MATLAB generator expects **N+1 timestamps**:

```text
trial 0 onset, trial 1 onset, ..., trial N-1 onset, final end boundary
```

The notebooks construct N onsets, then append the final boundary. For the
approximately 100 ms stimulus design, that boundary is the last onset plus
0.1 seconds. It closes the last interval; it is not another stimulus.

Use **one** of the following timing routes. They write the same output file.

### 5.1 New periodic-stimulus session: preview with `matlab_auto.ipynb`

Open [matlab_auto.ipynb](../matlab_auto.ipynb). This notebook handles the
approximately periodic stimulus-edge sequence used here. It cleans detected
transitions, trims the surrounding recording, and evaluates missing-edge
repairs against timing and trial-count constraints.

In the cell beginning `Change all the variables below to match recording`,
set:

```python
base_dir = r"/mnt/senzailab/Kai/#Recording/m20"
date = "260918"
num_of_rec = 2
analogue_input_channel = 3
is_convert_to_zero = False
is_save_on_list_time = False
final_boundary_interval_seconds = 0.1
```

Here `base_dir` is the **mouse directory**. The notebook appends the date and
session itself. Channel numbers are **zero-based**: `3` means the fourth
ADC channel. Confirm the recording wiring before reusing that value.

Keep saving disabled for the first pass. Run from the import cell through the
final export cell. It should report that `on_list_times.npy` was not changed.

The detector uses a raw threshold of `1000` and short-gap/short-run limits in
**samples**, not milliseconds. These are settings for this detector. The manual
notebook uses a different cleaning expression; inspect the raw trace before
transferring thresholds between them.

Review the printed counts and diagnostic plots:

- The final onset count must equal the `trials` count.
- The retained block must begin with the first actual stimulus and end with
  the last actual stimulus.
- Both rising and falling transitions can mark consecutive stimulus trials
  in this design. Counting only rising transitions is insufficient.
- Deleted and inferred events must agree with the raw ADC trace.
- Review every repaired gap. A real long stimulus interval should remain a
  long interval. Matching the total count alone does not establish alignment.
- The exported boundary count must equal the trial count plus one.

If the notebook reports an ambiguous repair or failed assertion, inspect the
indicated trace and stimulus log. Do not remove the assertion or fill all long
gaps just to obtain a matching count.

After the preview is correct, change:

```python
is_save_on_list_time = True
```

Rerun the notebook. Its final cell saves the onset file and prints the full
path. This route does not apply the session-edit database. Do not apply old
delete indices to its already-cleaned result.

### 5.2 Existing stored timing corrections: `matlab.ipynb`

Open [matlab.ipynb](../matlab.ipynb). Run the imports, then configure the main
parameter cell:

```python
base_dir = r"/mnt/senzailab/Kai/#Recording/m20"
date = "260918"
num_of_rec = 2
probe_list = ["A"]
analogue_input_channel = 3
is_convert_to_zero = False
is_signal_inverted = True
```

Run that cell. It reads the photodiode signal and stimulus MAT file, applies
the stored correction, checks the trial count, appends the last boundary,
and writes `on_list_times.npy`.

The correction database is at the mouse level:

```text
/mnt/senzailab/Kai/#Recording/m20/session_slices.sqlite3
```

Despite that filename, the current notebook uses `Utils.session_edits`.
Records can delete detected edges and interpolate an interval. Delete indices
refer to the original detected sequence; interpolation indices refer to the
sequence after deletion. Apply each record starting from raw detection.

Look for `Stored edits applied:` and confirm the date and session. The cell
writes its output immediately after its count check.
`is_overwrite_parameters` is not an onset-file preview switch.

Without a saved edit, this notebook uses a trimming branch based on detected
abnormal intervals. Verify the resulting start and end against the trace;
matching the trial count does not prove the correct block was selected.
Use the preview route for a new periodic session, or establish the correct
trim before saving a manual correction.

Run the later interval-printing and **ADC / Digital / onset-marker** plot
cells. Inspect the start, end, and abnormal intervals. Change
`start_time, end_time` in the plot cell to inspect another interval.

For regular squares or bars, skip the cell beginning `import pandas as pd`
that reads HD and Motive positions. Skip the historical patching and channel-11
experiment cells near the bottom. **Do not use Run All on this notebook**:
those cells include other recordings and additional save operations.

## 6. Check the MATLAB inputs

Run this cell in a remote notebook under `~/Developer/rfmapping`. It reads
the selected files and saves nothing. Change the four settings together.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

mouse = "m20"
date = "260918"
session_id = 2
probe = "A"

session_dir = (
    Path("/mnt/senzailab/Kai/#Recording")
    / mouse / date / f"{date}_{session_id}"
)
sorting_dir = session_dir / "kilosort" / f"Probe{probe}" / f"kilosort_{session_id}"

trials = loadmat(session_dir / f"{date}.mat")["trials"].ravel()
onsets = np.load(session_dir / "data/on_list_times.npy").ravel()
spike_seconds = np.load(
    session_dir / "data" / f"probe{probe}" / "adc_spike_time.npy",
    mmap_mode="r",
).ravel()
spike_clusters = np.load(sorting_dir / "spike_clusters.npy", mmap_mode="r").ravel()
spike_samples = np.load(sorting_dir / "spike_times.npy", mmap_mode="r").ravel()
labels = pd.read_csv(sorting_dir / "cluster_KSLabel.tsv", sep="\t")

assert onsets.size == trials.size + 1, (onsets.size, trials.size)
assert np.isfinite(onsets).all() and (np.diff(onsets) > 0).all()
assert spike_seconds.size == spike_clusters.size == spike_samples.size
assert spike_seconds.size > 0 and np.isfinite(spike_seconds).all()
assert (np.diff(spike_seconds) >= 0).all()
assert {"cluster_id", "KSLabel"}.issubset(labels.columns)

luminances = np.array([float(t["Square_Luminance"].item()) for t in trials])
good_ids = labels.loc[labels["KSLabel"] == "good", "cluster_id"].to_numpy()
present_good_ids = np.intersect1d(good_ids, np.unique(spike_clusters))
assert present_good_ids.size > 0

print("Session:", session_dir)
print("Trials / boundaries:", trials.size, onsets.size)
print("Luminance / trial count:", np.unique(luminances, return_counts=True))
print("Median stimulus interval (ms):", 1000 * np.median(np.diff(onsets)))
print("Onset range (seconds):", onsets[[0, -1]])
print("Spike range (seconds):", [spike_seconds.min(), spike_seconds.max()])
print("Spikes:", spike_seconds.size)
print("Good units with spikes in this session:", present_good_ids.size)
```

Confirm these points before MATLAB:

1. **Identity:** the printed path is the intended mouse, date, session, and
   probe. The stimulus log belongs to the same recording.
2. **Trial alignment:** N trials have N+1 boundaries, and the plotted boundaries
   identify the right stimulus block, not just a matching count.
3. **Spike alignment:** the three spike arrays have matching lengths and
   row order. Never independently sort or filter one while retaining the others.
4. **Time origin:** spike and onset times use matching recorded timestamp
   bases. Their ranges should plausibly overlap. Range overlap alone does not
   verify fine synchronization; inspect acquisition sync records and TTL
   timing if their relationship is uncertain. The current spike exporter
   does not fit the two clocks again.
5. **Polarity:** the selected luminance exists: `1` for ON or `0` for OFF.

The existing square-session ON map has 327 units. This is an example result,
not a required count for other recordings. Good-labeled IDs with no spikes
in the session do not create extra RF units.

## 7. Configure MATLAB for regular squares

### 7.1 Edit the actual generator

On the remote machine:

```sh
nano /mnt/ssd4.1/Matlab/RFmapping.m
```

Edit the settings **inside the existing function**. A `params` variable in
the MATLAB Command Window does not override the function's internal settings.

Set these lines for the worked example:

```matlab
params.onlyReadGoodUnits = true;

params.isBackgroundMoving = false;
params.isAllocentricPixelBins = false;
params.isRotation = false;
params.isVerticalBar = false;
params.barBinWidthDeg = 3;       % Used only by the vertical-bar branch.
params.isFineResolution = false;
params.isUseRealCoordinate = true;

is_on = true;
is_off = false;

params.base_dir = '/mnt/senzailab/Kai/#Recording/m20/';
params.date = '260918';
params.probelist = 'A';
params.sessionList = '2';

params.VSTimeWindow = [-0.1 0.4];
timeBinWidthMs = 1;

params.total_deg = 360;
params.screenWidthPix = 960;
params.screenHeightPix = 240;
params.screenDeg = 360;
```

Keep the existing bin-count calculation, `rotationOffsetSign`, helper-path
setup, and calls to `RFmapping_core`. This block shows settings to edit; it
does not replace the whole function.

### 7.2 Understand the settings

| Setting | Effect |
| --- | --- |
| `base_dir` | Mouse directory; the trailing `/` is required by current path concatenation |
| `date`, `sessionList`, `probelist` | Select the input and output directories |
| `onlyReadGoodUnits=true` | Select `good` entries in `cluster_KSLabel.tsv` |
| `is_on=true` | Include trials with `Square_Luminance == 1` |
| `is_off=true` | Generate a separate map from `Square_Luminance == 0` trials |
| Coordinate flags all `false` | Use regular screen coordinates |
| `isVerticalBar=false` | Use regular square-stimulus geometry |
| `isUseRealCoordinate=true` | Use recorded x and y positions, including actual display offsets |
| `VSTimeWindow=[-0.1 0.4]` | Save 100 ms before through 400 ms after onset |
| `timeBinWidthMs=1` | Save 500 time bins across that window |
| Screen dimensions | Match the recorded 960×240-pixel, 360° display |

The square branch obtains square size from the first trial. Use it for the
corresponding constant-size square design. Selecting the bar branch is not a
general method of increasing square-map resolution.

The regular generator currently loops over characters in `sessionList`:
`'23'` means sessions **2 and 3**, not session 23. This example uses one
single-digit session. Do not enter a multi-digit session as a character
string and assume it selects that recording. The free-moving entry point
handles session strings differently, as shown below.

## 8. Run MATLAB

Add the RF `Utils` directory and FMAToolbox helpers from the bundled
`buzcode-master`. Adding every subdirectory under `/mnt/ssd4.1/Matlab`
can expose archived duplicates.

From the **remote shell**:

```sh
matlab -batch "cd('/mnt/ssd4.1/Matlab'); addpath('/mnt/ssd4.1/Matlab/Utils'); addpath(genpath('/mnt/ssd4.1/Matlab/buzcode-master/externalPackages/FMAToolbox')); RFmapping"
```

Alternatively, in the **remote MATLAB Command Window**:

```matlab
cd('/mnt/ssd4.1/Matlab')
addpath('/mnt/ssd4.1/Matlab/Utils')
addpath(genpath('/mnt/ssd4.1/Matlab/buzcode-master/externalPackages/FMAToolbox'))

which RFmapping -all
which readNPY -all
which Sync -all
which FindInInterval -all

RFmapping
```

The `which` results should resolve to the authoritative tree above. The
generator uses `parfor`; resolve any parallel-execution or license error
before treating the run as complete. The regular archive writer uses MATLAB's
Java support, so do not launch it with `-nojvm`.

Check the log's selected session and probe:

```text
Working on session: 260918_2
ProbeA
```

The configured example should also report 1 ms per bin and 500 bins. The run
counts spikes relative to each onset and pools them by spatial position. It
writes the `.rfmap`, then CSV and PDF exports. Exporting can continue after
the source file has been written.

The expected ON output is:

```text
/mnt/senzailab/Kai/#Recording/m20/260918/260918_2/
  data/rfmapping/good/-100_400_1ms/ProbeA/
  regular_unitsSpikeCounts_260918_2.rfmap
```

This is one continuous path; the breaks above are for readability. Check it:

```sh
ls -lh '/mnt/senzailab/Kai/#Recording/m20/260918/260918_2/data/rfmapping/good/-100_400_1ms/ProbeA/regular_unitsSpikeCounts_260918_2.rfmap'
```

Rerunning identical settings writes the same output path. Changing the time
window, probe, good/all selection, or ON/OFF selection changes the relevant
parts of the path.

## 9. Check the generated file

Run this in the remote RF notebook environment:

```python
from pathlib import Path
import numpy as np
from Utils.rfmap import load_rf_maps

rf_path = Path(
    "/mnt/senzailab/Kai/#Recording/m20/260918/260918_2"
    "/data/rfmapping/good/-100_400_1ms/ProbeA"
    "/regular_unitsSpikeCounts_260918_2.rfmap"
)

maps = load_rf_maps(rf_path, unit_firing_rate=False)
first = maps[0]
edges = first.time_bin_edges_s

print("File:", rf_path)
print("Shape (unit, y, x, time):", maps.shape)
print("First unit IDs:", maps.unit_ids[:5])
print("Time range (ms):", 1000 * edges[[0, -1]])
print("Time bin width (ms):", 1000 * np.diff(edges).mean())
print("x positions:", first.x_positions)
print("y positions:", first.y_positions)

assert np.isclose(edges[0], -0.1)
assert np.isclose(edges[-1], 0.4)
assert np.allclose(np.diff(edges), 0.001)
assert maps.shape[-1] == 500

summed = maps.sum(0.0, 0.2, show_progress=False)
print("Shape after summing 0–200 ms:", summed.shape)
```

The existing example produces `(327, 7, 30, 500)`; summing 0–200 ms produces
`(327, 7, 30, 1)`. Counts and grids differ by recording. This check does not
run RF detection or write a result sidecar.
The timing assertions match this example's window and bin width. If you
change those MATLAB settings, update the expected edges, width, and bin count
here, and choose a summation interval inside the saved window.

The current regular writer saves an indexed, compressed NPZ archive with a
**`.rfmap` extension**. Keep that extension. The reader detects file contents
and also accepts older JSON-based sources. Seeing binary data in a text
editor is expected for the current regular format.

For visual inspection, use a current stable RF Map Viewer that supports
indexed NPZ. On a Mac with the app installed, choose **File → Open** and select
the file through the mounted data share, or use:

```sh
open -a "RF Map Viewer" "/path/on/your/Mac/result.rfmap"
```

That command runs on your **Mac**, and its path must be accessible on the Mac;
a remote Linux `/mnt/...` path is not automatically a local Mac path. Inspect
one unit's 0–200 ms spatial map and complete timeline. An older JSON-only
viewer cannot read the archive just by renaming it to `.json`.

If the data share is not mounted on your Mac, copy just the finished file and
open it there. Run these commands in a **local Mac terminal**:

```sh
scp 'hhw9l84:/mnt/senzailab/Kai/#Recording/m20/260918/260918_2/data/rfmapping/good/-100_400_1ms/ProbeA/regular_unitsSpikeCounts_260918_2.rfmap' ~/Downloads/
open -a "RF Map Viewer" ~/Downloads/regular_unitsSpikeCounts_260918_2.rfmap
```

The installed stable app verified for this guide is version 1.10.1. Opening
the copied file is sufficient for RF/timeline inspection; companion data such
as waveforms or HD curves require their session files as well.

### What the numbers mean

- `unit_firing_rate=False` loads pooled spike counts. The source has no trial
  axis and has not undergone baseline subtraction.
- Default `load_rf_maps(rf_path)` divides counts by `occupancyTimeSec`, the
  total qualifying stimulus-display time at each position. That denominator
  remains display time when you choose a different response window.
- `sum(0.0, 0.2)` includes `[0, 200 ms)`; its arguments are **seconds**.
  Endpoints must match stored time edges.
- MATLAB's CSV/PDF exports sum the full generation window. They need not match
  a viewer displaying only 0–200 ms.
- With stimuli about 100 ms apart, the −100–400 ms window overlaps neighboring
  stimuli. Each trial's window is assigned to that trial's position. The same
  physical spike can appear relative to several onsets.

The `.rfmap` has now been generated and checked. RF masks, centers, and
significance tests are subsequent analyses; use the
[Python RF analysis guide](rfmap.md) when needed.

## 10. Generate another supported map

### OFF instead of ON

Check that the MAT file contains `Square_Luminance == 0`. In `RFmapping.m`:

```matlab
is_on = false;
is_off = true;
```

The example OFF result is:

```text
<session>/data/rfmapping_off/good/-100_400_1ms/ProbeA/
regular_unitsSpikeCounts_260918_2_off.rfmap
```

Set both flags to `true` to generate separate maps when the recording contains
both polarities. Keep the full ordered trial sequence in the timing file;
do not remove OFF onsets when generating ON.

### Vertical bars

For an actual bar recording, keep the three coordinate flags `false` and set:

```matlab
params.sessionList = '3';
params.isVerticalBar = true;
params.barBinWidthDeg = 3;
is_on = true;
is_off = false;
```

First prepare **session 3's own** sorting outputs, onset file, and stimulus
MAT file. Reuse the environment setup, not session 2's arrays.

The existing `m20/260918/session 3` is ON-only. Its output is:

```text
<session>/data/rfmapping/good/-100_400_1ms/ProbeA/
regular_unitsSpikeCounts_260918_3_vertical_bar_pooled_bin3deg.rfmap
```

Its shape is `(324, 1, 120, 500)`. The 3°, 6°, 9°, and 12° bar widths are pooled
by their covered positions. A 3° output grid does not mean all trials were
independent 3° bars. The branch checks that the selected polarity covers
every horizontal native screen pixel; selecting OFF for this ON-only session
fails that check.
This implementation expects the specific 960×240-pixel, 360° design, `Y=0`,
all horizontal centers from −174° through 174° in 12° steps, and all four
widths 3°, 6°, 9°, and 12°.

### Free-moving RF

Use this branch to transform stimulus locations using the animal's position
and head direction. Complete sorting and onset preparation for that session.
It also needs:

```text
<date>.csv                         same-recording Motive positions
<date>.calib                       geometry and headplate calibration
data/processed/head_direction.json
data/hd_trials_times.npy           N calibrated HD values in screen pixels
data/position_trials_time.npy      N × 2 eye-position coordinates
```

The upstream launcher can produce Motive head-direction output when the
session is in `FREE_MOVING_SESSIONS` and the matching CSV/TAK pair is present.
That route also runs other behavior analyses. The RF handoff needs the
same-session HD output, CSV, and calibration. Regular RF processing does not
create a calibration file.

In `matlab.ipynb`, configure and run that session's timing cell, then the
cell beginning `import pandas as pd`. Set `headplate_name` to the tracked
rigid body and `camera_input_channel` to the exposure-signal channel. The
current cell detects low camera pulses with threshold `11000`; those settings
must match the recording.

The cell aligns CSV/HD frames to camera exposure midpoints, unwraps and
interpolates calibrated HD to trial onsets, and estimates eye position using
the headplate-to-eye offset. It writes both trial arrays. Confirm finite
HD and XY for every trial and verify any frame trimming against the actual
camera/CSV alignment. The generic truncation branch does not establish
alignment merely by making frame counts equal.

HD is stored as `calibrated_degrees / 360 * 960`, not degrees or radians.
XY and calibration radius must use the **same units, origin, and axes**.
Some calibrations use Motive CSV units rather than physical millimeters;
the current MATLAB core does not convert these units. The existing calibration
key for screen diameter is spelled `screen_diamter`.

In `/mnt/ssd4.1/Matlab/RFmapping_fm.m`, edit:

```matlab
params.base_dir = '/mnt/senzailab/Kai/#Recording/m20/';
params.date = '260918';
params.sessionList = "10";
params.probelist = 'A';
params.onlyReadGoodUnits = true;
params.isFineResolution = false;
params.saveinjson = false;
params.VSTimeWindow = [-0.1 0.4];
timeBinWidthMs = 1;
```

Keep its existing bin-count calculation and screen settings. With the same
MATLAB paths as Section 8, run **`RFmapping_fm`** instead of `RFmapping`.
The current free-moving generator is ON-only and produces:

```text
<session>/data/rfmapping/good/-100_400_1ms/ProbeA/
regular_unitsSpikeCounts_260918_10free_moving.rfmap
```

This entry point currently writes **JSON text** inside the `.rfmap`.
`saveinjson=false` selects the extension; it does not convert the payload
to NPZ or HDF5. Use the Section 9 reader check with this file's path and its
expected spatial shape.

## 11. Resolve a failed step

| What happened | What to check |
| --- | --- |
| Raw pipeline cannot find a session | Check the complete tree at `/mnt/ssd4.1/<date>_<session>/` and the configured session list. |
| Notebook raises `StopIteration` finding the descriptor | Check `<session>/<date>/Record Node .../experiment1/recording1/structure.oebin`. Extra nesting or a different recording index breaks the notebook's glob. |
| Cannot find the stimulus MAT | Put the correct session log at `<session>/<date>.mat` and inspect its `trials` variable. |
| No detected stimulus edges | Check ADC channel, polarity, raw trace, and the threshold for the selected detector. |
| Onset count does not match trials | Inspect extra edges outside the stimulus block, missing edges, real long intervals, and the identity of the stimulus log. Preserve presentation order. |
| Manual timing selects an incorrect start/end | Inspect the detected abnormal intervals and trim. Use the preview route or a verified session-specific correction. |
| Corrected onsets change on rerun | Check which timing route wrote the file and which stored edits were applied. The two routes use different correction sequences. |
| Spike and cluster arrays differ in length | Use matching per-session sorting outputs and regenerate the singular timestamp file from that session's sample indices. |
| Spikes and onsets have different time ranges | Check session identity, original stream timestamps, and whether one file was zeroed. Do not subtract separate stream origins to force agreement. |
| MATLAB cannot find a helper | Add the specific `Utils` and FMAToolbox paths in Section 8; inspect `which ... -all`. |
| MATLAB processes the wrong session or stimulus | Edit the function's internal settings. Notebook variables do not update them. |
| No good units enter the map | Check `cluster_KSLabel.tsv` and whether those IDs occur in this session's spike labels. |
| Bar coverage assertion fails | Check the actual stimulus design, polarity, timing alignment, and horizontal coverage. |
| File exists but MATLAB reported an error | Load the file and inspect the failed stage. The source is written before CSV/PDF exports. |
| Viewer reports invalid JSON for a new archive | Use a viewer supporting indexed NPZ, or validate with `load_rf_maps`. Renaming does not convert the format. |
| Viewer and exported CSV/PDF differ | Match the time window and count/rate normalization; MATLAB exports sum the full generation window. |

Further analysis and the existing installation/test reference are in
[Analysis reference](analysis_reference.md). The RFMap API, trial-shuffle
procedure, masks, and centers are in [Python RF analysis](rfmap.md).
