# RF mapping: raw data → intermediate files → `.rfmap`

This guide explains the file structure first, then how to create each stage
of the data. Code and data can live on your workstation, a compute server,
or mounted storage. Configure their locations once; no particular hostname,
username, or disk mount is required.

The examples use `mouse_01`, date `260918`, session `2`, and Probe `A`.
Replace these identifiers with your recording. The main route produces a
regular square-stimulus ON map; OFF and vertical-bar variants follow.

## 1. Understand the data structure

| Stage | Contents | Purpose |
| --- | --- | --- |
| **Raw inputs** | Open Ephys recording, original timestamps/settings/events, stimulus log | Recover spikes, stimulus timing, and stimulus positions |
| **Intermediate files and cache** | Concatenated signal, sorting results, per-session spike arrays, checked stimulus boundaries | Reuse sorting and timing work before RF generation |
| **Final RF result** | `.rfmap` containing counts, coordinates, time edges, unit IDs, occupancy | View and analyze the RF response |

### 1.1 Raw files

```text
RECORDING_ROOT/
└── <mouse>/<date>/<date>_<session>/              the session directory
    ├── <date>.mat                              original stimulus log
    └── <date>/
        └── Record Node <number>/
            ├── settings.xml                    acquisition/probe geometry
            └── experiment1/recording1/
                ├── structure.oebin              stream/channel metadata
                ├── sync_messages.txt
                ├── continuous/
                │   ├── OneBox-<id>.ProbeA/
                │   │   ├── continuous.dat
                │   │   ├── timestamps.npy
                │   │   └── sample_numbers.npy
                │   └── OneBox-<id>.OneBox-ADC/
                │       ├── continuous.dat
                │       ├── timestamps.npy
                │       └── sample_numbers.npy
                └── events/                     original sync events
```

`RECORDING_ROOT` is a directory you choose. The relative nesting below it is
the layout expected by the notebooks and MATLAB reader. The repeated date
directory is intentional.

Keep the actual Record Node and OneBox names from acquisition. Retain
`settings.xml`, which the upstream reader uses for probe geometry. Copying
only `continuous.dat` loses timing and metadata needed by this workflow.
Retain the corresponding streams for every recorded probe.

The stimulus MAT file must belong to this exact session. It contains a
struct array named `trials`, one element per stimulus in presentation order:

| Field | Meaning |
| --- | --- |
| `Square_PositionX` | Horizontal screen/RF coordinate, degrees |
| `Square_PositionY` | Vertical screen/RF coordinate, degrees |
| `Square_Size` | Square size or bar width, degrees |
| `Square_Luminance` | `1` for white/ON; `0` for black/OFF |

The stimulus log says **what appeared and where**. The ADC photodiode trace
says **when it appeared**. Neural recordings alone cannot reconstruct a
missing stimulus log.

### 1.2 Sorting work directory and cache

The supplied sorter and splitter use this working layout:

```text
RFMAP_WORK_DIR/
├── <date>_2/Record Node .../                    raw sorting input
├── <date>_3/Record Node .../                    another session, if selected
└── Data/
    └── <sorting-group>/
        └── ProbeA/
            ├── concat/
            │   ├── traces_cached_seg0.raw       concatenated neural signal
            │   └── probe.json
            ├── kilosort/                      joint sorting results
            │   ├── spike_times.npy
            │   ├── spike_clusters.npy
            │   └── cluster_KSLabel.tsv
            ├── kilosort_2/                    split session-2 results
            └── kilosort_3/                    split session-3 results
```

The absolute working directory is configurable. The relative
`Data/<sorting-group>` layout is required by the current splitter.

For sessions `[2, 3]` on `260918`, the group is `260918_23`. For `[2]`, it is
`260918_2`. Groups containing a multi-digit session use an explicit list,
such as `260918_sessions-2-10`. Session order is concatenation order.

Changing the RF response window does not require sorting again. Reuse matching
sorting outputs. To compare the same units across sessions, sort those
recordings together; equal cluster numbers from independent sorts do not
identify the same neuron.

### 1.3 Per-session intermediate files

After sorting, splitting, and timing preparation, the session directory gains:

```text
<session directory>/
├── <date>.mat
├── kilosort/ProbeA/kilosort_<session>/
│   ├── spike_times.npy
│   ├── spike_clusters.npy
│   └── cluster_KSLabel.tsv
└── data/
    ├── on_list_times.npy
    └── probeA/adc_spike_time.npy
```

Let **N** be the trial count and **M** the spike count for one session/probe:

| File | Shape / contents | Meaning |
| --- | --- | --- |
| `spike_times.npy` | M integers | Sample indices in this session's probe recording |
| `spike_clusters.npy` | M integers | Cluster ID for each spike in matching row order |
| `cluster_KSLabel.tsv` | `cluster_id`, `KSLabel` columns | Unit labels; current filtering selects `good` |
| `data/probeA/adc_spike_time.npy` | M `float64` seconds | Same spikes on the recorded timestamp axis |
| `data/on_list_times.npy` | N+1 increasing `float64` seconds | N onsets followed by the final end boundary |

At index `j`, the sample index, timestamp, and cluster ID must refer to the
**same spike**. Never independently reorder just one of these arrays.

For trial `i`:

```text
trials[i]             gives stimulus identity and position
on_list_times[i]      gives its onset
on_list_times[i + 1]  closes its display interval
```

The final boundary is not an extra trial. MATLAB reads spike **seconds** from
`adc_spike_time.npy`, not Kilosort sample indices.
Keep these time arrays as `float64`: the current MATLAB interval helper
expects double-precision spike times and does not support a float32 cache.

Keep the capitalization: `kilosort/ProbeA` versus `data/probeA`.
Saved manual timing corrections, such as `session_slices.sqlite3`, are
analysis decisions to retain with the dataset, not disposable cache.

### 1.4 Final RF result

For good units, ON stimuli, a −100 to 400 ms window, and 1 ms bins:

```text
<session directory>/data/rfmapping/good/-100_400_1ms/ProbeA/
├── regular_unitsSpikeCounts_<date>_<session>.rfmap
├── regular_<date>_<session>.csv
└── regular/
    ├── 001_unit_<ID>.pdf
    └── ...
```

The conceptual array shape is **`(unit, y, x, time)`**. For 7 vertical positions,
30 horizontal positions, and this time window, it is
`(number_of_units, 7, 30, 500)`.

The current regular writer saves a compressed, indexed NPZ archive with the
`.rfmap` extension:

| Field | Meaning |
| --- | --- |
| `unitPool` | Recorded unit IDs, in display order |
| `xPositions`, `yPositions` | Spatial coordinates |
| `timeBinEdges` | T+1 edges in seconds relative to onset |
| `occupancyTimeSec` | Total qualifying display time at each `(y, x)` |
| `unit_<ID>` | One unit's counts, shape `(y, x, time)` |
| `metadata` | Dimensions, response units, format, geometry information |

Each key is stored as an NPY entry in the archive. The Python reader handles
these details. Do not rename a binary `.rfmap` to `.json`.

## 2. Follow the conversion order

```text
RAW
  Probe signal + geometry ──> concatenate and sort ──> sorting cache
                                                           |
                                                           v
                                                    split by session
                                                           |
  Probe timestamps + session sample indices ────────────────┤
                                                           v
                                                 spike seconds + IDs

  ADC photodiode + stimulus trials ──> checked N+1 boundaries

INTERMEDIATE
  spike seconds + cluster IDs + good labels + trials + boundaries
                                  |
                                  v
                          MATLAB RF aggregation
                                  |
                                  v
FINAL                          .rfmap
```

| Step | Input | Output | Check before continuing |
| --- | --- | --- | --- |
| Sort and split | Raw probe recordings and geometry | Session sample indices, cluster IDs, labels | Session order and sample counts match |
| Convert spike times | Sample indices and original probe timestamps | `adc_spike_time.npy` | Every time still matches its cluster ID |
| Detect timing | Photodiode and original `trials` | `on_list_times.npy` | Correct stimulus block, N+1 boundaries |
| Generate RF | All intermediate inputs | `.rfmap`, CSV, PDFs | MATLAB finishes and the source loads |
| Inspect | `.rfmap` | Spatial and temporal views | Unit IDs, geometry, and edges are correct |

With existing per-session sorting, start at
[spike-time conversion](#5-convert-spike-sample-indices-to-seconds).
With checked spike and stimulus timing, start at
[RF generation](#7-convert-the-intermediate-files-to-an-rf-map).

## 3. Configure paths and environments

Run processing on a machine that can access the files and has the required
runtime. SSH is optional. Shell examples use **Bash**; change the example
locations to your own directories.

### 3.1 Code you need

| Directory | Required contents |
| --- | --- |
| `RFMAP_CODE_DIR` | This repository: `Utils/`, `matlab.ipynb`, `matlab_auto.ipynb` |
| `RF_MATLAB_DIR` | `matlab/`: MATLAB generator, `Utils/`, required FMAToolbox sources |
| `PIPELINE_CODE_DIR` | `preprocessing/pipeline/`: upstream sorter with `pipeline/run_script.py` |
| `SORTING_CODE_DIR` | `preprocessing/spikeinterface/`: session splitter and spike-time exporter |

These directories are included in this repository. The sorter uses its own
Python environment, and MATLAB requires a separate installation. Existing
per-session sorting files allow you to skip the sorting stage.

```text
RF_MATLAB_DIR/
├── RFmapping.m
├── Utils/
│   ├── RFmapping_core.m
│   ├── readNPY.m
│   └── ...                         keep the complete helper directory
└── buzcode-master/externalPackages/FMAToolbox/
    ├── Analyses/
    ├── General/
    ├── Helpers/
    └── Plot/
```

### 3.2 Set paths and recording identity once

In the terminal used to launch Python, Jupyter, and MATLAB:

```bash
# Replace these locations with your code and data directories.
export RFMAP_CODE_DIR="$HOME/code/rfmapping"
export RF_MATLAB_DIR="$RFMAP_CODE_DIR/matlab"
export PIPELINE_CODE_DIR="$RFMAP_CODE_DIR/preprocessing/pipeline"
export SORTING_CODE_DIR="$RFMAP_CODE_DIR/preprocessing/spikeinterface"
export RFMAP_WORK_DIR="$HOME/data/rf_work"
export RECORDING_ROOT="$HOME/data/recordings"

# Replace these values with your recording identifiers.
export RF_MOUSE="mouse_01"
export RF_DATE="260918"
export RF_SESSION="2"
export RF_SESSIONS="2"
export RF_PROBES="A"

export RF_MOUSE_DIR="$RECORDING_ROOT/$RF_MOUSE"
export RF_SESSION_DIR="$RF_MOUSE_DIR/$RF_DATE/${RF_DATE}_${RF_SESSION}"
export RF_SORT_CONFIG="$RFMAP_WORK_DIR/rf_sorting.yaml"
```

`RF_SESSION` is the session being mapped. `RF_SESSIONS` is the ordered sorting
group: use `"2 3"`, for example, to sort both recordings together.
`RF_PROBES` is `A` or `AB`, not `ProbeA`.

These variables are settings for the examples, not new configuration options
in every script. The commands explicitly pass their values into the code.
Set them again in a new terminal and keep your configuration with the analysis.

### 3.3 Prepare the RF Python environment

Use Python **3.12 or newer**. Create an environment if you do not already have
one for this code:

```bash
python3 -m venv "$RFMAP_CODE_DIR/.venv"
export RFMAP_PYTHON="$RFMAP_CODE_DIR/.venv/bin/python"
"$RFMAP_PYTHON" -m pip install -e "${RFMAP_CODE_DIR}[analysis]" jupyterlab ipykernel
```

For an existing environment, set `RFMAP_PYTHON` to its executable instead:

```bash
export RFMAP_PYTHON="/absolute/path/to/your/environment/bin/python"
```

Register a named kernel for the chosen environment, then start Jupyter from
the RF repository:

```bash
"$RFMAP_PYTHON" -m ipykernel install --user --name rfmapping --display-name "RF mapping"
cd "$RFMAP_CODE_DIR"
"$RFMAP_PYTHON" -m jupyterlab
```

Select the **RF mapping** kernel, then check it in a cell:

```python
import os
import sys
print(sys.executable)
print(os.environ["RF_SESSION_DIR"])
```

The executable must belong to the RF environment and the session path must
match your selection. Launching Jupyter from the configured terminal makes the
exported settings available to its kernel. Press **Shift+Enter** to run a cell.

Jupyter occupies this terminal while it runs. Use a second terminal for later
shell commands, repeating the path/recording exports and interpreter settings
there. Keep the two terminals configured for the same recording.

When switching sessions, update the full settings block, including the derived
`RF_MOUSE_DIR` and `RF_SESSION_DIR`. Stop the old Jupyter server and relaunch it
from the newly configured terminal; restarting only its kernel does not update
environment variables inherited by the existing server. Relaunch an existing
interactive MATLAB session as well if it reads settings with `getenv`.
Then rerun the notebook setup and check the printed session path.

### 3.4 Prepare sorting and MATLAB runtimes

The supplied sorter uses Kilosort with CUDA. Use a supported GPU machine and
follow its [installation instructions](preprocessing/pipeline/README.md).
For a checkout using its supplied `uv` setup:

```bash
cd "$PIPELINE_CODE_DIR"
uv sync
export PIPELINE_PYTHON="$PIPELINE_CODE_DIR/.venv/bin/python"
"$PIPELINE_PYTHON" -c 'import torch; print(torch.cuda.is_available())'
```

The CUDA check should return `True`. With another valid installation, point
`PIPELINE_PYTHON` at that environment instead. The RF reader and the
NumPy-based splitter do not require a GPU.

MATLAB needs Java support, the parallel execution used by `parfor`, graphics
export, and a working FMAToolbox MEX helper for its platform. See the first-run
check in [Section 7](#7-convert-the-intermediate-files-to-an-rf-map).

### 3.5 Linux/macOS and Windows

The shell examples use Bash, and the manual timing database connection uses
the Linux/macOS `unix-dotfile` locking method. Keep that connection on
Linux/macOS. A Windows computer running a remote Linux kernel also uses the
Linux settings; the kernel's operating system determines the paths and runtime.

For **native Windows Python**, use PowerShell environment variables and the
environment's `Scripts/python.exe`, change the notebook's data root, and make
the documented SQLite connection-line change in your Windows checkout before
opening `matlab.ipynb`. The exact instructions are in
[Platform setup and preserving existing data](docs/platforms_and_data_safety.md).
Changing the data path alone does not fix the SQLite locking selection.

**Do not overwrite or recreate an existing `session_slices.sqlite3` when moving
between computers.** The database format is portable; shared-drive locking and
the selected connection method are the compatibility issue. Preserve a closed
backup, transfer to a new destination, and confirm the database path before
running the timing cell. The platform guide includes a copy command that refuses
to replace an existing file.

## 4. Convert raw probe recordings to per-session sorting files

Skip this section if matching per-session Kilosort outputs already exist.
This route calls the configurable pipeline and splitter directly.

### 4.1 Arrange raw inputs and the session copy

For every session in `RF_SESSIONS`, put the complete raw recording at:

```text
<RFMAP_WORK_DIR>/<date>_<session>/Record Node .../
```

There is no inner date folder in this working layout. Copy each raw tree into
the session layout used by the RF notebooks:

```bash
for session_id in $RF_SESSIONS; do
    source_dir="$RFMAP_WORK_DIR/${RF_DATE}_${session_id}"
    target_dir="$RF_MOUSE_DIR/$RF_DATE/${RF_DATE}_${session_id}/$RF_DATE"
    mkdir -p "$target_dir"
    rsync -rlt --info=progress2 "$source_dir/" "$target_dir/"
done
```

Place each session's original stimulus log at:

```text
<RECORDING_ROOT>/<mouse>/<date>/<date>_<session>/<date>.mat
```

If raw files already live in the session layout, copy the inner date directory
in the other direction to prepare the sorter's working input. Retain the full
Record Node tree. The current splitter expects `OneBox-*.ProbeA` naming;
other acquisition formats require an appropriate upstream conversion.

### 4.2 Write the sorting configuration

The following command writes YAML containing actual expanded paths. YAML
does not automatically expand shell variables entered into its values.

```bash
mkdir -p "$RFMAP_WORK_DIR"
"$PIPELINE_PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml

work = Path(os.environ["RFMAP_WORK_DIR"])
date = os.environ["RF_DATE"]
sessions = [int(value) for value in os.environ["RF_SESSIONS"].split()]
tag = (
    "".join(str(value) for value in sessions)
    if all(1 <= value <= 9 for value in sessions)
    else "sessions-" + "-".join(str(value) for value in sessions)
)
config = {
    "run_name": f"{date}_{tag}",
    "session_paths": [str(work / f"{date}_{value}") for value in sessions],
    "local_output": str(work / "Data"),
    "remote_output": None,
    "per_shank": False,
    "copy_mode": "newer",
    "target_fs": None,
    "verbose": True,
    "overwrite": False,
    "job_kwargs": {
        "n_jobs": 1,
        "chunk_duration": "2s",
        "progress_bar": True,
        "mp_context": "spawn",
    },
    "openblas_threads": 1,
}
config_path = Path(os.environ["RF_SORT_CONFIG"])
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(config_path.read_text())
PY
```

Check the printed session paths and order.

- `local_output` must be `RFMAP_WORK_DIR/Data`, where the splitter will look.
- `run_name` must follow the group naming in Section 1.2. The example computes
  the matching name.
- `per_shank=False` gives the `kilosort/` input layout expected by this splitter.
- `target_fs=None` skips EEG downsampling, which RF mapping does not require.
- `remote_output=None` disables an additional archive copy by the pipeline.
  It does not disable sorting.

### 4.3 Run sorting

```bash
cd "$PIPELINE_CODE_DIR"
"$PIPELINE_PYTHON" -m pipeline.run_script "$RF_SORT_CONFIG"
```

The configuration path is a positional argument, not a `--config` option.
The command loads Open Ephys streams, caches concatenated signals, runs
sorting, and processes its synchronization stage. Preserve the ADC stream and
sync events for that stage even though EEG export is disabled.

For the one-session example, verify these outputs:

```text
<RFMAP_WORK_DIR>/Data/260918_2/ProbeA/
├── concat/traces_cached_seg0.raw
├── concat/probe.json
└── kilosort/
    ├── spike_times.npy
    ├── spike_clusters.npy
    └── cluster_KSLabel.tsv
```

Here `spike_times.npy` indexes the **concatenated** recording. With multiple
sessions, these indices are not yet valid single-session sample indices.

### 4.4 Split by session

Use exactly the same ordered session list as the sorting configuration:

```bash
cd "$SORTING_CODE_DIR"
"$RFMAP_PYTHON" - <<'PY'
import os
from pathlib import Path
from split_kilosort_by_recording import split_kilosort_by_recording

split_kilosort_by_recording(
    data_base_folder=Path(os.environ["RFMAP_WORK_DIR"]),
    remote_output=Path(os.environ["RF_MOUSE_DIR"]) / os.environ["RF_DATE"],
    date=os.environ["RF_DATE"],
    sessions=[int(value) for value in os.environ["RF_SESSIONS"].split()],
    probe_list=list(os.environ["RF_PROBES"]),
    num_channels=384,
    dtype_bytes=2,
    fs=30000,
    overwrite_outputs=False,
)
PY
```

The values `384`, `2`, and `30000` describe a 384-channel, int16, 30 kHz probe
recording. Confirm them against your recording. Incorrect channel count or
sample size gives incorrect session boundaries.

This splitter is a function. Running `python split_kilosort_by_recording.py`
alone does not perform splitting.

It measures recording lengths, checks their total against the cached
concatenation, splits aligned spike arrays, and subtracts each session's
sample offset. Cluster IDs remain unchanged. Outputs are copied to:

```text
<session directory>/kilosort/ProbeA/kilosort_<session>/
```

The argument `remote_output` can be a local path; its name does not require
network storage. Destination copies can replace existing files.
`overwrite_outputs=False` protects the local split directory, not all
destination copies. Reuse the existing sorting group when reproducing an
analysis instead of replacing its unit identities with another sorting run.

The RF generator reads `good` labels from `cluster_KSLabel.tsv`.
Changing only `cluster_group.tsv` does not change its current unit selection.

## 5. Convert spike sample indices to seconds

After the raw session copy and split sorting output exist:

```bash
"$RFMAP_PYTHON" "$SORTING_CODE_DIR/generate_adc_spike_time.py" \
    "$RF_MOUSE_DIR" "$RF_DATE" "$RF_SESSION" "$RF_PROBES"
```

The arguments are **mouse directory, date, session ID, probe letters**.
Repeat for other sessions as needed.

For each spike, the exporter performs:

```text
adc_spike_time[j] = original_probe_timestamps[session_spike_samples[j]]
```

It writes `<session>/data/probeA/adc_spike_time.npy` in the original spike
order. Its default preserves the recorded timestamp origin; do not pass
`--convert-to-zero` with the timing setup below.

This script does not fit a new clock transformation. The probe timestamps and
ADC timestamps used for stimuli must already have a valid shared timing
relationship. Check acquisition synchronization if that is uncertain.
Subtracting a separate start time from each stream does not synchronize them.

Do not substitute the concatenated pipeline's **`adc_spike_times.npy`**
(plural) for the **per-session `adc_spike_time.npy`** (singular). Their
construction and timing provenance differ.

## 6. Convert the photodiode trace into trial boundaries

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
Open the notebook from `RFMAP_CODE_DIR` in the Jupyter session started in
Section 3, which inherits the exported path settings.
The 0.1-second final interval is specific to this stimulus design; use the
actual final display duration for a different design.

### 6.1 New periodic-stimulus session: preview with `matlab_auto.ipynb`

**This is preview code. Reviewing its result is a critical step. After it
finishes, inspect the detected onsets against the photodiode trace before
saving or continuing to RF generation. A completed run and a matching trial
count do not establish correct timing.**

Open [matlab_auto.ipynb](matlab_auto.ipynb). This notebook handles the
approximately periodic stimulus-edge sequence used here. It cleans detected
transitions, trims the surrounding recording, and evaluates missing-edge
repairs against timing and trial-count constraints.

In the cell beginning `Change all the variables below to match recording`,
set:

```python
import os

base_dir = os.environ["RF_MOUSE_DIR"]
date = os.environ["RF_DATE"]
num_of_rec = int(os.environ["RF_SESSION"])
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

The same rule applies to every preview or diagnostic plot used in this
workflow: inspect the result after it finishes, especially the first and last
stimuli and any repaired interval. Resolve an incorrect preview before saving
or moving to the next step.

### 6.2 Existing stored timing corrections: `matlab.ipynb`

Open [matlab.ipynb](matlab.ipynb) and read its first Markdown cell. Before running
the main parameter cell, preserve the existing mouse database and timing files
as described in the [database instructions](docs/platforms_and_data_safety.md#2-preserve-and-check-the-existing-database).
Run the imports, then configure the main parameter cell:

```python
import os

base_dir = os.environ["RF_MOUSE_DIR"]
date = os.environ["RF_DATE"]
num_of_rec = int(os.environ["RF_SESSION"])
analogue_input_channel = 3
is_convert_to_zero = False
is_signal_inverted = True
```

Run that cell. It reads the photodiode signal and stimulus MAT file, applies
the stored correction, checks the trial count, appends the last boundary,
and writes `on_list_times.npy`.

The correction database is at the mouse level:

```text
<RECORDING_ROOT>/<mouse>/session_slices.sqlite3
```

Despite that filename, the current notebook uses `Utils.session_edits`.
Records can delete detected edges and interpolate an interval. Delete indices
refer to the original detected sequence; interpolation indices refer to the
sequence after deletion. Apply each record starting from raw detection.

Check that this is the **existing database for the selected mouse** before
opening it. A missing path causes `check_session_edits()` to create a new
database. An empty database at a mistyped path does not recover earlier
corrections. A Windows connection error or legacy-schema error is not a reason
to delete, initialize over, or replace the original database.

Look for `Stored edits applied:` and confirm the date and session. The cell
writes its onset output immediately after its count check and can store new
corrections when no record exists. `is_overwrite_parameters` protects neither
the database nor `on_list_times.npy`; this cell has no preview switch.

Without a saved edit, this notebook uses a trimming branch based on detected
abnormal intervals. Verify the resulting start and end against the trace;
matching the count does not prove the correct trial block was selected.
Use the preview route for a new periodic session, or establish the correct
trim before saving a manual correction.

Run the later interval-printing and **ADC / Digital / onset-marker** plot
cells. Inspect the start, end, and abnormal intervals. Change
`start_time, end_time` in the plot cell to inspect another interval.

Run only the timing and inspection cells described above. **Do not use Run All
on a historical copy of this notebook**: later experiment cells can select
another recording or overwrite a timing file.

## 7. Convert the intermediate files to an RF map

Use `RFmapping_core(params)` with the session configuration below.

### 7.1 Check MATLAB helpers once

Start MATLAB from a terminal containing the exported settings:

```bash
matlab
```

In the MATLAB Command Window:

```matlab
RF_MATLAB_DIR = getenv('RF_MATLAB_DIR');
fmatDir = fullfile(RF_MATLAB_DIR, 'buzcode-master', ...
    'externalPackages', 'FMAToolbox');

addpath(genpath(fmatDir))
addpath(fullfile(RF_MATLAB_DIR, 'Utils'))
addpath(RF_MATLAB_DIR)

which RFmapping_core -all
which readNPY -all
which Sync -all
which FindInInterval -all
which PlotColorMap -all
mexext
```

These names must resolve to the supplied MATLAB tree. Add the specific
directories above instead of recursively adding an entire tree containing
archived copies.

`FindInInterval` needs a working MEX binary for the current MATLAB platform.
The adjacent `FindInInterval.m` is documentation, not a replacement
implementation. If no compatible binary is available, compile the supplied
source with a compiler supported by your MATLAB installation:

```matlab
mex -setup C
intervalDir = fullfile(fmatDir, 'General');
mex('-outdir', intervalDir, fullfile(intervalDir, 'FindInInterval.c'));
rehash
which FindInInterval -all
```

The regular archive writer uses Java, so keep the JVM enabled. The generator
also uses `parfor` and figure export; resolve runtime/toolbox errors before
treating a run as complete.

### 7.2 Create a session configuration

Save the following as `run_rf.m` in your working directory
(`RFMAP_WORK_DIR`). Create that directory first if you skipped sorting.

```matlab
% Code locations inherited from the terminal that launched MATLAB.
RF_MATLAB_DIR = getenv('RF_MATLAB_DIR');
fmatDir = fullfile(RF_MATLAB_DIR, 'buzcode-master', ...
    'externalPackages', 'FMAToolbox');
addpath(genpath(fmatDir));
addpath(fullfile(RF_MATLAB_DIR, 'Utils'));
addpath(RF_MATLAB_DIR);

% Recording identity.
params.base_dir = [fullfile(getenv('RECORDING_ROOT'), ...
    getenv('RF_MOUSE')), filesep];
params.date = getenv('RF_DATE');
params.sessionList = getenv('RF_SESSION');
params.probelist = getenv('RF_PROBES');

% Good units and regular square-stimulus geometry.
params.onlyReadGoodUnits = true;
params.isBackgroundMoving = false;
params.isAllocentricPixelBins = false;
params.isRotation = false;
params.rotationOffsetSign = +1;
params.isVerticalBar = false;
params.barBinWidthDeg = 3;
params.isFineResolution = false;
params.isUseRealCoordinate = true;

% Match these values to the stimulus display.
params.total_deg = 360;
params.screenWidthPix = 960;
params.screenHeightPix = 240;
params.screenDeg = 360;

% Relative-time window: seconds. Time-bin width: milliseconds.
params.VSTimeWindow = [-0.1 0.4];
timeBinWidthMs = 1;
nbinsExact = diff(params.VSTimeWindow) * 1000 / timeBinWidthMs;
assert(abs(nbinsExact - round(nbinsExact)) < 1e-9, ...
    'Window width must be divisible by time-bin width.');
params.nbins = round(nbinsExact);

% ON = 1; OFF = 0.
params.lum = 1;
RFmapping_core(params);
```

Use the display dimensions for your actual stimulus. The worked values
describe the 960×240-pixel, 360° setup. The regular square branch expects a
constant square size and reads it from the first trial.

`isUseRealCoordinate=true` uses the recorded x/y values, including actual
display offsets. All three movement/coordinate flags remain false for an
ordinary screen-coordinate map.

The current regular core loops through characters in `sessionList`.
`'23'` selects sessions **2 and 3**, not session 23. This example uses the
single-digit session `2`; a multi-digit regular session is not supported by
simply entering that number as a character string.

If you prefer the existing `RFmapping.m` wrapper, edit its internal settings
instead. Creating a `params` variable and then calling `RFmapping` does not
override those internal settings. The self-contained example above must end
with **`RFmapping_core(params)`**.

### 7.3 Generate the result

From the configured terminal:

```bash
matlab -batch "run(fullfile(getenv('RFMAP_WORK_DIR'), 'run_rf.m'))"
```

For the example identifiers, check the log reports:

```text
Working on session: 260918_2
ProbeA
```

The configured window has 500 one-millisecond bins. The generator writes:

```text
<RF_SESSION_DIR>/data/rfmapping/good/-100_400_1ms/ProbeA/
regular_unitsSpikeCounts_260918_2.rfmap
```

The general filename is `regular_unitsSpikeCounts_<date>_<session>.rfmap`.
The lines above form one path. For the first selected probe, check it with:

```bash
probe_letter="${RF_PROBES:0:1}"
ls -lh "$RF_SESSION_DIR/data/rfmapping/good/-100_400_1ms/Probe$probe_letter/regular_unitsSpikeCounts_${RF_DATE}_${RF_SESSION}.rfmap"
```

The `.rfmap` is written before the CSV/PDF exports. Wait for the full command
to finish; a later export error can leave an already-written source file.
Rerunning the same configuration writes the same result paths.

### 7.4 What this conversion calculates

For every unit and trial, the mapper counts spikes in time bins relative to
that trial's onset. It then sums those trial histograms at the matching
spatial position:

```text
counts[unit, position, time_bin]
    = sum of per-trial counts over matching trials

occupancyTimeSec[position]
    = sum of display durations over those same trials
```

Matching includes the selected luminance. Regular squares match the recorded
x/y location; bar maps use the stimulus footprint.

The saved values are pooled spike counts. There is no trial axis, baseline
subtraction, or trial averaging in this output. Occupancy is stimulus
**display time**, not the summed length of response-analysis windows.

With stimuli about 100 ms apart, a −100 to 400 ms window spans neighboring
stimuli. The same physical spike can appear relative to several onsets,
each assigned to that trial's position. Negative or late activity requires
interpreting this overlap.

## 8. Read and inspect the final result

In the RF notebook, load the generated file for the first selected probe:

```python
import os
from pathlib import Path
import numpy as np
from Utils.rfmap import load_rf_maps

session_dir = Path(os.environ["RF_SESSION_DIR"])
date = os.environ["RF_DATE"]
session_id = os.environ["RF_SESSION"]
probe = os.environ["RF_PROBES"][0]
rf_path = (
    session_dir
    / "data/rfmapping/good/-100_400_1ms"
    / f"Probe{probe}"
    / f"regular_unitsSpikeCounts_{date}_{session_id}.rfmap"
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

summed = maps.sum(0.0, 0.2, show_progress=False)
print("Shape after summing 0–200 ms:", summed.shape)
```

For a `(U, H, W, 500)` source, summing gives `(U, H, W, 1)`.
Unit count and spatial grid depend on your recording. If you changed the
generation window or bin width, update the folder name to match. Choose a
summation interval inside the saved window.

- `unit_firing_rate=False` preserves stored spike counts.
- Default `load_rf_maps(rf_path)` divides counts by `occupancyTimeSec`.
  That denominator remains display time when a different response window is
  selected.
- `sum(0.0, 0.2)` means `[0, 200 ms)`. Arguments are **seconds**, and both
  endpoints must match stored time edges.
- MATLAB CSV/PDF exports sum the full generated time window, so they need not
  match a 0–200 ms view.

For visual inspection, open the file in an RF Map Viewer version supporting
indexed NPZ `.rfmap` files. Use the viewer code's own installation/launch
instructions, then **File → Open**. The file must be accessible to the machine
running the viewer. Older JSON-only viewers cannot read a binary archive by renaming
its extension.

## 9. Variations

Start each variant from the complete configuration in Section 7.2 and change
the specified settings. Prepare the intermediate inputs for the selected
session first.

### OFF or both polarities

Check the stimulus log contains `Square_Luminance == 0`. To generate OFF,
change the final call in `run_rf.m` to:

```matlab
params.lum = 0;
RFmapping_core(params);
```

The result is:

```text
<session>/data/rfmapping_off/good/-100_400_1ms/ProbeA/
regular_unitsSpikeCounts_<date>_<session>_off.rfmap
```

To generate both, call the core once with `params.lum=1` and once with
`params.lum=0`. Preserve the full ordered stimulus sequence in
`on_list_times.npy`; do not remove OFF onsets when generating an ON map.

### Vertical bars

Select the actual bar session in the recording settings. In the MATLAB
configuration, keep the three coordinate flags false and set:

```matlab
params.isVerticalBar = true;
params.barBinWidthDeg = 3;
params.lum = 1;
RFmapping_core(params);
```

This implementation expects the specific 960×240-pixel, 360° design:
`Y=0`, all centers from −174° to 174° in 12° steps, and all widths 3°, 6°,
9°, and 12°. The selected polarity must cover every horizontal native pixel.

The output is:

```text
<session>/data/rfmapping/good/-100_400_1ms/ProbeA/
regular_unitsSpikeCounts_<date>_<session>_vertical_bar_pooled_bin3deg.rfmap
```

Widths are pooled by their covered positions. The 3° grid does not imply
independent 3° bars on every trial, and this pooling does not deconvolve the
different widths. ON-only recordings cannot produce an OFF bar map without
the missing OFF trials.

## 10. Reuse intermediates and diagnose failures

### What must be regenerated?

| Change | Work to repeat |
| --- | --- |
| Response window or RF time-bin width | MATLAB RF generation; reuse sorting and checked onset times |
| ON/OFF selection or good-unit inclusion | RF generation using the complete original trial sequence and matching labels |
| Corrected stimulus boundaries | Inspect and save the corrected onset times, then regenerate RF |
| New sorting or changed spike rows | Recreate matching per-session arrays and spike seconds, then RF |

Keep the raw data, original trial log, sorting-group order, configuration,
and manual timing corrections. Derived files are reusable only with the
inputs and settings that produced them.

### Existing files that a rerun can replace

There is no single overwrite flag covering the pipeline. Check the target
paths before running a save cell, and preserve any result you need to retain.

| Writer | Existing data affected | Control |
| --- | --- | --- |
| `matlab.ipynb` main cell | Mouse-level correction records and session `on_list_times.npy` | Correction methods can update records; onset output is saved unconditionally after validation. `is_overwrite_parameters` does not protect them. |
| `matlab_auto.ipynb` final cell | The same session `on_list_times.npy` | Keep `is_save_on_list_time=False` for preview; `True` replaces an existing file. It does not read or update the correction database. |
| Spike-time exporter | `data/probeA/adc_spike_time.npy` | Regeneration writes the same target. Preserve a prior timestamp alignment before changing its method. |
| MATLAB RF generator | Same-named `.rfmap`, CSV and PDF outputs | Identical output paths are reused on a rerun. Preserve an earlier run before generating a replacement. |
| Optional Python RF detection | `.npz` at `result_path` | A different valid cache key can trigger recalculation and replacement; use a distinct result path to retain alternatives. Loading or summing a `.rfmap` alone does not write this sidecar. |

The [platform guide](docs/platforms_and_data_safety.md#3-other-path-and-cache-assumptions)
covers filename capitalization, reused paths, and platform-specific MATLAB
binaries.

### Common failures

| Symptom | Check |
| --- | --- |
| Sorting cannot locate raw input | Each configured session path must contain the complete Record Node tree, including `settings.xml`. |
| Splitter cannot locate concatenation | Check `RFMAP_WORK_DIR/Data/<group>/ProbeA`, matching run-name rules and identical session order. |
| Session sample counts do not sum to the cache length | Verify raw files, channel count, sample dtype, group order, and the cached concatenation's identity. |
| Notebook cannot find `structure.oebin` | The current notebooks search `<session>/<date>/*/experiment1/recording1/structure.oebin`. Check nesting and recording index. |
| Notebook lacks an environment variable | Launch Jupyter/kernel with the settings from Section 3, or configure the same values explicitly in its parameter cell. |
| Onset and trial counts disagree | Inspect extra/missing transitions, real long intervals, the selected signal, and the exact session's stimulus log. |
| Manual timing selects an incorrect start/end | Inspect its detected abnormal intervals and trim; use the preview route or a verified session-specific correction. |
| SQLite reports `no such vfs: unix-dotfile` or fails to initialize on Windows | Apply the native Windows connection change in the platform guide; retain the existing database. |
| Stored corrections disappear after moving computers | Check the exact mouse-level database path, saved date/session keys, and whether a new empty database was created at another location. Do not copy over either database. |
| Spike and cluster arrays differ in length | Use outputs from the same split sorting run and regenerate the per-session spike seconds. |
| Spikes and onsets use different time ranges | Check session identity, original timestamps, and independent zeroing; do not shift streams just to force overlap. |
| MATLAB loads the wrong helper | Check `which ... -all` against `RF_MATLAB_DIR` and remove archived duplicates from the path. |
| `FindInInterval` is unavailable | Use a compatible MEX binary or compile the supplied C source for the active MATLAB installation. |
| Wrong session processed | Check the environment and `params`; remember that `RFmapping` resets its own settings and the regular core loops through session characters. |
| No good units appear | Check `cluster_KSLabel.tsv` and whether those IDs have spikes in this session. |
| Bar coverage assertion fails | Verify the supported geometry, selected luminance, timing alignment, and actual horizontal coverage. |
| `.rfmap` exists after a failed run | Load the source and inspect the failed stage; CSV/PDF export occurs after source writing. |
| Viewer expects JSON | Use an indexed-NPZ-capable viewer or the Python reader; renaming does not convert storage. |
