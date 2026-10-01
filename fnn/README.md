# Minimal population FNN

Open [minimal_fnn.ipynb](minimal_fnn.ipynb) on `hhw9l84` and select
**Python (rfmapping)**. Run All from `~/Developer/rfmapping/fnn`; the local Mac copy
is for reading and editing. The notebook is self-contained.

The model follows the three-branch decoder in
[Ajabi et al. (2023), Extended Data Fig. 4d](https://www.nature.com/articles/s41586-023-05813-2/figures/9).
It predicts HD from population spikes and displays a learned polar radius.
The primary figure now shows polar views of the training reference, full RF
interval, held-out test interval, and radius-colored full interval. See
[PAPER_COMPARISON.md](PAPER_COMPARISON.md) for the figure/method differences
and the public-code search, including the separately published ZIG decoder.
Session and raw-file conventions come from
[`../../glm/minimal_glm.ipynb`](../../glm/minimal_glm.ipynb).

Default session: **m19 / 260827_10 / Probe A / hp4**.
Raw directory: `/mnt/senzailab/Kai/#Recording/m19/260827/260827_10`.
The GLM sample selects unit 261; this notebook uses the population of
`KSLabel == "good"` units with training rate at least 1 Hz.

| File under the session | Purpose |
|---|---|
| `data/session_info.json` | Resolve the raw ADC directory |
| ADC `timestamps.npy` and `continuous.dat` | Complete camera low-pulse midpoints, channel 1, threshold 11000 |
| `data/processed/head_direction.json` | hp4 HD and original frame indices |
| `260827.calib` | Session-specific HD zero: `screen_center` |
| `data/on_list_times.npy` | First/last stimulus times defining the analysis interval |
| `data/probeA/adc_spike_time.npy` | Spike times in ADC seconds |
| `kilosort/ProbeA/kilosort_10/spike_clusters.npy` | Cluster IDs aligned with the spike-time array |
| `kilosort/ProbeA/kilosort_10/cluster_group.tsv` | `cluster_id`, `KSLabel` quality labels |

The resolved ADC directory for this example is:

```text
/mnt/senzailab/Kai/#Recording/m19/260827/260827_10/260827/Record Node 102/experiment1/recording1/continuous/OneBox-107.OneBox-ADC
```

No trial MAT, Motive CSV, GLM cache, or external model helper is required.
Change settings in the first code cell to use a compatible session.
This sample requires its own calibration and the listed label schema.

The notebook uses nonoverlapping 100-ms count bins and midpoint HD, with
no temporal smoothing. The first 70% of time trains the model; the next
10% selects the stopping epoch; the last 20% is the final test. Each split
has a one-second guard. Unit selection and normalization use training only.
It compares FNN decoding with a constant circular-mean baseline, a linear
ridge decoder, and descriptive time-shift controls.

Results are written to `outputs/m19_260827_10_probeA/`: the primary
`paper_projection` and additional `validation` figures (both PNG/SVG),
`metrics.csv`, `manifest.json`, split/unit/training/control tables,
`samples.npz`, and `model.pt`. The last cell reloads saved raw counts and
weights and verifies the reconstructed test predictions.

This RF session is an adaptation for checking the decoder and file pipeline.
It does not establish cue-reset/drift behavior or a biological ring attractor.
The radius has arbitrary learned scale and is not uniquely identified as
physiological gain; the notebook explains the product ambiguity.

## Remote execution

The existing environment already contains the dependencies. Run interactively,
or execute and embed outputs with:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping/fnn && MPLBACKEND=Agg ~/.virtualenvs/rfmapping/bin/python -' <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path('minimal_fnn.ipynb')
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=600, kernel_name='rfmapping').execute()
nbformat.write(notebook, path)
PY
```

## Executed sample

Run All completed on `hhw9l84` on 2026-09-13 in approximately 6 seconds.
The notebook includes its executed tables and white-background figure.

- 159 good labels; 127 units pass the training-only 1-Hz threshold.
- 11,997 aligned bins; train/validation/test exposure is 839.7/118.9/238.9 s.
- Best validation epoch: 7; the fixed patience setting stops at epoch 27.

| Decoder | Test mean error | Test median error |
|---|---:|---:|
| FNN | 24.00° | 19.16° |
| Linear ridge | 24.37° | 20.98° |
| Constant training circular mean | 77.55° | 80.29° |

The median error over time-shift controls is 86.49°. FNN and linear decoding
perform similarly; this run does not establish a substantial FNN advantage.
The descriptive radius–population-rate correlation is only **r = 0.124**.
This raw mean-rate diagnostic does not implement the paper's tuning-dependent,
baseline-normalized gain estimator, so it does not test the paper's gain claim.

Verification passed for independently binned raw spikes from units 3, 243,
and 261, split guards, training normalization, an independent circular-error
calculation, source file size/mtime, opaque PNG output, and saved-model
prediction reconstruction. Exact results are in
[metrics.csv](outputs/m19_260827_10_probeA/metrics.csv) and
[manifest.json](outputs/m19_260827_10_probeA/manifest.json).
