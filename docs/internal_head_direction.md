# Internal head-direction decoding

Run `decode_internal_head_direction.py` on `hhw9l84` with the rfmapping virtualenv
(verified with Pynapple 0.11.3, using `compute_tuning_curves` and `decode_bayes`):

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && \
  ~/.virtualenvs/rfmapping/bin/python decode_internal_head_direction.py \
  "/mnt/senzailab/Kai/#Recording/m19/260827/260827_10" --probe A'
```

The default output is `SESSION/data/processed/internal_head_direction.json`:

```json
{
  "internal_direction": {
    "frames": [0, 1, 2],
    "head_direction_deg": [357.0, 357.0, null]
  }
}
```

The script preserves every source frame ID and its order. Values are the
Pynapple Bayesian decoder's maximum-posterior angle in degrees, using the same
angular convention as the tracking input. A frame receives the estimate of its
containing spike-count bin; angles are not interpolated. The tracking input is
never overwritten.

Defaults:

- Select good Kilosort units with saved `hd_class == 3` from
  `data/tuning_curves/ProbeA/tuning_curves.tc`. Classification is read as saved,
  not recomputed. Its original metadata is retained in the decoder sidecar.
- Fit fresh tuning curves on the **first baseline** in `data/interval_table.csv`,
  intersected with valid tracking support, using 60 angle bins and the measured
  camera sampling rate. Unvisited angle bins are excluded from the decoder.
- Decode camera-covered periods with 100 ms spike-count bins and a uniform prior.
  Each camera acquisition segment is decoded separately in bounded-size chunks.
- Write `null` for unsynchronized frames, frames outside requested/full decoding
  bins, bins with no selected-unit spikes, and invalid posteriors. A trailing
  partial bin is not decoded. Missing tracked angles outside training do not
  prevent neural decoding when the frame timestamp is known.

Useful options:

| Option | Effect |
| --- | --- |
| `--probe A B` | Combine selected neurons from both probes into one decoded direction. |
| `--all-good` | Use every good unit without requiring saved HD classification. |
| `--units 12 35 81` | Use explicit good unit IDs from a single selected probe. |
| `--hd-class 1` | Use exactly class 1 instead of class 3. |
| `--train-interval 10 70` | Train on explicit ADC-relative seconds. |
| `--decode-interval 70 80` | Decode only this range; preserve other frames as null. |
| `--bin-size 0.05 --bins 90` | Change spike-count duration and angle resolution. |
| `--basler` | Accept sparse frames and the `hd` field; use channel 6, active low. |
| `--output /tmp/internal_head_direction.json` | Save outside the recording. |
| `--overwrite` | Replace existing decoder output and its metadata sidecar. |

The script reads existing `data/session_info.json`, camera timing (saved frame
times, sync data, or ADC pulses), `data/processed/head_direction.json`,
`data/probeA/adc_spike_time.npy`, and Kilosort cluster assignments/labels. Supply
`--kilosort-dir` if multiple Kilosort directories exist. Motive uses camera
channel 1, active high; `--camera-channel` overrides the channel. Spike and
camera times are aligned to the same ADC origin by the existing timing helper.

For a combined population, pass `--probe A B`. The saved HD class selection is
applied independently to each probe, and their spikes enter the same Bayesian
decoder. Identical cluster IDs on different probes remain separate neurons.
Each probe must have exactly one Kilosort directory; `--kilosort-dir` and
`--units` are single-probe options.

`internal_head_direction.metadata.json` records the selected units, source
paths, ADC origin, training/decoding intervals, parameters, and valid-frame
counts. Training and decoding overlap by default: this file is a fitted neural
population estimate, **not held-out decoding accuracy**. Use disjoint training
and decoding intervals when evaluating generalization; saved HD unit selection
may itself have used data from the evaluation interval.

For multiple probes, the sidecar records `probes`, `kilosort_dirs`, per-probe
`unit_selection`, and a `decoder_unit_map` connecting each unique decoder ID to
its original probe and unit ID. The output JSON still contains one
`internal_direction` time series. Single-probe metadata retains its existing
`probe`, `kilosort_dir`, and `unit_selection` fields and original unit IDs.

## Train across sessions and decode separate sessions

Use `decode_head_direction_across_sessions.py` for recordings with the same
jointly sorted probe and cluster identities. For example, train on all valid
tracked camera coverage in m19 sessions 9 and 10 and apply the saved population
decoder to sessions 5 and 7:

```sh
~/.virtualenvs/rfmapping/bin/python decode_head_direction_across_sessions.py \
  --train-session \
  '/mnt/senzailab/Kai/#Recording/m19/260827/260827_9' \
  '/mnt/senzailab/Kai/#Recording/m19/260827/260827_10' \
  --decode-session \
  '/mnt/senzailab/Kai/#Recording/m19/260827/260827_5' \
  '/mnt/senzailab/Kai/#Recording/m19/260827/260827_7' \
  --probe A --model-output /path/to/model.npz \
  --camera-clock '/mnt/senzailab/Kai/#Recording/m19/260827/260827_5=/path/to/260827_5_camera_clock.npz'
```

The selection is the **union of saved training-session `hd_class == 3` IDs**,
intersected with good Kilosort IDs in every training session. Classifications
are used as saved; their source metadata is retained because the training
sessions may have used different classifiers. Target HD classes are not read.
The same units enter both target populations, including units with zero target
spikes. This requires independently verified matching cluster identities;
matching numeric IDs from unrelated sorts is insufficient.

Pynapple computes each training session's spike counts and occupancy samples.
The script converts occupancy to seconds with that session's camera sampling
rate, sums counts and occupancy separately, and divides once to obtain pooled
rates. It excludes angle bins unvisited across all training sessions. Defaults
are 60 angle bins, 100 ms decoding bins, and a uniform prior. No interval table
is required, and training and target session paths must be disjoint.

The reusable NPZ contains unit IDs, angular bin edges and centers, pooled spike
counts, occupancy seconds, visited bins, and rate curves. Its metadata sidecar
records training sources, class provenance, intervals, ADC origins, and a SHA256
hash. Each target reloads this exact saved artifact and writes the usual
`data/processed/internal_head_direction.json` plus metadata. Target tracking is
read for frame IDs only; target angles are not used to fit or select anything.
Existing model and decoding outputs require `--overwrite` to replace them.

The optional repeated `--camera-clock SESSION=NPZ` is an explicit audited
target-clock override. The NPZ must contain `exposure_times_s` (ADC-relative
seconds), scalar `adc_time_origin_s`, and scalar JSON string `metadata_json`.
The JSON must include `allowed_trailing_frames` and should record the independent
timing evidence. The target's frame count must match the clock plus exactly that
audited trailing count; those unsynchronized frames stay null. Without an
override, only the standard zero/one trailing-frame case is accepted. The clock
artifact path, hash, and evidence are copied to the target metadata. In the m19
session 5 example, the override is required because 792 trailing camera frames
occur after the electrophysiology acquisition ends; the source tracking and
sync files are preserved.
