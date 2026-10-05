# Agent Instructions

## Scope and execution

- RF acquisition, MATLAB integration, tuning, and spatial analysis live here.
  GLMs belong in `../glm`; viewers belong in `../rfmapping_gui`.
- Execute project code only via `ssh hhw9l84`, from `~/Developer/rfmapping`,
  using `~/.virtualenvs/rfmapping/bin/python`. Do not execute project Python
  locally. Compare relevant files before remote validation: checkouts can differ.
- Remote recordings under `/mnt/senzailab/Kai/#Recording` and original MATLAB
  sources under `/mnt/ssd4.1/Matlab` are authoritative. Exclude `.m` files from
  Python dependency checks. Legacy `vs.py` is outside the standalone RF gate.

## Entrypoints and semantics

- Follow `README.md` for the current raw-to-RF workflow and code layout.
  Root notebooks are active; keep `locate_rf.py` at its MATLAB bridge path.
  Maintenance scripts live in `scripts/`; optional video entries remain at the
  root. Reusable Python code lives in `Utils/`.
- `matlab_auto.ipynb` previews periodic-stimulus timing; `matlab.ipynb` applies
  stored timing edits. `freemoving.ipynb` prepares camera frame times. These
  are distinct workflows, not interchangeable viewers.
- Separate loading, selection, statistics, and plotting as described in the
  README. GLM's copied `Utils/` can differ; reconcile semantics before sharing.
- Preserve raw counts, occupancy, zero-versus-missing bins, and native angles.
  A `.rfmap` filename alone does not identify its JSON/NPZ/HDF5 contract.
- For apparent pre-stimulus responses, inspect full `timeBinEdges` and raw trial
  timing: negative/late bins can overlap adjacent stimuli about 100 ms apart.
  Plot range filters the 2-D map only; timelines retain their full time axis.
- Figures, axes, and exports use opaque white backgrounds and dark labels;
  set Matplotlib figure/axes/savefig face colors to white and transparency off.

## Local workflows and evidence

- Inspect both the Git index and files on disk before cleanup. Optional video,
  decoder, and calibration workflows may be ignored but still actively used;
  removal from Git alone does not establish retirement.
- Preserve `.codex_tmp/` snapshots and existing generated evidence. Keep new
  scratch work out of active entrypoints; record durable findings once.
- Run relevant checks remotely, distinguishing core-package tests from optional
  local workflows and recording-dependent tests.
