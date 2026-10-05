# Circular statistics contracts and package evaluation

`Utils.statistic_utils` owns ordinary Rayleigh, the existing JS coefficient,
Fisher–Lee correlation, and pairing-permutation statistics. Model fitting calls
`fisher_lee_statistics`; it does not implement its own correlation or shuffle
loop. `Utils.hd_rf_prediction.fisher_lee_correlation` remains an import alias for
existing callers.

## Public contracts

- `rayleigh_uniformity`: unweighted angles in degrees; omit nonfinite values;
  return `n`, resultant length `r`, `z`, and analytic `p`. Fewer than three
  observations have no p-value.
- `circular_correlation`: existing JS coefficient from sine-centered radians
  using SciPy's circular mean. A balanced sample has no unique mean direction;
  its computed coefficient depends on floating-point cancellation. Preserve this
  existing convention rather than silently changing previously reported results.
- `fisher_lee_correlation`: at least three finite aligned pairs in degrees;
  invalid pairs raise, while constant or antipodal samples return `None`.
  Pairwise sine differences do not require a mean direction.
- `fisher_lee_statistics`: the same coefficient plus optional two-sided
  unit-pairing permutations, comparing absolute rho with the existing numerical
  tie tolerance and plus-one correction. `test=False` draws no permutations.
  The caller decides whether its cohort permits inference. The HD/RF fit disables
  inference for a cohort selected on its HD+RF outcome.

The occupancy-adjusted Rayleigh test in `Utils.tuning_curve_utils` has a different
sampling model and remains separate.

## pycircstat2 evaluation, 2026-10-02

Evaluated **pycircstat2 0.1.15** in an isolated remote scratch directory without
installing it into the analysis environment. The official
[correlation API](https://circstat.github.io/pycircstat2/reference/correlation/)
and [hypothesis API](https://circstat.github.io/pycircstat2/reference/hypothesis/)
document the coefficient definitions. Online development documentation can differ
from the evaluated release, so the recorded version identifies this comparison.

Five stored real-recording cohorts contained 13, 25, 25, 16, and 14 paired units
(93 total). Every `(mouse, date, probe, unit_id)` key was unique within its cohort.
On these saved inputs, ordinary coefficients agreed to floating-point precision.
These are saved pairing artifacts; raw data and current classifications were not
recomputed for this evaluation.

For 100 random pairs of 97 angles, maximum absolute coefficient errors were
`2.50e-16` (JS) and `3.47e-17` (Fisher–Lee). Rayleigh analytic p-values were exactly
equal when supplied the same `r` and `n`.

A direct dependency replacement would nevertheless change behavior:

- Package JS returns NaN for uniform or nearly uniform mean directions where the
  current SciPy convention returns a finite, numerically unstable value.
- Package Fisher–Lee raises for some small but defined angular variations due to
  its absolute near-zero denominator tolerance. The current implementation retains
  these samples and returns `None` for numerically constant/antipodal samples.
- Package raw-angle Rayleigh does not implement this project's omission of
  nonfinite observations or minimum-three-observation p-value policy.
- Package inference uses a jackknife for Fisher–Lee and an asymptotic test for JS;
  these do not replace the existing unit-pairing permutation p-values.
- On a 97-unit, 1,000-permutation synthetic benchmark, repeated package Fisher–Lee
  calls took 0.156 s versus 0.022 s for the cached implementation. Permutation
  p-values agreed. This single benchmark measures callback overhead, not overall
  pipeline performance.

**Decision:** retain the current numerical implementations and existing SciPy
dependency. Adding pycircstat2 would require compatibility logic and would not
remove the cached permutation implementation; ordinary Rayleigh's remaining
analytic expression alone does not justify another dependency.

The refactor preserved every returned field exactly for 88 synthetic/edge-case
fits and the five saved recording cohorts. The 23 focused direction-statistics
and circular-regression tests passed remotely against isolated local-source
snapshots; full-module integration validation is a separate gate.

Evaluation scripts and machine-readable evidence are preserved under
`.codex_tmp/circular-package-audit-20261002/` locally and on `hhw9l84`:
`audit.py`, `results.json`, `validate_refactor.py`, and `refactor_results.json`.
