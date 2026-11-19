# Poster materials

The active notebook is `../poster.ipynb`; current SVG/PNG figures are in
`../poster_output/current/`. The Downloads shortcut is
`~/Downloads/poster_materials/`. Earlier exports remain in `poster_output/`
and the revision snapshots, outside the current figure index.

Run the notebook on **hhw9l84**, from `~/Developer/rfmapping`, using
`~/.virtualenvs/rfmapping/bin/python`. It reads saved recording inputs and
exports editable SVGs and opaque white PNGs. It does not generate missing
recording stages. Previous `poster_assets/` or `poster_output/` files are not
inputs: the notebook can start with an empty output directory.

The first code cell lists experimental and display choices. Population counts,
RF proportions, EBB quality and EBB example ranking are computed from the
current recording inputs. `RFMAPPING_RECORDING_ROOT` can override the recording
root and `POSTER_OUTPUT_DIR` can override the export directory.

Change `preview_unit` in the first code cell to switch the sample within this
same-day cohort. ON/OFF rectangles, the ON RF polar view, the same-cell free-moving RF/EBB
row, PSTH, filenames and provenance follow that parameter. The independent EBB
example is configured by `ebb_example_spec`: the user selected M19 260831_2,
Probe A, unit 125. Its same-session HD class is 3; RF positivity is not required
for this illustration. The EBB map and measured HD × EBB figure use this same
cell. Its computed split-half map correlation is 0.298, so it is not evidence
of stable independent EBB coding. Rotation units are configured separately.

Both sample canvases are 18.3 by 6.25 inches. The standalone RF blocks are
14 by 6.25 inches, with stacked ON/OFF rectangles and one ON annulus beside
them. A rounded stepped frame joins the ON rectangle and annulus, following
the supplied reference. Rectangular maps are 8 inches wide at native placement;
the HD polar is 3 inches and the colored ON annulus is 3.8 inches in diameter.
Each RF row has its own horizontal Hz scale below its rectangle. For unit 269, display-only highlight
exclusions remove row 5, columns 8-10 (one-based) from both sample outlines:
10 displayed bins versus 13 saved detector bins. Source masks, rate maps,
population statistics and response timing retain their original values.

The second left-column example is M22 unit 289 from the same S2/S9 pairing:
two saved ON RFs, with S9 HD tuning, ON/OFF rectangular firing-rate maps and
one ON annular view. It uses the same layout as unit 269. Its OFF map has
no saved excitatory RF and receives no detection outline. Both rectangular
maps retain the same 8-inch width as unit 269.

## Current source choices

Single-day population panels use **M22, 2026-10-08, Probe A**; the independent
EBB/HD example uses **M19, 2026-08-31, Probe A**:

| Panel | Source |
| --- | --- |
| Regular ON/OFF sample and RF identification | S2, unit 269 |
| HD paired with regular RF | S9, open arena |
| Rotation examples | Units 287 and 305; S2 head-fixed, S7 screen map, S7 animal map |
| Middle rotation file | `rotation_30_261008_7+.rfmap` |
| Animal-coordinate rotation file | `rotation_30_261008_7.rfmap` |
| Freely moving visual RF | S10, unit 269, full-height bars |
| Independent EBB sample | M19 260831 S2, unit 125, user-selected |
| Same-cell RF/EBB row | Unit configured by `preview_unit`; S10 RF / S11 EBB |
| HD paired with EBB | S11, the same wall session |
| HD × EBB bearing joint firing rate | M19 S2 unit 125; saved frame HD and boundary-ray geometry |
| HD × EBB marginal product | Same M19 cell; separate normalized HD and EBB curves, dimensionless |
| M22 circular RF-region arrows | Class-3 S9 HD peak and each saved S2 ON RF region center |

RF and rotation maps are actual firing rates in 0–200 ms, normalized by the
number of contributing presentations and time-window duration. Each map has
its own explicit Hz scale, from zero to the map's maximum. Both rotation files
use their own saved `stimulusPresentationCounts`. For the legacy free-moving
source, each run reconstructs presentation counts from the original stimulus
MAT, onsets, corrected XY, calibrated HD offsets and calibration. The native
pixel geometry follows `RFmapping_core_fm.m`; reconstructed occupancy must
match the saved RF file. `free_moving_exposure.npz` is an output, never an input.

EBB quality is computed from `restored_maps.npz`, `position_correction.npz`,
`frame_spike_counts.npz` and `audit.json` in the saved S11 spatial stage. The
notebook checks frame identity, reconstructs raw spike maps and occupancy, and
verifies the saved smoothing. It computes full, temporal-half and unheld maps;
unvisited bins remain missing in the statistics. `m22_261008_wall_quality.json`,
`ebb_example_candidates.csv` and `ebb_quality_maps.npz` are newly written in the
current output directory on every run.

The joint figure conditions the selected sample's raw frame spike counts and
occupancy on HD and egocentric boundary bearing, separately for the existing
0–8, 8–16 and >16 cm distance-bin bands. HD bins are 12°, bearing bins retain
the native 6° spacing. Counts and occupancy are smoothed separately with a
12° Gaussian on both circular axes before division. Originally unvisited bins
remain blank. Summing over HD must reproduce the saved raw EBB count and
occupancy maps. The NPZ retains those unsmoothed 3D maps and each band's raw
counts, exposure and rates; the CSV and JSON record the plotted values and
parameters. Smoothing and HD bin count are explicit notebook settings.
The separately labeled marginal-product figure has the reference image's
separable curve construction; it is not a measured or GLM-predicted joint rate.

M22's circular plot uses the same construction as the slides: one arrow for
each saved excitatory ON RF region, positioned at its cell's S9 HD peak and
pointing at HD peak + RF azimuth. The HD peak uses 180 bins and 12° smoothing.
This is the numeric angular-sum schematic, not a calibrated world-coordinate
transform. The per-region table is exported as `m22_poster_rf_region_arrows.csv`.
These definitions, the two sessions, HD bin count/smoothing, fixed arrow length,
and clockwise zero-at-north display are printed directly on figure 21.

The comparison cohort is computed from saved excitatory ON RF in S2 and
class-3 HD in S9. All selected cells are retained in the wall comparison; this does not
assert that every cell also meets the S11 HD classification threshold.
S9 HD orders the RF/HD figure. **S11 HD independently orders the EBB figure.**
The full four-panel S11 HD/EBB figure is in the library. The poster uses
`16_poster_hd_rf_ebb_shared_scale`: the three EBB bands sit below the HD/RF pair,
and all five heatmaps share one 0/max colorbar. The two rows retain their
separate S9 and S11 orders. Angle axes have five ticks; HD labels are
180, 270, 0, 90, 180, while RF and EBB labels run from -180 through 0 to 180.
Full unit IDs remain in CSV provenance, while poster axes show cell rank.

The standalone `14_poster_hd_rf_sorted` has two tall heatmaps. Set
`hd_rf_panel_aspect` in the first notebook cell to change their physical
height/width ratio (currently 1.6); this is a display setting only.

## Figure index

`poster_output/current/figure_index.csv` lists current exports.
`poster_figure_sources.json` records source paths, SHA-256 hashes, selections,
rate checks, row orders, coordinates and statistics. `poster_contact_sheet.png`
previews the six requested figure families.

- `01` / `02`: polar RF / grayscale RF firing rate, ON and OFF for units 269 and 289.
- `03` / `04`: EBB polar without / with exact 8 and 16 cm guides.
- `06`: full same-session HD/EBB comparison.
- `07`–`17`: poster panels, including the bar-RF / EBB row, shared-scale
  HD/RF/EBB comparison, and the additional multiple-ON-RF sample.
- `19`: the selected EBB cell's HD × boundary-bearing joint firing-rate maps.
- `20`: standalone reference-layout RF blocks for units 269 and 289.
- `21`: M22 circular plot, one arrow per saved ON RF region.

The left sample includes ON/OFF rectangles and one ON annulus. Unit 269 has no saved
excitatory OFF RF; the OFF rate map is still shown, without a false detection
outline. RF identification uses the contiguous saved region perimeter in both
rectangular and polar views, with no internal outlines between detected bins.
The unit-269 sample has the three user-requested highlight exclusions above.
All firing-rate RF maps use a horizontal Hz colorbar below their lower-right
corner, including rotation maps. Each map retains its own 0-to-maximum scale.
EBB colorbars remain vertical. The free-moving RF is exactly 8 inches wide
and 8 × 84/360 inches high, matching the head-fixed RF display at 100% placement.

## Coordinates and statistics

Elevation is relative to the middle stimulus-row center:
`elevation = native_middle_row_y - native_y`. The seven-row M22 maps have
centers +36, +24, +12, 0, -12, -24, -36 degrees and outer edges +/-42 degrees.
The S10 bar map has one measured elevation row; it is displayed as full-height
stripes without inventing elevation samples or labeling a vertical RF extent.

M14's rows centered at +/-40 degrees are removed before smoothing and pooling.
The remaining rows keep their native 10-degree spacing and +/-35-degree edges.
All seven-mouse poster panels now apply current **HD class 3**, with **no ADN
restriction**, as confirmed on 2026-11-10. M14 retains 10 HD + ON RF cells;
the seven-mouse map contains **129 HD + ON RF cells** among 254 RF-assessed HD
cells. The previous 168-cell map included 39 cells outside this HD cohort.
Coverage marginals count:
distinct units with a saved RF anywhere in each azimuth column or elevation row,
counting each unit once per column or row from unsmoothed masks.

RF profiles sum raw counts across every native elevation row in 0–200 ms and use the
existing peak-normalized profile preparation. HD and EBB curves are likewise
peak-normalized for comparison; colorbars say normalized response with endpoints
`0` and `max` (each cell's peak).

The ON RF proportion follows the user's convention: missing RF records count
as zero. Counts among 404 HD cells are **275 zero, 100 one, 29 two or more**.
The pie uses **2+ (dark blue), 1 (light blue), 0 (gray)**, with counts and
percentages alongside it. All values are calculated from the current inputs.
The CSV retains whether a saved RF record existed. OFF and inhibitory RFs are
excluded. Thus 129/404 (31.9%) have at least one ON RF; multiple RFs are
29/404 (7.2%) of all HD cells, or 29/129 (22.5%) of RF-positive HD cells.
The 275 zero-RF category contains 125 assessed zeros and 150 unassessed cells,
merged only for this explicitly requested ratio convention.

Response timing uses **2,253 RF unit-positions from the same 129 HD cells**.
The population curve pools all spatial bins inside the saved RF masks, with
equal weight per bin, and shows mean +/- descriptive SEM across those bins.
Bins within a cell are correlated and are not independent biological replicates.
The sample curve retains SEM across its RF positions. Per-cell means and an
equal-weight, seven-animal summary based on those cell means are exported
separately. The population CSV records both `n_rf_positions` and `n_cells`.
The complete time axis is retained; these curves are not threshold-defined
onset-latency estimates.

`seven_mice_population_units.csv` retains raw-map presence, RF QC, saved
assessment status and the membership used by every panel. The fraction CSV
also records `n_on_rf_for_ratio` and `ratio_category` after missing-as-zero
mapping. The per-recording CSV states both denominators explicitly.
`seven_mice_population_audit.json` records the source-level recomputation:
the original raw-map QC, GUI smoothing, RF masks and RF centers reproduce
exactly for every recording/probe. No recording inputs are overwritten.
Run `python -m scripts.verify_poster_population` remotely after regeneration
for the separate raw-count, membership, coverage and direction verification.

The current M22 S11 EBB uses saved 12-degree angles and 20 distance bins, from 0 to
28.284 cm. Distance bands select centers at <=8, 8–16, and >16 cm; their actual
supports are 0–8.485, 8.485–15.556, and 15.556–28.284 cm. Dotted circles are
exactly 8/16 cm and do not rebin data. The polar and rectangular EBB maps display
the saved smoothed rates. Saved smoothing is checked against Gaussian-smoothed
spike counts divided by smoothed occupancy. The analysis metadata specifies
sigma = (2.5 angular bins, 5 distance bins), retaining 30° angular smoothing.
Archived 6° inputs explicitly use the historical (5, 5) parameters and must
reproduce the saved map before being accepted. Raw occupancy/counts
remain intact; profile statistics still use only the finite measured bins.

Current coverage, held-frame fraction and split-half stability are recomputed
in `poster_output/current/m22_261008_wall_quality.json`. Historical values from
the earlier 6° stage do not describe the regenerated input. These are descriptive
checks, not EBB significance tests.

The current direction panel is `18_poster_seven_mice_peak_sums.svg`, regenerated
from the same 129 HD + ON RF cells: M14 10, M15 38, M19 13, M21 12, M22 18,
M23 31 and M24 7. It retains one all-height RF peak plus one HD peak per cell,
with 15 histogram bins; it is not an RF-region histogram. The archived 130-cell
reference had M23's ADN restriction and an obsolete 20-cell M24 cohort.
`seven_mice_reviewed_results.pdf` contains the four current population panels;
`seven_mice_reviewed_overview.png` previews them together.

## Native poster

The current, manually edited poster is `~/Downloads/kai_poster.ai`, exported
as `~/Downloads/kai_poster.pdf`. The 2026-11-11 PSTH refresh replaces the
population curve and band with pooled RF-bin mean/SEM and updates their labels,
preserving the sample curve and current poster layout.

`layout/build_poster.jsx` places the vector panels in the scanned three-column
layout, preserving the original title, authors and branding from
`reference/kai_poster_source.ai`. Its earlier design outputs are
`~/Downloads/kai_poster_design.ai` and `~/Downloads/kai_poster_design.pdf`.
The PDF preview is `~/Downloads/kai_poster_design.png`.

For the current eight-inch RF layout, run the notebook, then use
`layout/update_poster_20261110.jsx` in Illustrator on the Mac. It updates the
affected regions and archives the previous objects on a hidden layer. The
live header is retained; coverage, proportions, timing and direction panels
use the regenerated population outputs.
The Introduction and Results text, references and supporting measurements are
stored in `layout/poster_copy.json`. Session/source details stay in provenance.

`layout/update_rf_sample_layout.jsx` is the earlier update for the first RF
sample alone; its placement predates the second sample.

`layout/update_rf_samples_layout.jsx` updates only the two RF examples and
repositions the rotation/timing panels below them. Both examples use the same
8-inch rectangles and 3-inch HD/polar maps, with compact vertical spacing.

The earlier `layout/update_sample.jsx` expects intact named groups for all three
sample panels. Use `layout/update_poster_20261110.jsx` for the current poster.

`layout/update_plot_style.jsx` expects the earlier eight named figure groups:
RF examples, rotation, coverage, fractions, free movement, and the three
comparison panels. It retains existing group centers and fitting bounds.
