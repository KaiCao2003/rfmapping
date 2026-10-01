# Paper comparison and public-code search

Checked on 2026-09-13. The original notebook was an implementation check with
an independently designed diagnostic figure. It was not a reproduction of
the paper's full analysis or figure layout.

## What was different in the plots?

| Panel | Paper | This notebook |
|---|---|---|
| [Fig. 1h](https://www.nature.com/articles/s41586-023-05813-2/figures/1) | Baseline FNN projection with polar grid and cyclic angle colors | Originally only held-out z1/z2 on Cartesian axes with twilight colors. Now adds polar views of training, all rows, and test, plus a radius-colored view. |
| [Fig. 2c–d](https://www.nature.com/articles/s41586-023-05813-2/figures/2) | Baseline versus full cue-manipulation session; radius-colored states and reconstructed activity bumps | Our training segment is an in-sample RF reference, not a verified stable-cue baseline. No cue-manipulation epoch or activity-bump reconstruction is claimed. |
| [Fig. 1i–j](https://www.nature.com/articles/s41586-023-05813-2/figures/1) | ZIG Bayesian log likelihood, decoded HD, and error distribution | FNN angles and an empirical error CDF. The 19.16° FNN error is not directly comparable to the paper's decoder result. |
| [Fig. 2e](https://www.nature.com/articles/s41586-023-05813-2/figures/2) | Radius versus a separately estimated network gain | Our scatter uses raw mean firing rate. It does not estimate the paper's gain. |

The new `paper_projection.png` uses actual saved z coordinates, one common
radial scale, and measured-HD hue. There is no pointwise radial normalization,
smoothing, or exclusion of points to create a narrower ring. A Cartesian-to-polar
round trip checks that the displayed coordinates reconstruct z1/z2. The trained
model, input normalization, time bins, splits, and predictions are retained.
The reference ring has radius CV 0.183, versus 0.226 in the held-out interval:
its width is present in the learned coordinates, not just the plotting axes.

## Method differences that remain

The [Methods](https://www.nature.com/articles/s41586-023-05813-2) use ADN calcium
data, deconvolved activity with a three-frame moving average (about 100 ms),
and baseline training. Our sample uses 100-ms electrophysiology counts,
training-selected good units, and a chronological RF-session split. We have
not established that this unit population is ADN. The exact input normalization,
optimizer, initialization, and stopping choices in the authors' FNN code are
unavailable; the notebook's choices are documented adaptations. The network
diagram and activation choices were checked against
[Extended Data Fig. 4d](https://www.nature.com/articles/s41586-023-05813-2/figures/9).

Another distinction is the radius definition. The Methods derive an approximate
radius `R_hat = 1/g`, while the geometric radius of the z-plane is `norm(z)`.
These agree only when `g * norm(z)` is approximately one. In our test interval,
the median `abs(g * norm(z) - 1)` is 0.139 (13.9%). Both quantities are saved;
the new figure explicitly plots `norm(z)`. Without the original plotting script,
the exact radius choice used to render the authors' panels remains unverified.
[Source: Analysis of low-dimensional representation](https://www.nature.com/articles/s41586-023-05813-2).

The paper's gain calculation is:

```text
alpha(t) = sum_i r_i(t) * f_i(theta_decoded(t))
           / sum_i f_i(theta_decoded(t))^2
```

Here f_i is a baseline tuning curve. The gain is then smoothed and normalized
to its baseline mean. This differs from both the FNN middle branch g and a
raw population mean. Consequently, the earlier statement that our r = 0.124
does not support the paper's gain interpretation was too strong: **that
diagnostic does not test the paper's estimator or experimental claim**.
[Source: Calculation of network gain](https://www.nature.com/articles/s41586-023-05813-2).

## Public code: what was actually found?

**No public implementation of this paper's three-branch polar FNN was located
in the sources checked. This is a search result, not proof that no copy exists.**
The paper's [Code availability statement](https://www.nature.com/articles/s41586-023-05813-2)
says source code is available on request to the corresponding authors.

| Checked source | Finding |
|---|---|
| [Brandon Lab publications](https://www.markbrandonlab.com/publications), [navigation](https://www.markbrandonlab.com/navigation), and [modeling](https://www.markbrandonlab.com/techniques/computational-modeling) | Found the paper, its PDF, and related research links; no FNN repository link. |
| [First author ZakiAjabi](https://github.com/ZakiAjabi) / [public repository API](https://api.github.com/users/ZakiAjabi/repos?per_page=100&type=owner) | One public owned repository: a fork of `math-as-code`. The author identity is also supported by contributions to the lab's ConnectomeInfluenceCalculator. |
| [DrugowitschLab repositories](https://api.github.com/orgs/DrugowitschLab/repos?per_page=100) | Inspected 26 public repository names and descriptions. Ring-attractor projects identify different papers; no matching Ajabi polar-FNN project was identified. |
| [Keinath Lab](https://www.wedobrainstuff.com/) | The Nature paper is linked, without a matching code link. |
| [Wei lab repositories](https://api.github.com/orgs/wei-bbc-lab/repos?per_page=100) | Four public repositories. `Split-Trial-Analysis` reuses this dataset for a later study; it is not the 2023 FNN implementation. |
| [Figshare dataset](https://doi.org/10.6084/m9.figshare.21792689) / [metadata API](https://api.figshare.com/v2/articles/21792689) | One 5,447,444,135-byte `Data.zip`. Its ZIP directory contains `Data/Cue_rotation.mat`, `Data/Cue_shift_2min.mat`, and `Data/Cue_shift_20sec.mat`; no source-code files. Only 451 bytes of archive metadata were fetched to inspect its directory, not the multi-GB dataset. |
| Web, GitHub repository/readme, Zenodo, and ModelDB searches | Searched the DOI, title, author name, and polar/FNN/head-direction terms. No matching original FNN code was identified. Repository/readme search is not an exhaustive search of every code file. |

Relevant public code **does** exist:

- [zhd96/zig](https://github.com/zhd96/zig): the ZIG encoding/decoding method
  cited as Wei et al. by the Nature paper. Includes
  [`code/decoding.py`](https://github.com/zhd96/zig/blob/master/code/decoding.py),
  [`code/tutorial.ipynb`](https://github.com/zhd96/zig/blob/master/code/tutorial.ipynb),
  and [`examples in paper/head_direction.ipynb`](https://github.com/zhd96/zig/blob/master/examples%20in%20paper/head_direction.ipynb).
  It targets deconvolved calcium signals and documents TensorFlow 1.9;
  it is a different component from the three-branch polar FNN.
- [wei-bbc-lab/Split-Trial-Analysis](https://github.com/wei-bbc-lab/Split-Trial-Analysis):
  later analysis scripts whose README explicitly lists the Ajabi Nature 2023
  dataset and the ZIG decoder as dependencies.

The original FNN training/plotting scripts and normalization details remain
the missing pieces for matching the authors' implementation. The paper lists
Zaki Ajabi and Mark Brandon as corresponding authors; no contact message
has been sent.
