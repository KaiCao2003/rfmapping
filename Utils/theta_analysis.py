"""Pure spike rhythmicity and LFP phase calculations.

Inputs share one seconds clock. No function loads, generates, or saves files.
Epochs are half-open, and pairs or interpolation never cross epoch/LFP gaps.
The paper does not specify every numerical spectral detail; the returned
parameters make this implementation's discretization explicit.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import false_discovery_control


_THETA_SUMMARY_DTYPES = {
    "unit_id": "object", "n_moving_spikes": "int64", "paper_eligible": "bool",
    "theta_index": "float64", "peak_frequency_hz": "float64",
    "is_theta_rhythmic": "boolean", "mean_vector_length": "float64",
    "preferred_phase_rad": "float64", "rayleigh_p": "float64", "rayleigh_q_bh": "float64",
    "is_phase_locked": "boolean", "precession_candidate": "boolean",
    "phase_precession_index": "float64", "peak_frequency_per_cycle": "float64",
    "n_phase_spikes": "int64", "rhythmicity_status": "object",
    "phase_locking_status": "object", "precession_status": "object", "theta_status": "object",
}


def analyze_theta_units(
    spikes_by_unit, theta, *, tracking_intervals_s, moving_intervals_s,
    min_spikes_exclusive=100, spatial_coverage=1.,
    min_spatial_coverage_exclusive=.80,
):
    """Analyze already movement-selected spikes, independently of CB/CD tuning.

    Tracking intervals bound time ACG pairs, while movement intervals condition
    phase occupancy. The optional coverage gate retains the original paper
    eligibility rule without needing center-cell results. Below-coverage units
    retain descriptive measurements and nullable classification labels.
    """
    coverage_pass = spatial_coverage > min_spatial_coverage_exclusive
    unit_results, rows = {}, []
    for unit, moving in spikes_by_unit.items():
        row = dict(
            unit_id=unit, n_moving_spikes=len(moving),
            paper_eligible=len(moving) > min_spikes_exclusive and coverage_pass,
            theta_index=np.nan, peak_frequency_hz=np.nan,
            is_theta_rhythmic=pd.NA,
            mean_vector_length=np.nan, preferred_phase_rad=np.nan,
            rayleigh_p=np.nan, rayleigh_q_bh=np.nan,
            is_phase_locked=pd.NA, precession_candidate=pd.NA,
            phase_precession_index=np.nan, peak_frequency_per_cycle=np.nan,
            n_phase_spikes=0,
            rhythmicity_status="not_assessed", phase_locking_status="not_assessed",
            precession_status="not_assessed", theta_status="insufficient_spikes",
        )
        rows.append(row)
        if len(moving) <= min_spikes_exclusive:
            continue

        rhythm = theta_rhythmicity(moving, tracking_intervals_s)
        lock = phase_locking(moving, theta, exposure_intervals_s=moving_intervals_s)
        precession = phase_precession(moving, theta)
        unit_results[unit] = dict(
            rhythmicity=rhythm, phase_locking=lock, phase_precession=precession,
        )
        row.update(
            theta_index=rhythm["theta_index"], peak_frequency_hz=rhythm["peak_frequency_hz"],
            mean_vector_length=lock["mean_vector_length"],
            preferred_phase_rad=lock["preferred_phase_rad"], rayleigh_p=lock["rayleigh_p"],
            phase_precession_index=precession["phase_precession_index"],
            peak_frequency_per_cycle=precession["peak_frequency_per_cycle"],
            n_phase_spikes=lock["n_spikes"],
            rhythmicity_status="computed" if np.isfinite(rhythm["theta_index"]) else "no_acg_pairs",
            phase_locking_status="computed" if lock["n_spikes"] > min_spikes_exclusive else "insufficient_phase_spikes",
            precession_status="computed" if precession["eligible"] else "insufficient_phase_acg_support",
            theta_status="computed" if coverage_pass else "descriptive_only_insufficient_spatial_coverage",
        )
        if row["paper_eligible"] and np.isfinite(rhythm["theta_index"]):
            row["is_theta_rhythmic"] = rhythm["is_theta_rhythmic"]
        if row["paper_eligible"] and lock["n_spikes"] > min_spikes_exclusive:
            row["is_phase_locked"] = lock["is_phase_locked"]
            if precession["eligible"]:
                row["precession_candidate"] = precession["precession_candidate"]

    # Keep a complete, typed table when no good units were supplied. Preserve
    # native unit-ID types for populated tables instead of coercing IDs to text.
    summary = pd.DataFrame(rows, columns=list(_THETA_SUMMARY_DTYPES)).astype({
        name: dtype for name, dtype in _THETA_SUMMARY_DTYPES.items() if name != "unit_id"
    }).set_index("unit_id", drop=False)
    assessed = summary["is_phase_locked"].notna() & summary["rayleigh_p"].notna()
    if assessed.any():
        summary.loc[assessed, "rayleigh_q_bh"] = false_discovery_control(
            summary.loc[assessed, "rayleigh_p"].to_numpy(), method="bh",
        )
    return dict(
        summary=summary, units=unit_results,
        metadata=dict(
            filter_parameters=theta["parameters"], phase_qc=theta["phase_qc"],
            movement_support="explicit movement-selected spikes and exposure intervals",
            precession_support="moving spikes paired within continuous LFP runs",
            acg_support="movement-filtered spikes paired across speed dips, within continuous tracked support",
            classification_scope=(
                f"NA if <={min_spikes_exclusive} movement spikes or spatial coverage "
                f"<={min_spatial_coverage_exclusive:.0%}"
            ),
            min_spikes_exclusive=min_spikes_exclusive,
            spatial_coverage=float(spatial_coverage),
            min_spatial_coverage_exclusive=min_spatial_coverage_exclusive,
            phase_alpha=.01, phase_multiple_testing="Benjamini-Hochberg across assessed units",
            exact_published_numerical_replication=False,
        ),
    )


def _spikes(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not np.all(np.isfinite(values))
            or np.any(np.diff(values) < 0)):
        raise ValueError("Spike timestamps must be a finite, sorted 1-D array.")
    return values


def _intervals(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.reshape(0, 2)
    if (values.ndim != 2 or values.shape[1] != 2
            or not np.all(np.isfinite(values))
            or np.any(values[:, 1] <= values[:, 0])
            or np.any(values[1:, 0] < values[:-1, 1])):
        raise ValueError("Intervals must be finite, sorted, nonoverlapping [start, end) pairs.")
    return values


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return float(value)


def _pair_acg(segments, bin_width, max_lag):
    """Count distinct event pairs; nearest-centered bins retain true coincidences."""
    bin_width = _positive(bin_width, "bin width")
    max_lag = _positive(max_lag, "maximum lag")
    radius = int(round(max_lag / bin_width))
    if radius < 1 or not np.isclose(radius * bin_width, max_lag):
        raise ValueError("Maximum lag must be an integer multiple of bin width.")
    positive = np.zeros(radius + 1, dtype=np.int64)
    n_events = 0
    for events in segments:
        events = np.asarray(events, dtype=float)
        n_events += len(events)
        # Vectorize pairs separated by each event ordinal. The largest shift
        # is the local event count within max_lag, not the whole recording.
        for shift in range(1, len(events)):
            difference = events[shift:] - events[:-shift]
            selected = difference <= max_lag
            if not np.any(selected):
                break
            bins = np.floor(difference[selected] / bin_width + .5).astype(np.int64)
            positive += np.bincount(bins, minlength=radius + 1)[:radius + 1]
    # A pair of different coincident spikes contributes both directions. There
    # is no automatic n_events delta at zero: event i is never paired to itself.
    counts = np.r_[positive[:0:-1], 2 * positive[0], positive[1:]]
    lags = np.arange(-radius, radius + 1, dtype=float) * bin_width
    return lags, counts, n_events


def _acg_spectrum(counts, bin_width, *, spectrum_convention, detrend, n_fft=None):
    if spectrum_convention not in {"magnitude", "power"}:
        raise ValueError("spectrum_convention must be 'magnitude' or 'power'.")
    counts = np.asarray(counts, dtype=float)
    values = counts - counts.mean() if detrend else counts
    if n_fft is None:
        n_fft = len(values)
    if not isinstance(n_fft, (int, np.integer)) or n_fft < len(values):
        raise ValueError("n_fft must be an integer at least the length of the ACG.")
    # A time shift changes phase only; magnitude avoids assigning negative
    # finite-window Fourier coefficients to 'negative power'. No smoothing,
    # lag-edge correction, taper, or implicit baseline subtraction is applied.
    spectrum = np.abs(np.fft.rfft(values, n=n_fft))
    if spectrum_convention == "power":
        spectrum = spectrum ** 2
    return np.fft.rfftfreq(n_fft, bin_width), spectrum


def _spectral_index(frequency, spectrum, peak_band, half_width, reference_bands):
    low, high = map(float, peak_band)
    if not 0 <= low < high <= frequency[-1]:
        raise ValueError("Peak band must be increasing and within the represented spectrum.")
    half_width = _positive(half_width, "peak half width")
    peak_selection = (frequency >= low) & (frequency <= high)
    if not peak_selection.any():
        raise ValueError("Peak band contains no Fourier frequencies.")
    reference = np.zeros(len(frequency), dtype=bool)
    for lower, upper in reference_bands:
        if not 0 <= lower < upper <= frequency[-1]:
            raise ValueError("Reference bands must lie within the represented spectrum.")
        reference |= (frequency >= lower) & (frequency <= upper)
    if not reference.any():
        raise ValueError("Reference bands contain no Fourier frequencies.")
    peak_indices = np.flatnonzero(peak_selection)
    peak_index = peak_indices[np.argmax(spectrum[peak_selection])]
    peak = float(frequency[peak_index])
    around_peak = np.abs(frequency - peak) <= half_width
    denominator = float(np.mean(spectrum[reference]))
    numerator = float(np.mean(spectrum[around_peak]))
    ratio = numerator / denominator if denominator > 0 else np.nan
    return dict(peak_frequency=peak if denominator > 0 else np.nan,
                index=float(ratio), peak_mean_spectrum=numerator,
                reference_mean_spectrum=denominator)


def theta_rhythmicity(
    spike_times_s, intervals_s, *, bin_s=.002, max_lag_s=2.,
    theta_band_hz=(4., 11.), peak_half_width_hz=1.,
    reference_band_hz=(0., 125.), threshold=5.,
    spectrum_convention="magnitude", detrend=False,
):
    """Calculate the paper's theta spectral ratio from within-epoch spike ACGs.

    Supply speed > 2.5 cm/s spikes and continuous acquisition/tracking epochs
    upstream. Separate speed bouts would additionally discard pairs across a
    brief pause; the notebook retains these pairs within valid tracking. This
    function never infers movement from spikes. The default is abs(FFT of
    ACG), not its square: an ACG transform estimates the spike spectrum. The
    optional ``power`` convention instead squares the ACG transform. This
    choice and mean subtraction are recorded and must not be silently mixed.
    The published threshold is >= 5; sparse/empty inputs remain unclassified.
    """
    spikes, intervals = _spikes(spike_times_s), _intervals(intervals_s)
    threshold = _positive(threshold, "threshold")
    segments = [spikes[np.searchsorted(spikes, start):np.searchsorted(spikes, end)]
                for start, end in intervals]
    lags, counts, n_spikes = _pair_acg(segments, bin_s, max_lag_s)
    frequency, spectrum = _acg_spectrum(
        counts, bin_s, spectrum_convention=spectrum_convention, detrend=detrend,
    )
    result = _spectral_index(frequency, spectrum, theta_band_hz,
                             peak_half_width_hz, [reference_band_hz])
    index = result.pop("index")
    peak = result.pop("peak_frequency")
    return dict(
        theta_index=index, peak_frequency_hz=peak,
        is_theta_rhythmic=bool(np.isfinite(index) and index >= threshold),
        n_spikes=n_spikes, duration_s=float(np.diff(intervals, axis=1).sum()),
        acg_lags_s=lags, acg_counts=counts, frequency_hz=frequency,
        spectrum=spectrum, **result,
        parameters=dict(bin_s=bin_s, max_lag_s=max_lag_s,
                        theta_band_hz=list(theta_band_hz),
                        peak_half_width_hz=peak_half_width_hz,
                        reference_band_hz=list(reference_band_hz), threshold=threshold,
                        spectrum_convention=spectrum_convention, detrend=bool(detrend),
                        self_pairs=False, lag_bin_assignment="nearest_center",
                        interval_semantics="half_open", cross_epoch_pairs=False),
    )


def extract_theta_phase(
    times_s, voltage, *, sample_rate_hz=None,
    passband_hz=(5., 10.), stopband_hz=(4., 11.),
    edge_exclusion_s=1., gap_factor=1.5, sampling_tolerance=.02,
    passband_ripple_db=1., stopband_attenuation_db=40.,
    amplitude_floor_multiplier=64.,
):
    """Zero-phase elliptic bandpass then Hilbert phase on each valid LFP run.

    The paper gives pass/stop edges, but does not fully specify filter design.
    Here an elliptic SOS filter is designed with explicit single-pass ripple/
    attenuation, then applied forward/backward (doubling the dB response).
    NaN voltage or timestamp gaps split runs. A configurable edge is excluded
    from both ends of every run; no sample or phase is interpolated over gaps.
    The input must already be appropriately antialias-filtered if downsampled.
    Hilbert phase 0 is the positive filtered-voltage peak; trough is +/- pi.
    Constant runs and Hilbert amplitude at the floating-point numerical floor
    have undefined phase and are excluded. This floor scales with the input
    signal; it is not a physiological theta-power criterion. No high-amplitude
    artifact threshold or automatic theta-bout selection is applied.
    """
    times = np.asarray(times_s, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    if (times.ndim != 1 or len(times) < 3 or voltage.shape != times.shape
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError("LFP timestamps must be finite, strictly increasing and match a 1-D voltage array.")
    measured_dt = float(np.median(np.diff(times)))
    fs = 1. / measured_dt if sample_rate_hz is None else _positive(sample_rate_hz, "sample rate")
    dt = 1. / fs
    if not 0 <= sampling_tolerance < .5:
        raise ValueError("sampling_tolerance must lie in [0, .5).")
    if not 1 < gap_factor:
        raise ValueError("gap_factor must exceed 1.")
    if not np.isfinite(edge_exclusion_s) or edge_exclusion_s < 0:
        raise ValueError("edge_exclusion_s must be finite and nonnegative.")
    floor_multiplier = _positive(amplitude_floor_multiplier, "amplitude_floor_multiplier")
    differences = np.diff(times)
    gaps = differences > gap_factor * dt
    if np.any(np.abs(differences[~gaps] / dt - 1.) > sampling_tolerance):
        raise ValueError("LFP sampling within a run is not regular at the supplied/measured sample rate.")
    if not 0 < stopband_hz[0] < passband_hz[0] < passband_hz[1] < stopband_hz[1] < fs / 2:
        raise ValueError("Require 0 < stop_low < pass_low < pass_high < stop_high < Nyquist.")
    sos = signal.iirdesign(
        passband_hz, stopband_hz, gpass=passband_ripple_db,
        gstop=stopband_attenuation_db, ftype="ellip", fs=fs, output="sos",
    )
    padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    filtered = np.full(len(times), np.nan)
    amplitude = np.full(len(times), np.nan)
    phase = np.full(len(times), np.nan)
    unwrapped = np.full(len(times), np.nan)
    segment_ids = np.full(len(times), -1, dtype=np.int64)
    valid = np.isfinite(voltage)
    boundaries = np.r_[0, np.flatnonzero(gaps | ~valid[:-1] | ~valid[1:]) + 1, len(times)]
    support = []
    next_segment = 0
    run_qc = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start <= padlen or not np.all(valid[start:stop]):
            continue
        indices = np.arange(start, stop)
        source = voltage[start:stop]
        median = float(np.median(source))
        scale = max(abs(median), float(np.quantile(np.abs(source - median), .95)))
        floor = max(np.finfo(float).tiny, floor_multiplier * np.finfo(float).eps * scale)
        qc = dict(start_s=float(times[start]), end_s=float(times[stop - 1]),
                  input_scale=scale, amplitude_numerical_floor=floor,
                  constant_or_numerically_flat=bool(np.ptp(source) <= floor),
                  below_numerical_floor_samples=0, valid_phase_samples=0)
        run_qc.append(qc)
        if qc["constant_or_numerically_flat"]:
            # Even nonzero DC can leave tiny filter/Hilbert residuals; their
            # angle is numerical noise, not a measured oscillatory phase.
            continue
        local = signal.sosfiltfilt(sos, source, padlen=padlen)
        analytic = signal.hilbert(local)
        filtered[start:stop] = local
        local_amplitude = np.abs(analytic)
        amplitude_ok = np.isfinite(local_amplitude) & (local_amplitude > floor)
        qc["below_numerical_floor_samples"] = int(np.count_nonzero(~amplitude_ok))
        keep = ((times[indices] >= times[start] + edge_exclusion_s)
                & (times[indices] <= times[stop - 1] - edge_exclusion_s)
                & amplitude_ok)
        if np.count_nonzero(keep) < 2:
            continue
        selected = indices[keep]
        # Retain the full-run unwrap before selection. Removing an invalid
        # amplitude sample must neither compress phase nor bridge its gap.
        local_phase = np.angle(analytic)
        local_unwrapped = np.unwrap(local_phase)
        cuts = np.flatnonzero(np.diff(selected) != 1) + 1
        for run in np.split(selected, cuts):
            if len(run) < 2:
                continue
            relative = run - start
            phase[run] = local_phase[relative]
            unwrapped[run] = local_unwrapped[relative]
            amplitude[run] = local_amplitude[relative]
            segment_ids[run] = next_segment
            support.append([times[run[0]], times[run[-1]]])
            next_segment += 1
            qc["valid_phase_samples"] += len(run)
    return dict(
        times_s=times.copy(), filtered_signal=filtered, amplitude=amplitude,
        phase_rad=phase, unwrapped_phase_rad=unwrapped,
        segment_id=segment_ids, valid_mask=segment_ids >= 0,
        support_intervals_s=np.asarray(support, dtype=float).reshape(-1, 2),
        sample_rate_hz=fs,
        phase_qc=dict(runs=run_qc,
                      constant_run_count=sum(q["constant_or_numerically_flat"] for q in run_qc),
                      valid_phase_samples=int(np.count_nonzero(segment_ids >= 0))),
        parameters=dict(passband_hz=list(passband_hz), stopband_hz=list(stopband_hz),
                        filter_design="elliptic_sos_forward_backward",
                        passband_ripple_db=passband_ripple_db,
                        stopband_attenuation_db=stopband_attenuation_db,
                        edge_exclusion_s=edge_exclusion_s, gap_factor=gap_factor,
                        sampling_tolerance=sampling_tolerance,
                        phase_zero="positive_filtered_voltage_peak",
                        n_sos_sections=len(sos),
                        amplitude_floor_multiplier=floor_multiplier,
                        amplitude_floor_method="multiplier * float64_eps * max(abs(input_median), p95(abs(input-input_median)))",
                        amplitude_numerical_floors=[q["amplitude_numerical_floor"] for q in run_qc],
                        physiological_amplitude_threshold=None,
                        automatic_high_amplitude_artifact_rejection=False),
    )


def summarize_lfp_reference(lfp, theta, moving_intervals_s):
    """Prepare full-baseline PSD/QC and a five-second movement trace for plotting.

    Welch uses finite, nonoverlapping four-second blocks. The excerpt is
    centered on the longest moving interval. With no movement, PSD/QC remain
    available and the trace is empty with status ``no_movement``. Returned
    traces are small copies, so callers can release the full raw LFP afterward.
    """
    times = lfp["times_s"]
    voltage = lfp["voltage_uv"]
    fs = lfp["sample_rate_hz"]
    nwin = int(round(4 * fs))
    blocks = voltage[:len(voltage) // nwin * nwin].reshape(-1, nwin)
    finite_blocks = np.isfinite(blocks).all(axis=1)
    frequency, block_psd = signal.welch(
        blocks[finite_blocks], fs=fs, window="hann",
        nperseg=nwin, noverlap=0, detrend="constant", axis=-1,
    )
    psd = block_psd.mean(axis=0)
    theta_band = (frequency >= 4) & (frequency <= 11)
    reference_band = (frequency >= 1) & (frequency <= 40)
    total_power = np.trapezoid(psd[reference_band], frequency[reference_band])
    fraction = np.trapezoid(psd[theta_band], frequency[theta_band]) / total_power
    movement = _intervals(moving_intervals_s)
    midpoint = np.mean(movement[np.argmax(np.diff(movement, axis=1))]) if len(movement) else np.nan
    selected = np.abs(times - midpoint) <= 2.5
    return dict(
        summary=dict(
            valid_lfp_seconds=np.diff(theta["support_intervals_s"], axis=1).sum(),
            theta_amplitude_median_uv=np.nanmedian(theta["amplitude"]),
            raw_std_uv=np.nanstd(voltage), finite_fraction=np.isfinite(voltage).mean(),
            phase_valid_fraction=theta["valid_mask"].mean(),
            constant_lfp_runs=theta["phase_qc"]["constant_run_count"],
            power_4_11_over_1_40=fraction, welch_seconds=4., welch_overlap_seconds=0.,
            spectrum_scope="full baseline; finite 4-second blocks; mean Hann periodograms",
        ),
        spectrum=dict(frequency_hz=frequency, psd_uv2_per_hz=psd),
        trace=dict(
            status="movement_excerpt" if len(movement) else "no_movement",
            midpoint_s=midpoint, time_from_midpoint_s=times[selected] - midpoint,
            voltage_uv=voltage[selected], filtered_uv=theta["filtered_signal"][selected],
            amplitude_uv=theta["amplitude"][selected],
        ),
    )


def _theta_arrays(theta: Mapping):
    times = np.asarray(theta["times_s"], dtype=float)
    unwrapped = np.asarray(theta["unwrapped_phase_rad"], dtype=float)
    ids = np.asarray(theta["segment_id"])
    if (times.ndim != 1 or unwrapped.shape != times.shape or ids.shape != times.shape
            or ids.dtype.kind not in "iu" or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0) or np.any(ids < -1)
            or not np.all(np.isfinite(unwrapped[ids >= 0]))):
        raise ValueError("Theta requires aligned increasing times, finite valid phases, and integer segment IDs.")
    blocks = []
    for segment_id in np.unique(ids[ids >= 0]):
        indices = np.flatnonzero(ids == segment_id)
        if len(indices) < 2 or np.any(np.diff(indices) != 1):
            raise ValueError("Each theta segment must be one contiguous run of at least two samples.")
        blocks.append((int(segment_id), indices))
    return times, unwrapped, ids, blocks


def sample_theta_phase(spike_times_s, theta: Mapping):
    """Interpolate unwrapped phase inside each valid run, never across a gap."""
    spikes = _spikes(spike_times_s)
    times, unwrapped, _, blocks = _theta_arrays(theta)
    phases = np.full(len(spikes), np.nan)
    segment_ids = np.full(len(spikes), -1, dtype=np.int64)
    for segment_id, indices in blocks:
        selected = (spikes >= times[indices[0]]) & (spikes <= times[indices[-1]])
        phases[selected] = np.interp(spikes[selected], times[indices], unwrapped[indices])
        segment_ids[selected] = segment_id
    return dict(phase_rad=(phases + np.pi) % (2 * np.pi) - np.pi,
                unwrapped_phase_rad=phases, segment_id=segment_ids,
                valid_mask=segment_ids >= 0)


def phase_locking(spike_times_s, theta: Mapping, *, n_phase_bins=18, p_threshold=.01,
                  exposure_intervals_s=None):
    """Return circular spike statistics and occupancy-normalized phase rates.

    Rayleigh's finite-n approximation is applied to spikes (no assumption of
    equal phase occupancy in the rate histogram). Raw count and seconds arrays
    remain separate, with NaN rate for unobserved bins and zero for observed
    bins with no spikes. ``p_threshold=.01`` is an explicit implementation
    choice; the Long et al. paper does not establish this phase-locking cutoff.
    Supply movement exposure intervals to condition BOTH spike statistics and
    occupancy on movement. Omission uses every valid LFP support interval.
    """
    if not isinstance(n_phase_bins, (int, np.integer)) or n_phase_bins < 3:
        raise ValueError("n_phase_bins must be an integer >= 3.")
    if not 0 < p_threshold < 1:
        raise ValueError("p_threshold must lie between 0 and 1.")
    spikes = _spikes(spike_times_s)
    sampled = sample_theta_phase(spikes, theta)
    exposure = None if exposure_intervals_s is None else _intervals(exposure_intervals_s)
    if exposure is not None:
        if len(exposure):
            interval_ids = np.searchsorted(exposure[:, 0], spikes, side="right") - 1
            exposed = (interval_ids >= 0) & (spikes < exposure[np.maximum(interval_ids, 0), 1])
        else:
            exposed = np.zeros(len(spikes), dtype=bool)
        sampled["valid_mask"] &= exposed
        sampled["phase_rad"][~sampled["valid_mask"]] = np.nan
    phases = sampled["phase_rad"][sampled["valid_mask"]]
    n = len(phases)
    vector = np.mean(np.exp(1j * phases)) if n else complex(np.nan, np.nan)
    mvl = float(min(abs(vector), 1.)) if n else np.nan
    preferred = float(np.angle(vector)) if n else np.nan
    if n:
        resultant = n * mvl
        rayleigh_p = float(np.clip(np.exp(np.sqrt(1 + 4 * n + 4 * (n * n - resultant ** 2))
                                         - (1 + 2 * n)), 0., 1.))
    else:
        rayleigh_p = np.nan
    times, unwrapped, _, blocks = _theta_arrays(theta)
    edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    occupancy = np.zeros(n_phase_bins)
    def cumulative_exposure(query):
        if not len(exposure):
            return np.zeros_like(query)
        duration = exposure[:, 1] - exposure[:, 0]
        prefix = np.r_[0., np.cumsum(duration)]
        ids = np.searchsorted(exposure[:, 0], query, side="right") - 1
        clipped_ids = np.maximum(ids, 0)
        result = prefix[clipped_ids] + np.clip(
            query - exposure[clipped_ids, 0], 0., duration[clipped_ids],
        )
        return np.where(ids >= 0, result, 0.)
    for _, indices in blocks:
        # Each observed time interval contributes exactly once. Midpoint phase
        # approximates its bin assignment at the supplied LFP sampling rate.
        midpoint = (unwrapped[indices[:-1]] + unwrapped[indices[1:]]) / 2
        midpoint = (midpoint + np.pi) % (2 * np.pi) - np.pi
        weights = (np.diff(times[indices]) if exposure is None else
                   np.maximum(0., np.diff(cumulative_exposure(times[indices]))))
        occupancy += np.histogram(midpoint, edges, weights=weights)[0]
    counts = np.histogram(phases, edges)[0]
    rates = np.divide(counts, occupancy, out=np.full(n_phase_bins, np.nan), where=occupancy > 0)
    return dict(
        n_spikes=n, n_spikes_excluded=len(spikes) - n,
        preferred_phase_rad=preferred, mean_vector_length=mvl,
        pairwise_phase_consistency=float((n * mvl ** 2 - 1) / (n - 1)) if n > 1 else np.nan,
        rayleigh_p=rayleigh_p, rayleigh_z=float(n * mvl ** 2) if n else np.nan,
        is_phase_locked=bool(np.isfinite(rayleigh_p) and rayleigh_p < p_threshold),
        phase_bin_edges_rad=edges, phase_counts=counts,
        phase_occupancy_s=occupancy, phase_rate_hz=rates,
        spike_phase_rad=sampled["phase_rad"], spike_valid_mask=sampled["valid_mask"],
        parameters=dict(n_phase_bins=n_phase_bins, p_threshold=p_threshold,
                        rayleigh_approximation="Zar_finite_n_exponential",
                        occupancy_assignment="sample_interval_phase_midpoint",
                        exposure_selection="all_valid_lfp" if exposure is None else "explicit_intervals",
                        exposure_intervals_s=None if exposure is None else exposure.tolist()),
    )


def phase_precession(
    spike_times_s, theta: Mapping, *, bin_cycles=.01, max_lag_cycles=4.,
    peak_band=(.85, 1.3), peak_half_width=.025,
    reference_bands=((.5, .85), (1.3, 1.8)), minimum_peak_count=5,
    threshold=1.5, n_fft=8192, spectrum_convention="magnitude", detrend=False,
    frequency_tolerance=None, require_monotonic_theta=False,
):
    """Exploratory paper-style phase-ACG spectral test in cumulative theta cycles.

    Phase ACGs use distinct event pairs within individual valid LFP runs. The
    test requires an ACG peak > 5 pairs, spectral ratio >= 1.5 against EACH
    sideband, and a peak above 1 cycle/cycle. Testing each sideband separately
    is a conservative reading of reference 33's ambiguous "both bands" wording.
    A faster spike cycle corresponds
    to advancing spike phase. Candidate classification additionally requires
    a frequency margin (default half an independent Fourier bin) because a
    padded finite-window spectrum can put perfectly locked spikes just above 1.
    Bin size, FFT padding and peak search band are explicit implementation
    choices because the paper does not give all numerical details. Padding
    interpolates the spectrum and does not improve its independent resolution.
    This is not a spatial phase-versus-position regression.
    """
    threshold = _positive(threshold, "threshold")
    if not np.isfinite(minimum_peak_count) or minimum_peak_count < 0:
        raise ValueError("minimum_peak_count must be finite and nonnegative.")
    spikes = _spikes(spike_times_s)
    sampled = sample_theta_phase(spikes, theta)
    _, unwrapped, _, blocks = _theta_arrays(theta)
    monotonic = all(np.all(np.diff(unwrapped[indices]) > 0) for _, indices in blocks)
    segments = []
    for segment_id, _ in blocks:
        cycles = sampled["unwrapped_phase_rad"][sampled["segment_id"] == segment_id] / (2 * np.pi)
        # Sorting coordinates does not change the symmetric pair-difference
        # histogram. Preserve local phase reversals and report them; callers
        # can explicitly require monotonic phase for a more restrictive gate.
        segments.append(np.sort(cycles))
    lags, counts, n_spikes = _pair_acg(segments, bin_cycles, max_lag_cycles)
    frequencies, spectrum = _acg_spectrum(
        counts, bin_cycles, spectrum_convention=spectrum_convention,
        detrend=detrend, n_fft=n_fft,
    )
    result = _spectral_index(frequencies, spectrum, peak_band, peak_half_width, reference_bands)
    peak = result.pop("peak_frequency")
    result.pop("index")
    sideband_means = np.array([
        np.mean(spectrum[(frequencies >= low) & (frequencies <= high)])
        for low, high in reference_bands
    ])
    denominator = float(sideband_means.max())
    index = result["peak_mean_spectrum"] / denominator if denominator > 0 else np.nan
    result["reference_mean_spectrum"] = denominator
    independent_resolution = 1. / (2 * max_lag_cycles + bin_cycles)
    if frequency_tolerance is None:
        frequency_tolerance = independent_resolution / 2
    if not np.isfinite(frequency_tolerance) or frequency_tolerance < 0:
        raise ValueError("frequency_tolerance must be finite and nonnegative.")
    peak_count = int(counts.max(initial=0))
    eligible = bool((monotonic or not require_monotonic_theta)
                    and peak_count > minimum_peak_count and np.isfinite(index))
    candidate = bool(eligible and index >= threshold and peak > 1. + frequency_tolerance)
    return dict(
        phase_precession_index=index, peak_frequency_per_cycle=peak,
        acg_peak_count=peak_count, eligible=eligible,
        is_phase_precessing=candidate, precession_candidate=candidate,
        paper_peak_above_one=bool(np.isfinite(peak) and peak > 1.),
        sideband_mean_spectrum=sideband_means,
        theta_phase_monotonic=monotonic, n_spikes=n_spikes,
        acg_lags_cycles=lags, acg_counts=counts,
        frequency_per_cycle=frequencies, spectrum=spectrum, **result,
        parameters=dict(bin_cycles=bin_cycles, max_lag_cycles=max_lag_cycles,
                        peak_band=list(peak_band), peak_half_width=peak_half_width,
                        reference_bands=[list(band) for band in reference_bands],
                        minimum_peak_count=minimum_peak_count, threshold=threshold,
                        n_fft=n_fft, spectrum_convention=spectrum_convention,
                        detrend=bool(detrend), exact_published_replication=False,
                        classification_requires_monotonic_theta=bool(require_monotonic_theta),
                        reference_rule="ratio_to_each_band_use_largest_mean",
                        frequency_tolerance=frequency_tolerance,
                        independent_frequency_resolution=independent_resolution),
    )
