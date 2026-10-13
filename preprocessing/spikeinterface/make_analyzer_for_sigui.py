import spikeinterface.full as si
import probeinterface as pi
from pathlib import Path

def make_analyzer_for_sigui(
    probe_list: list[str], session_name: str, *,
    data_base_folder: Path, analyzer_output_dir: Path,
):
    # =========================
    # probe_list: list[str] = ["A",]
    # session_name: str = "260730_123"

    for probe in probe_list:
        base_folder = data_base_folder / "Data" / session_name / f"Probe{probe}"

        # Concatenated continuous/probe.dat or .raw file
        RAW_FILE = Path(f"{base_folder}/concat/traces_cached_seg0.raw")
        probe_group = pi.read_probeinterface(f"{base_folder}/concat/probe.json")
        # Kilosort4 output folder
        KS4_FOLDER = Path(f"{base_folder}/kilosort/")

        # Output SortingAnalyzer folder
        ANALYZER_FOLDER = analyzer_output_dir / f"{session_name}_{probe}"

        # Neuropixels
        SAMPLING_FREQUENCY = 30000
        NUM_CHANNELS = 384
        DTYPE = "int16"

        # Open Ephys / NP  scale
        GAIN_TO_UV = 0.195
        OFFSET_TO_UV = 0.0

        TIME_AXIS = 0


        # =========================
        # 2. Load recording
        # =========================

        print("Loading raw binary recording...")

        recording = si.read_binary(
            file_paths=str(RAW_FILE),
            sampling_frequency=SAMPLING_FREQUENCY,
            num_channels=NUM_CHANNELS,
            dtype=DTYPE,
            gain_to_uV=GAIN_TO_UV,
            offset_to_uV=OFFSET_TO_UV,
            time_axis=TIME_AXIS,
        )
        recording = recording.set_probegroup(probe_group)

        print(recording)


        # =========================
        # 3. Load Kilosort4 sorting
        # =========================

        print("Loading Kilosort4 output...")

        sorting = si.read_kilosort(
            folder_path=str(KS4_FOLDER),
        )

        print(sorting)
        print("Unit ids:", sorting.unit_ids[:10], "...")
        print("Number of units:", len(sorting.unit_ids))


        # =========================
        # 4. Create SortingAnalyzer
        # =========================

        print("Creating SortingAnalyzer...")

        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            folder=str(ANALYZER_FOLDER),
            format="binary_folder",
            overwrite=True,
        )

        print(analyzer)


        # =========================
        # 5. Compute extensions for GUI
        # =========================

        print("Computing extensions...")

        analyzer.compute("random_spikes")
        analyzer.compute("waveforms")
        analyzer.compute("templates")
        analyzer.compute("spike_amplitudes")
        analyzer.compute("correlograms")

        """
        # Optional but useful
        try:
            analyzer.compute("unit_locations")
        except Exception as e:
            print("unit_locations failed, skipping.")
            print(e)

        try:
            analyzer.compute("template_similarity")
        except Exception as e:
            print("template_similarity failed, skipping.")
            print(e)
        """

        print("\nDone.")
        print(f"Analyzer folder: {ANALYZER_FOLDER}")
