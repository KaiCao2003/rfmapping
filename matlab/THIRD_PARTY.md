# MATLAB dependencies and compilation

The regular RF generator is included in this directory. The dependency subset below was copied unchanged from the working MATLAB source tree on 2026-09-22. Exact source locations, snapshot revisions, and file hashes are recorded in [source provenance](../docs/source_provenance.md).

## FMAToolbox subset

The following files come from the FMAToolbox copy under `buzcode-master/externalPackages/FMAToolbox/`:

| File | Used for |
| --- | --- |
| `Analyses/Sync.m` | Aligning spike times to stimulus times. |
| `General/FindInInterval.c` | Compiled interval lookup called by `Sync`. |
| `General/FindInInterval.m` | Help text for the compiled function; this file is not an executable MATLAB replacement. |
| `Helpers/isdscalar.m` | Input checks used by the included functions. |
| `Helpers/isdvector.m` | Input checks used by the included functions. |
| `Helpers/isstring_FMAT.m` | Input checks used by the included functions. |
| `Plot/Bright.m` | Colormap used by the included plot function. |
| `Plot/PlotColorMap.m` | RF figure rendering. |

The original notices credit **Michaël Zugaro** and specify the **GNU General Public License, version 3 or any later version**. Those notices remain in the source. The original supplied GPL text is included unchanged at [`buzcode-master/LICENSE`](buzcode-master/LICENSE).

This is the dependency subset needed by the included RF code, not a complete FMAToolbox or buzcode installation. The copied source directory had no separate Git metadata identifying its original upstream revision; the provenance file records its exact bytes instead. The files under `Utils/` come from the RF MATLAB source repository and are not relabeled as FMAToolbox files.

## Compile FindInInterval on the computer running MATLAB

Compiled MEX files depend on the operating system, processor architecture, and MATLAB compatibility. No precompiled MEX file is included. Compile the provided C source on each computer that will run the MATLAB RF generator.

1. Open MATLAB.
2. Set `rfMatlabDir` to this repository's `matlab` directory using the path syntax for your computer.
3. Run the following commands:

```matlab
% Linux/macOS example:
rfMatlabDir = '/path/to/rfmapping/matlab';

% Windows example: use this assignment instead of the one above.
% rfMatlabDir = 'C:\data\code\rfmapping\matlab';

fmatDir = fullfile(rfMatlabDir, 'buzcode-master', ...
    'externalPackages', 'FMAToolbox');
addpath(fullfile(rfMatlabDir, 'Utils'));
addpath(genpath(fmatDir));

mex -setup C
intervalDir = fullfile(fmatDir, 'General');
mex('-outdir', intervalDir, fullfile(intervalDir, 'FindInInterval.c'));

which FindInInterval -all
```

Select a C compiler supported by your installed MATLAB release when prompted. Linux, macOS, and Windows use the same MATLAB commands above; the compiler and generated MEX extension differ. A file built on one platform should not be copied to another as its executable dependency.

The first result from `which FindInInterval -all` must be the newly built MEX file. If MATLAB finds only `FindInInterval.m`, compilation has not produced an available executable. If it finds a different installation first, resolve that path order before starting the RF generator.

The interval helper expects spike timestamps as MATLAB doubles. The `adc_spike_time.npy` input must therefore use NumPy `float64`. The timestamp exporter preserves the source probe timestamp dtype; use the original Open Ephys timestamps in seconds, without converting them to `float32`.

See the [main README](../README.md) for data preparation and the complete RF generation configuration.
