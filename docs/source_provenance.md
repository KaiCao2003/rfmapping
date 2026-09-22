# Source provenance

This publication was assembled on **2026-09-22** from the working sources below. The remote copies were checked against their source files with SHA-256 after copying. All 33 remote source files, including the existing license, are byte-for-byte copies; the hashes below identify the actual snapshot even where the source was uncommitted.

## Source trees

| Published files | Source | Source HEAD at copy time | Snapshot state |
| --- | --- | --- | --- |
| Root Python helpers, notebooks, and documentation | [KaiCao2003/rfmapping](https://github.com/KaiCao2003/rfmapping), local working tree | `320257dcfe89584d2077d475a2f80e34c650c250` | Includes current working changes. The publication narrows the file set, removes notebook outputs and unrelated cells, and updates documentation and packaging. |
| `matlab/RFmapping.m` and `matlab/Utils/` | [KaiCao2003/rfmapping_matlab](https://github.com/KaiCao2003/rfmapping_matlab), `hhw9l84:/mnt/ssd4.1/Matlab` | `71ee85bf61fee947fdf433a7037bb72a8e59baca` | Current working files, including the modified and untracked files listed below. |
| `matlab/buzcode-master/` | The FMAToolbox subset and license already present under `hhw9l84:/mnt/ssd4.1/Matlab/buzcode-master` | No separate upstream commit was available in this source directory | This directory is ignored by the MATLAB source repository. Its contents are identified by file hashes, not by the MATLAB repository HEAD. |
| `preprocessing/pipeline/` | [8Nero/pipeline](https://github.com/8Nero/pipeline), `hhw9l84:/home/kai/pipeline` | `04bed444276e028b35f694d2333663b18204c68e` | All 11 included files match the clean tracked source files. The upstream working tree also contains local configuration files; those are excluded. |
| `preprocessing/spikeinterface/` | [KaiCao2003/spikeinterface](https://github.com/KaiCao2003/spikeinterface), `hhw9l84:/home/kai/spikeinterface` | `b190c8c309363ee9f6383a8dae29b037fd03468f` | The splitter is modified and the timestamp exporter is untracked in that source tree. Both are copied from the current working files. |

The source paths in this table are provenance records. Users do not need access to those machines or directories; the included files are available in this repository.

## Uncommitted MATLAB files included

- Modified: `RFmapping.m`, `Utils/RFmapping_core.m`.
- Untracked: `Utils/RFmapping_group_spike_times.m`, `Utils/ReadSpikeData.m`, `Utils/SaveRfPatternCsv.m`, `Utils/SaveRfPatternPdf.m`.
- The other included files in `matlab/Utils/` are tracked and unchanged at the source HEAD above.

The regular RF source keeps its existing shared coordinate branches. This publication does not include the dedicated free-moving entry point or core, and the documented regular RF configuration disables the background-motion, allocentric-bin, and rotation options.

## Attribution and licenses

`preprocessing/pipeline/` is an attributed snapshot of **8Nero/pipeline**, whose project metadata names **Tuguldur** as the author. Its source, project metadata, dependency lock, and upstream README are preserved. No license file was present among that upstream repository's tracked files at the recorded revision; this publication does not invent or assign one.

The upstream pipeline README describes installing and updating the upstream project. To run the bundled snapshot as part of this workflow, follow the paths and separate environment instructions in the [main README](../README.md).

FMAToolbox author notices and its GPL terms remain in the copied source files. The supplied GPL text is preserved in [`matlab/buzcode-master/LICENSE`](../matlab/buzcode-master/LICENSE). See [MATLAB dependencies and compilation](../matlab/THIRD_PARTY.md) for the exact included subset and platform-specific MEX setup.

No new repository-wide license is assigned by this publication.

## SHA-256 of copied remote files

These hashes apply to the published file bytes. They were compared with the corresponding authoritative remote working files on the copy date.

| Published path | SHA-256 |
| --- | --- |
| `matlab/buzcode-master/externalPackages/FMAToolbox/Analyses/Sync.m` | `6bca4045cd14e8ffa3bf5f4e224287ec28a1a5fab639aebe709d88cd4e5cb1b5` |
| `matlab/buzcode-master/externalPackages/FMAToolbox/General/FindInInterval.c` | `7cbc2a1961cc69730ae1d408e3066bf21726bd3c57dbe56a2389a0a234c5bfe5` |
| `matlab/buzcode-master/externalPackages/FMAToolbox/General/FindInInterval.m` | `442752923ab2b39502df24f104417f000eec94d854017c9a935e97b6681e75ae` |
| `matlab/buzcode-master/externalPackages/FMAToolbox/Helpers/isdscalar.m` | `cb64ed9035de59d72ee5ea7fe41abe5126746b71cab591486a4531eee2a0b29a` |
| `matlab/buzcode-master/externalPackages/FMAToolbox/Helpers/isdvector.m` | `d8e2ba580717cd90cdf671c0fb3ac60000a722de7d4e34758f507a97b5b330dc` |
| `matlab/buzcode-master/externalPackages/FMAToolbox/Helpers/isstring_FMAT.m` | `690e00143760c697932e446fbbdaa0d840fa3aa480f9c7ba0cb988b584d820d8` |
| `matlab/buzcode-master/externalPackages/FMAToolbox/Plot/Bright.m` | `3f57004238a890911d2458577015bfb9a63a1d0ef8e774daa0d5d5553d0223b2` |
| `matlab/buzcode-master/externalPackages/FMAToolbox/Plot/PlotColorMap.m` | `281268ebeab01774cc235f512aa0099aaeb01fdd2c6eaee0cce2ae0565ce11c3` |
| `matlab/buzcode-master/LICENSE` | `589ed823e9a84c56feb95ac58e7cf384626b9cbf4fda2a907bc36e103de1bad2` |
| `matlab/RFmapping.m` | `06d43dfd41c6786d38e87e804f63af0395b84e03d521f4f225c674d55514c725` |
| `matlab/Utils/readNPY.m` | `67d5b2ee04ef7480373d0aa9098919e17db09778c6cefd94f571f4306aee4934` |
| `matlab/Utils/ReadSpikeData.m` | `b318b9fb8f2c71147732388695e6081082b88805673e75943de02c43c255e740` |
| `matlab/Utils/RFmapping_coarsen_trial_spatial_mask.m` | `3104d009345c8e7cdc1fdbb1c9f4420ac43e9dfe8d1d64922b5cc5a1b123704a` |
| `matlab/Utils/RFmapping_core.m` | `14cb1fc1f06f520432def4878bfa7d79116f852fc156e920b1cb073265bba269` |
| `matlab/Utils/RFmapping_group_spike_times.m` | `a80b2e174b79c564bf3d42bc1a9b012f76a99435b24b50ab313c2d3b941ad464` |
| `matlab/Utils/RFmapping_trial_spatial_mask.m` | `e1bc24cab0159d35fd72f46ba63365d2b6eba2cae544c354bbd9bb26865db18a` |
| `matlab/Utils/RFmapping_trial_spike_counts.m` | `52ae714656475f3dce91f7de35553888ea2996eaaa28462168102c41d9e7d6be` |
| `matlab/Utils/RFmapping_vertical_bar_coverage.m` | `6fe1a3114d6e0a5a235e01a6b89363fc31c3d0a8249ac544cb807356692d5f0f` |
| `matlab/Utils/SaveRfPatternCsv.m` | `cbbea04c925fcc38256353e9ddcbde062b4109835879188a637c061f089d5b47` |
| `matlab/Utils/SaveRfPatternPdf.m` | `4325834488fcce605fe05a55210a80f404bb6453bbad92c69a6e1406647c24f9` |
| `preprocessing/pipeline/config_template.yaml` | `451bffabca96e1e153e4b4c382feccf7aa94d7697b50c4c616d2d84e98d1daba` |
| `preprocessing/pipeline/pipeline/__init__.py` | `b42f42c0be0d9bc6eb55726764d1a4fdea196898fa78eb4b2ad3ab74932ba682` |
| `preprocessing/pipeline/pipeline/decimation.py` | `52d34d7614e6322afd416077dd10ba3f6a3135b14526ec7512767677c1c90684` |
| `preprocessing/pipeline/pipeline/operations.py` | `2f099c5d8605785e31058b0b038de36b3d8cc11e3ed2e28bc238333371d36273` |
| `preprocessing/pipeline/pipeline/probe.py` | `ad96772e369a6ac32e1148a1d4a150e38713546bbdaf06a115a24219d41e87ec` |
| `preprocessing/pipeline/pipeline/run_script.py` | `945ff89e6331183203adbcc59cc4108ba68c1fae1eadf924520bb3519136eaff` |
| `preprocessing/pipeline/pipeline/sort.py` | `1d3446131667025f002815a1a4e6dfeef2ba47c5936b15226fd20ec16e47f78f` |
| `preprocessing/pipeline/pipeline/utils.py` | `6c93ebb23a1e8f7fbb3e40c80af8f6a0b3f1f0aa46c09ddf440a6fd34c70160d` |
| `preprocessing/pipeline/pyproject.toml` | `e588678241c435647095bfa96aa5b274f38e204aae0ac97568a6211b47a4dfc2` |
| `preprocessing/pipeline/README.md` | `94064cf882ba5ea8d0191b8f47a3fa3b2f146a5032f1ed5de4961841c3afb740` |
| `preprocessing/pipeline/uv.lock` | `a382a29242db7c91af64b855a96be1fbeb50c9e2cafaf7f894a6f48f39938102` |
| `preprocessing/spikeinterface/generate_adc_spike_time.py` | `58e5e6a411d1bfac00cfcdb29ea02b188b5d05c6b1a607c97cda4db34d680f1e` |
| `preprocessing/spikeinterface/split_kilosort_by_recording.py` | `0350d2cd732e64256b37927b0eb601122289ca3212e3ec44c32d4d66baa9298d` |

Source mapping for this table:

- Strip `matlab/` and append the remaining path to `/mnt/ssd4.1/Matlab/`.
- Strip `preprocessing/pipeline/` and append the remaining path to `/home/kai/pipeline/`.
- Strip `preprocessing/spikeinterface/` and append the remaining path to `/home/kai/spikeinterface/`.
