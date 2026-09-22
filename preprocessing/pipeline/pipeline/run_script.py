import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from .operations import downsample_probes, load_probes, sort_probes_subprocess, concatenate_probes, synchronize_probes
from .utils import setup_pipeline, copy_to_remote


def main():
    parser = argparse.ArgumentParser(description="Automated spike sorting pipeline")
    parser.add_argument('config', nargs='?', type=str, default='config.yaml')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    config = setup_pipeline(args.config, debug=args.debug)

    probes = load_probes(config['session_paths'])

    concatenate_probes(probes, config)

    probe_paths = {
        name: Path(config['local_output']) / name / 'concat'
        for name in probes if name != 'OneBox-ADC'
    }

    sort_probes_subprocess(probe_paths, config)

    if config.get('target_fs') is not None:
        downsample_probes(probe_paths, config)

    synchronize_probes(probes, config)

    if config.get('remote_output') is not None:
        copy_to_remote(
            local_path=config['local_output'],
            remote_path=config['remote_output'],
            overwrite_mode=config['copy_mode'],
        )


if __name__ == "__main__":
    main()
