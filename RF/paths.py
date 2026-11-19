"""RF source and artifact names shared by analysis entrypoints.

Each stimulus owns a directory below its probe. Legacy flat sources remain
readable; readers prefer an existing organized source without generating it.
"""

from pathlib import Path
import re


RF_KIND_PATTERN = r"(?:regular|bar_\d+(?:\.\d+)?deg|(?:rotation|egocentric|allocentric_pixelbins)(?:_30)?|free_moving)"
_SOURCE = re.compile(rf"^(?P<kind>{RF_KIND_PATTERN})_(?P<session>\d{{6}}_\d+)$")
_LEGACY = re.compile(r"^(?P<kind>.+)_unitsSpikeCounts_(?P<session>\d{6}_\d+)(?P<tail>.*)$")


def is_organized_rf_source(source):
    source = Path(source)
    match = _SOURCE.fullmatch(source.stem)
    return bool(source.suffix in (".rfmap", ".json") and match
                and source.parent.name == match['kind'])


def organized_rf_source(source):
    """Return the organized destination, without moving or creating anything."""
    source = Path(source)
    if is_organized_rf_source(source):
        return source
    match = _LEGACY.fullmatch(source.stem)
    if match is None:
        return source
    kind, tail = match['kind'], match['tail']
    tail = re.sub(r"_(?:off|gray)$", "", tail)
    if tail in ("free_moving", "_free_moving"):
        kind = 'free_moving'
    elif bar := re.fullmatch(r"_vertical_bar_pooled_bin(\d+(?:\.\d+)?)deg", tail):
        kind = f"bar_{bar[1]}deg"
    elif tail:
        return source
    if re.fullmatch(RF_KIND_PATTERN, kind) is None:
        return source
    return source.parent / kind / f"{kind}_{match['session']}{source.suffix}"


def resolve_rf_source(source):
    """Prefer the organized copy of a known legacy path when it exists."""
    source = Path(source)
    organized = organized_rf_source(source)
    return organized if organized.is_file() else source


def legacy_rf_source_stem(source):
    """Identify legacy CSVs and source-qualified comparison exports."""
    source = Path(source)
    if not is_organized_rf_source(source):
        return source.stem
    match = _SOURCE.fullmatch(source.stem)
    kind, session = match['kind'], match['session']
    if kind.startswith('bar_'):
        stem = f'regular_unitsSpikeCounts_{session}_vertical_bar_pooled_bin{kind[4:]}'
    elif kind == 'free_moving':
        stem = f'regular_unitsSpikeCounts_{session}free_moving'
    else:
        stem = f'{kind}_unitsSpikeCounts_{session}'
    if 'rfmapping_off' in source.parts:
        stem += '_off'
    elif 'rfmapping_gray' in source.parts:
        stem += '_gray'
    return stem


def rf_comparison_names(source, probe, *, rf_type, rf_only=True):
    """Return the source-qualified filename and the shared filename owned by regular ON."""
    source = Path(source).absolute()
    stem = legacy_rf_source_stem(source)
    selection = '_rfonly' if rf_only else ''
    suffix = f'{rf_type}_x_2d{selection}_Probe{probe}.csv'
    regular_on = 'rfmapping' in source.parts and re.fullmatch(r'regular_unitsSpikeCounts_\d{6}_\d+', stem)
    return f'{stem}_{suffix}', f'rf_{suffix}' if regular_on else None


def rf_source_path(session, probe, *, kind='regular', root_kind='rfmapping',
                   time_folder='-100_400_1ms'):
    session = Path(session)
    probe = str(probe).removeprefix('Probe')
    directory = session / 'data' / root_kind / 'good' / time_folder / f'Probe{probe}'
    organized = directory / kind / f'{kind}_{session.name}.rfmap'
    if organized.is_file():
        return organized
    suffix = '_off' if root_kind == 'rfmapping_off' else '_gray' if root_kind == 'rfmapping_gray' else ''
    if kind.startswith('bar_'):
        stem = f'regular_unitsSpikeCounts_{session.name}_vertical_bar_pooled_bin{kind[4:]}'
    elif kind == 'free_moving':
        stem = f'regular_unitsSpikeCounts_{session.name}free_moving'
    else:
        stem = f'{kind}_unitsSpikeCounts_{session.name}'
    legacy = directory / f'{stem}{suffix}.rfmap'
    return legacy if legacy.is_file() else organized


def rf_probe(source):
    for parent in Path(source).parents:
        if re.fullmatch(r'Probe[A-Za-z]', parent.name):
            return parent.name.removeprefix('Probe')
    raise ValueError(f'RF source is not inside a Probe directory: {source}')


def rf_projection_path(source, *, axis='x', rf_only=True, rf_type='excitatory'):
    """Name a native projection beside its organized RF source; perform no I/O."""
    if axis not in ('x', 'y'):
        raise ValueError(f'Unknown RF projection axis: {axis}')
    labels = {'excitatory': 'ext', 'inhibitory': 'inh'}
    if rf_type not in labels:
        raise ValueError(f'Unknown RF polarity: {rf_type}')
    selection = f'rfonly_{labels[rf_type]}' if rf_only else 'all'
    return organized_rf_source(source).with_name(f'tc_{axis}_{selection}.csv')


def legacy_rf_projection_paths(source, *, axis='x', rf_only=True, rf_type='excitatory'):
    """List old native CSV names, keeping luminance and stimulus identity."""
    source = organized_rf_source(source)
    root = next(parent for parent in source.parents
                if parent.name in ('rfmapping', 'rfmapping_off', 'rfmapping_gray'))
    short_stem = source.stem + root.name.removeprefix('rfmapping')
    stems = dict.fromkeys((legacy_rf_source_stem(source), short_stem))
    polarity = '_inhibitory' if rf_only and rf_type == 'inhibitory' else ''
    selection = '_rfonly' if rf_only else ''
    axes = ('', '_x') if axis == 'x' else ('_y',)
    return tuple(root.parent / f'{stem}_Probe{rf_probe(source)}{polarity}_1d{suffix}{selection}.csv'
                 for stem in stems for suffix in axes)


def resolve_rf_projection(source, *, axis='x', rf_only=True, rf_type='excitatory'):
    """Read canonical names first, with legacy-file support and no generation."""
    options = dict(axis=axis, rf_only=rf_only, rf_type=rf_type)
    canonical = rf_projection_path(source, **options)
    if canonical.is_file():
        return canonical
    existing = [path for path in legacy_rf_projection_paths(source, **options) if path.is_file()]
    if existing:
        if len(existing) > 1:
            first = existing[0].read_bytes()
            if any(path.read_bytes() != first for path in existing[1:]):
                raise ValueError(f'Conflicting legacy RF projections: {existing}')
        return existing[0]
    return canonical
