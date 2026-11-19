"""Explicit cohort selection and summaries for the seven-mouse poster panels."""

import numpy as np
import pandas as pd


POSTER_RECORDINGS = [
    ('m14', '260609', 1, 3, ('A',)), ('m15', '260630', 1, 3, ('A', 'B')),
    ('m19', '260827', 9, 2, ('A',)), ('m21', '261006', 9, 3, ('A',)),
    ('m22', '261008', 9, 2, ('A',)), ('m23', '261015', 9, 2, ('A',)),
    ('m24', '261104', 10, 2, ('A',)),
]
RF_CATEGORIES = ['0 ON RF', '1 ON RF', '>1 ON RF', 'RF unassessed']


def population_unit_table(hd, result, raw_unit_ids, qc, *, adn_ids=None):
    """Keep failed RF QC and absent raw maps distinct from measured zero RFs."""
    ids = np.union1d(hd['unit_id'], raw_unit_ids)
    table = pd.DataFrame(index=pd.Index(ids, name='unit_id'))
    table['hd_class'] = pd.Series(hd['unit_data']['hd_class'], index=hd['unit_id'],
                                  dtype='Int64').reindex(ids)
    table['is_hd'] = table.hd_class.eq(3).fillna(False)
    table['raw_rf_present'] = np.isin(ids, raw_unit_ids)
    table['rf_record_present'] = np.isin(ids, result.unit_ids)
    table['n_on_rf'] = pd.Series(result.center_2d.sum(axis=(1, 2)),
                                 index=result.unit_ids, dtype='Int64').reindex(ids)
    table['rf_zero_or_missing_bins'] = pd.Series(qc['zero_bins'], index=qc['unit_ids'],
                                                 dtype='Int64').reindex(ids)
    table['assessment'] = 'raw RF absent'
    table.loc[table.raw_rf_present, 'assessment'] = 'failed RF bin QC'
    table.loc[table.rf_record_present, 'assessment'] = 'assessed'
    table['category'] = 'RF unassessed'
    for selected, category in ((table.n_on_rf.eq(0), '0 ON RF'),
                              (table.n_on_rf.eq(1), '1 ON RF'),
                              (table.n_on_rf.gt(1), '>1 ON RF')):
        table.loc[selected.fillna(False), 'category'] = category
    table['adn_annotation_available'] = adn_ids is not None
    table['in_adn_annotation'] = (pd.array(np.isin(ids, adn_ids), dtype='boolean')
                                  if adn_ids is not None else pd.NA)
    return table


def select_hd_population(table, *, anatomy='all_hd'):
    """Anatomical restrictions are explicit, never inferred from file presence."""
    keep = table.is_hd.copy()
    if anatomy == 'm23_adn':
        keep &= table.mouse.ne('m23') | table.in_adn_annotation.fillna(False)
    elif anatomy == 'available_adn':
        keep &= ~table.adn_annotation_available | table.in_adn_annotation.fillna(False)
    elif anatomy != 'all_hd':
        raise ValueError(f'Unknown anatomical selection: {anatomy}')
    return table.loc[keep].copy()


def fraction_summary(table):
    counts = table.category.value_counts().reindex(RF_CATEGORIES, fill_value=0)
    total = int(counts.sum())
    assessed = total - int(counts['RF unassessed'])
    positive = int(counts['1 ON RF'] + counts['>1 ON RF'])
    return dict(n_hd=total, n_assessed=assessed, n_unassessed=total - assessed,
                n_on_rf_positive=positive, n_single=int(counts['1 ON RF']),
                n_multiple=int(counts['>1 ON RF']), n_zero=int(counts['0 ON RF']),
                on_rf_percent_assessed=100 * positive / assessed,
                on_rf_percent_all_hd=100 * positive / total,
                multiple_percent_rf_positive=100 * int(counts['>1 ON RF']) / positive)
