#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify v5 deliverables."""
import numpy as np, json, csv
from pathlib import Path
from scipy.io import loadmat

datasets = [
    ('ygh', Path(r'D:\LMC\data\phantom_ygh\最终处理结果\推理')),
    ('user', Path(r'D:\LMC\data\水膜数据处理\最终处理结果\推理')),
]

for name, out in datasets:
    print('=' * 70)
    print('DATASET:', name)
    print('=' * 70)
    d = np.load(str(out / 'sr_results_v5.npz'))
    print('\nNPZ keys ({}):'.format(len(d.files)))
    for k in sorted(d.files):
        a = d[k]
        if a.dtype.kind in ('U', 'S', 'O'):
            print('  {:40s} shape={:15s} dtype={:10s} value={!r}'.format(
                k, str(a.shape), str(a.dtype), a.item() if a.size == 1 else a.tolist()))
        else:
            print('  {:40s} shape={:15s} dtype={:10s} min={:.4f} max={:.4f}'.format(
                k, str(a.shape), str(a.dtype), float(a.min()), float(a.max())))

    bad = []
    for k in d.files:
        a = d[k]
        if a.dtype.kind == 'f' and a.ndim in (2, 3):
            mn, mx = float(a.min()), float(a.max())
            if mn < -1e-6 or mx > 1.0 + 1e-6:
                bad.append((k, mn, mx))
    print('\nRange check outside [0,1]:', bad if bad else 'ALL OK')

    m = loadmat(str(out / 'sr_quantitative_v5.mat'))
    vars_ = [k for k in m.keys() if not k.startswith('__')]
    print('\nMAT variables ({}):'.format(len(vars_)))
    for k in sorted(vars_):
        print('  {:40s} shape={} dtype={}'.format(k, m[k].shape, m[k].dtype))

    meta = json.load(open(str(out / 'inference_meta_v5.json'), 'r', encoding='utf-8'))
    print('\nMeta top-level keys:', list(meta.keys()))
    print('  selected_orientation =', meta.get('selected_orientation'))
    print('  orientation_tie      =', meta.get('orientation_tie'))
    print('  tied_orientations    =', meta.get('tied_orientations'))
    print('  overall_best         =', meta.get('overall_best'))
    print('  arrays list length   =', len(meta.get('arrays', [])))
    if meta.get('arrays'):
        print('  first 3 array entries:')
        for e in meta['arrays'][:3]:
            print('   ', e)

    with open(str(out / 'metrics_v5.csv')) as f:
        rows = list(csv.DictReader(f))
    print('\nmetrics_v5.csv rows:', len(rows))

    def load_summary(p):
        with open(p) as f:
            return list(csv.DictReader(f))

    s4 = load_summary(out / 'metrics_summary.csv')
    s5 = load_summary(out / 'metrics_summary_v5.csv')
    print('v4 summary rows={}, v5 summary rows={}'.format(len(s4), len(s5)))
    print('v4 summary:')
    for r in s4:
        print('  ', {k: r[k] for k in list(r.keys())[:6]})
    print('v5 summary:')
    for r in s5:
        print('  ', {k: r[k] for k in list(r.keys())[:6]})
    print()
