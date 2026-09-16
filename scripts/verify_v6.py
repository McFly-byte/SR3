#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent v6 acceptance verification."""
import json, csv, zipfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

JOBS = [
    ('ygh',  Path(r'D:\LMC\data\phantom_ygh\最终处理结果\推理'), 'identity'),
    ('user', Path(r'D:\LMC\data\水膜数据处理\最终处理结果\推理'), 'transpose'),
]

def psnr(a, b):
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))

def centroid(a):
    """(row, col) centroid weighted by intensity."""
    s = a.sum()
    if s <= 0: return (float('nan'), float('nan'))
    rr, cc = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return (float((rr*a).sum()/s), float((cc*a).sum()/s))

report = {}

for name, out, expect_t in JOBS:
    r = {'checks': {}}
    d = np.load(str(out/'sr_results_v6.npz'))
    # 1. display_transform matches expectation
    got_t = str(d['display_transform'])
    r['checks']['display_transform_value'] = {'expected': expect_t, 'got': got_t, 'pass': got_t == expect_t}

    # 2. model vs display correspondence
    inv = []
    for k in d.files:
        if k.startswith('model_'):
            dk = 'display_' + k[len('model_'):]
            if dk not in d.files: continue
            a, b = d[k], d[dk]
            if expect_t == 'identity':
                ok = np.allclose(a, b, atol=1e-6)
            else:  # transpose
                if a.ndim == 2:
                    ok = np.allclose(a.T, b, atol=1e-6)
                else:
                    ok = all(np.allclose(a[i].T, b[i], atol=1e-6) for i in range(a.shape[0]))
            if not ok:
                inv.append(k)
    r['checks']['model_display_correspondence'] = {'mismatches': inv, 'pass': len(inv)==0}

    # 3. range [0,1] for all float arrays
    bad_range = []
    for k in d.files:
        a = d[k]
        if a.dtype.kind == 'f' and a.ndim >= 2:
            if a.min() < -1e-5 or a.max() > 1+1e-5:
                bad_range.append((k, float(a.min()), float(a.max())))
    r['checks']['range_0_1'] = {'violations': bad_range, 'pass': len(bad_range)==0}

    # 4. PSNR invariance across all 3 variants × 3 models (best seeds)
    max_diff = 0.0
    per_variant = []
    hr_m = d['model_hr_bicubic_64']; hr_d = d['display_hr_bicubic_64']
    hr_n_m = d['model_hr_native']; hr_n_d = d['display_hr_native']
    for mk in ['healthy_ema','healthy_raw','current_mixed_raw']:
        for v in ['raw_sr_best','dc_sr_best','dc_denoised_sr_best','final_sr']:
            key = f'{mk}_{v}'
            if f'model_{key}' not in d.files or f'display_{key}' not in d.files:
                per_variant.append({'array': key, 'missing': True}); continue
            pm = psnr(d[f'model_{key}'], hr_m)
            pd_ = psnr(d[f'display_{key}'], hr_d)
            diff = abs(pm-pd_)
            max_diff = max(max_diff, diff)
            per_variant.append({'array': key, 'psnr_model': round(pm,6), 'psnr_display': round(pd_,6), 'diff': diff})
    r['checks']['psnr_invariance_all_variants'] = {
        'max_abs_diff': max_diff, 'tolerance': 1e-6, 'pass': max_diff < 1e-6,
        'n_arrays_checked': len(per_variant)
    }

    # 5. HR centroid orientation check
    hr_n = d['display_hr_native']
    cr, cc = centroid(hr_n)
    H, W = hr_n.shape
    r['checks']['hr_centroid'] = {
        'shape': [H, W],
        'centroid_row': round(cr,3), 'centroid_col': round(cc,3),
        'center_row': H/2-0.5, 'center_col': W/2-0.5,
        'note': 'ygh: ellipse hot bottom expected (centroid row > center_row); user: ring centered'
    }
    # hot region position (top quartile mean row)
    thresh = np.percentile(hr_n, 90)
    hot = hr_n >= thresh
    r['checks']['hr_hot_centroid'] = {
        'row_mean_of_hot': float(hot.any() and np.nonzero(hot)[0].mean()),
        'col_mean_of_hot': float(hot.any() and np.nonzero(hot)[1].mean()),
    }

    # 6. ZIP listing
    zname = 'waterfilm_ygh_sr_results_v6.zip' if name=='ygh' else 'waterfilm_user_sr_results_v6.zip'
    zpath = out/zname
    with zipfile.ZipFile(str(zpath)) as zf:
        names = sorted(zf.namelist())
    non_v6 = [n for n in names if '_v6' not in n]
    r['zip'] = {'path': str(zpath), 'size_kb': round(zpath.stat().st_size/1024,1),
                'file_list': names, 'non_v6_files': non_v6,
                'pass': len(non_v6)==0}
    report[name] = r

out_json = Path(r'D:\LMC\projects\Image-Super-Resolution-via-Iterative-Refinement\scripts\v6_acceptance_report.json')
json.dump(report, open(out_json,'w',encoding='utf-8'), indent=2, ensure_ascii=False, default=str)

# Print summary
for name, r in report.items():
    print('='*60); print(name.upper()); print('='*60)
    for ck, cv in r['checks'].items():
        if isinstance(cv, dict) and 'pass' in cv:
            print(f"  [{'PASS' if cv['pass'] else 'FAIL'}] {ck}: {cv}")
        else:
            print(f"        {ck}: {cv}")
    print(f"  ZIP: {r['zip']['size_kb']} KB, {len(r['zip']['file_list'])} files, pass={r['zip']['pass']}")
print('\nReport saved to', out_json)
