import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def _safe_mean(vals):
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


def _summarize_rows(rows, group_key):
    metric_keys = [k for k in rows[0].keys() if k not in {'index', 'patient_id', 'slice_idx', 'met_id', 'lowres'}]
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    out = {}
    for key, items in grouped.items():
        summary = {'count': len(items)}
        for metric_key in metric_keys:
            vals = [float(item[metric_key]) for item in items if item[metric_key] not in ('', 'None', None)]
            summary[metric_key] = _safe_mean(vals)
        out[key] = summary
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to metrics.csv or result directory that contains metrics.csv.')
    args = parser.parse_args()

    csv_path = args.path
    if os.path.isdir(csv_path):
        csv_path = os.path.join(csv_path, 'metrics.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'metrics.csv not found: {csv_path}')

    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError('metrics.csv is empty.')

    metric_keys = [k for k in rows[0].keys() if k not in {'index', 'patient_id', 'slice_idx', 'met_id', 'lowres'}]
    overall = {'count': len(rows)}
    for metric_key in metric_keys:
        vals = [float(item[metric_key]) for item in rows if item[metric_key] not in ('', 'None', None)]
        overall[metric_key] = _safe_mean(vals)

    summary = {
        'overall': overall,
        'by_patient_id': _summarize_rows(rows, 'patient_id'),
        'by_met_id': _summarize_rows(rows, 'met_id'),
        'by_lowres': _summarize_rows(rows, 'lowres'),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
