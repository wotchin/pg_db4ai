#!/usr/bin/env python3
"""Read-only SIMD evidence audit and summary; never train or rerun a baseline.
Use the end-to-end directory of the evidence ZIP, plus the frozen data directory.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def summarize(root: Path, data: Path, out: Path) -> None:
    records = json.loads((root / 'results.json').read_text())
    modes = ('off', 'portable', 'auto')
    groups = defaultdict(list)
    checked = 0
    for record in records:
        name = f"{record['dataset']}-{record['seed']}"
        case, inputs = root / name, data / name
        meta = json.loads((inputs / 'meta.json').read_text())
        arrays = np.load(inputs / 'data.npz')
        if hashlib.sha256(np.ascontiguousarray(arrays['X_train']).tobytes()).hexdigest() != record['train_X_sha256']:
            raise ValueError(f'{name}: frozen training matrix changed')
        if hashlib.sha256(np.ascontiguousarray(arrays['train_ids']).tobytes()).hexdigest() != record['train_order_sha256']:
            raise ValueError(f'{name}: frozen row order changed')
        predictions, fixed = {}, {}
        for mode in modes:
            prefix = case / f"{record['method']}-{mode}"
            expected_ids = np.arange(record['n_test'])
            for suffix, dest in (('-predictions.csv', predictions), ('-fixed-model.csv', fixed)):
                a = np.loadtxt(str(prefix) + suffix, delimiter=',', ndmin=2)
                if not np.array_equal(a[:, 0], expected_ids) or not np.isfinite(a[:, 1]).all():
                    raise ValueError(f'{name}: invalid prediction alignment or non-finite value')
                dest[mode] = a[:, 1]
            pred, target = predictions[mode], arrays['y_test']
            if meta['task'] == 'regression':
                diff = target - (pred * meta['y_scale'] + meta['y_mean'])
                sse = np.dot(diff, diff)
                quality = dict(rmse=float(np.sqrt(sse / len(diff))), mae=float(np.abs(diff).mean()),
                               r2=float(1 - sse / np.sum((target - target.mean()) ** 2)))
            else:
                recalls, f1s = [], []
                for label in np.unique(target):
                    tp = np.count_nonzero((pred == label) & (target == label))
                    fp = np.count_nonzero((pred == label) & (target != label))
                    fn = np.count_nonzero((pred != label) & (target == label))
                    recalls.append(tp / (tp + fn))
                    f1s.append(2 * tp / (2 * tp + fp + fn))
                quality = dict(accuracy=float((pred == target).mean()),
                               balanced_accuracy=float(np.mean(recalls)), macro_f1=float(np.mean(f1s)))
            for metric, value in quality.items():
                if not np.isclose(value, record['training_quality'][mode][metric], rtol=1e-11, atol=1e-12):
                    raise ValueError(f'{name} {mode} {metric}: quality mismatch')
            for op in ('fit', 'predict'):
                times = np.asarray(record[f'{op}_{mode}_ms'])
                if len(times) != 7 or not np.isfinite(times).all() or np.any(times <= 0):
                    raise ValueError(f'{name}: invalid timing evidence')
            if record['training_metrics'][mode]['iterations'] != 200:
                raise ValueError(f'{name}: unexpected iteration budget')
            checked += 1
        for mode in modes:
            for values, key in ((predictions, 'trained_model_max_abs_diff'), (fixed, 'same_model_max_abs_diff')):
                delta = float(np.max(np.abs(values[mode] - values['off'])))
                if delta != record[key][mode]:
                    raise ValueError(f'{name}: reported prediction delta mismatch')
        groups[(record['dataset'], record['method'])].append(record)
    summary = []
    for (dataset, method), cases in sorted(groups.items()):
        row = dict(dataset=dataset, method=method, features=cases[0]['features'], splits=len(cases))
        for op in ('fit', 'predict'):
            for mode in modes:
                row[f'{op}_{mode}_median_ms'] = float(np.median([np.median(c[f'{op}_{mode}_ms']) for c in cases]))
            paired = [np.median(np.asarray(c[f'{op}_off_ms']) / c[f'{op}_auto_ms']) for c in cases]
            row[f'{op}_auto_speedup'] = float(np.median(paired))
        row['max_trained_prediction_abs_diff'] = max(c['trained_model_max_abs_diff']['auto'] for c in cases)
        row['max_same_model_abs_diff'] = max(c['same_model_max_abs_diff']['auto'] for c in cases)
        summary.append(row)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(summary)
    result = dict(cases=len(records), independently_checked_mode_results=checked,
                  summary_groups=len(summary), all_trained_outputs_equal=all(r['max_trained_prediction_abs_diff'] == 0 for r in summary),
                  same_model_max_abs_difference=max(r['max_same_model_abs_diff'] for r in summary),
                  external_baselines_executed=0)
    (out / 'independent-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path); parser.add_argument('frozen_data', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    summarize(args.results, args.frozen_data, args.output)
