"""Summarize joint-repair records without selecting or discarding failures."""
import argparse
import json
from pathlib import Path
import statistics


def mean(values):
    return statistics.mean(values)


def repaired(row):
    return row['coordinates'] == 'scale-aware' and row['temporal_lr'] == .05


def summarize(directory):
    source = json.loads((directory / 'confirmation.json').read_text(encoding='utf-8'))
    if sorted(case['seed'] for case in source['cases']) != list(range(6, 12)):
        raise ValueError('confirmation must include all six frozen seeds6-11')
    cases = []
    for case in source['cases']:
        if len(case['rows']) != len(source['arms']):
            raise ValueError(f"incomplete confirmation seed {case['seed']}")
        rows = case['rows']
        ours = next(r for r in rows if r['method'] == 'partition' and repaired(r))
        legacy = next(r for r in rows if r['coordinates'] == 'legacy')
        equal = next(r for r in rows if r['method'] == 'partition' and r['temporal_lr'] == .01 and r['coordinates'] == 'scale-aware')
        simple = [r for r in rows if r['method'] == 'gauss']
        best = min(simple, key=lambda r: mean(r['roi_check_mse']))
        cases.append(dict(seed=case['seed'], sigma=case['sigma'],
            initial_position_mse=case['initial']['position_mse'], legacy_position_mse=legacy['position_mse'],
            repaired_position_mse=ours['position_mse'],
            legacy_roi_check_mse=mean(legacy['roi_check_mse']),
            repaired_roi_check_mse=mean(ours['roi_check_mse']),
            equal_rates_roi_check_mse=mean(equal['roi_check_mse']),
            best_gauss_calls=best['calls'], best_gauss_roi_check_mse=mean(best['roi_check_mse']),
            roi_reduction_vs_legacy=mean(legacy['roi_check_mse']) / mean(ours['roi_check_mse']),
            roi_reduction_vs_best_gauss=mean(best['roi_check_mse']) / mean(ours['roi_check_mse']),
            full_reduction_vs_best_gauss=min(mean(r['reference_check_mse']) for r in simple) / mean(ours['reference_check_mse']),
            position_reduction_vs_legacy=legacy['position_mse'] / ours['position_mse'],
            position_ratio_to_initial=ours['position_mse'] / case['initial']['position_mse'],
            repaired_log_scale_mse=ours['log_scale_mse'],
            repaired_steps=ours['accepted_steps'], legacy_steps=legacy['accepted_steps'],
            repaired_elapsed=ours['elapsed'], last_accepted_seconds=ours['last_accepted_seconds']))
    out = dict(confirmation=cases, aggregates={})
    for key in ('roi_reduction_vs_legacy', 'roi_reduction_vs_best_gauss', 'full_reduction_vs_best_gauss',
                'position_reduction_vs_legacy', 'position_ratio_to_initial'):
        values = [case[key] for case in cases]
        out['aggregates'][key] = dict(min=min(values), median=statistics.median(values), max=max(values))
    out['image_wins_vs_best_gauss'] = sum(c['roi_reduction_vs_best_gauss'] > 1 for c in cases)
    out['geometry_improvements_vs_initial'] = sum(c['position_ratio_to_initial'] < 1 for c in cases)
    for name in ('same_roi', 'repeats', 'stress_xyz', 'equal_steps'):
        path = directory / f'{name}.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        results = []
        for case in data['cases']:
            for row in case['rows']:
                results.append(dict(seed=case['seed'], repeat=row.get('repeat', 0),
                    method=row['method'], calls=row['calls'], coordinates=row['coordinates'],
                    temporal_lr=row['temporal_lr'], roi_check_mse=mean(row['roi_check_mse']),
                    full_check_mse=mean(row['reference_check_mse']), position_mse=row['position_mse'],
                    position_ratio_to_initial=row['position_mse']/case['initial']['position_mse'],
                    log_scale_mse=row['log_scale_mse'], steps=row['accepted_steps'], elapsed=row['elapsed']))
        out[name] = results
    sequence_path = directory / 'sequence_confirmation.json'
    if sequence_path.exists():
        sequence = json.loads(sequence_path.read_text(encoding='utf-8'))
        if sorted(c['seed'] for c in sequence['cases']) != [14, 15, 16, 17]:
            raise ValueError('incomplete sequence confirmation')
        out['sequence_confirmation'] = []
        for case in sequence['cases']:
            rows = {r['arm']: r for r in case['rows']}
            long = rows['long-partition32']; part = rows['sequence-partition8']; gauss = rows['sequence-gauss8']
            out['sequence_confirmation'].append(dict(seed=case['seed'],
                initial_position_mse=case['initial']['position_mse'],
                long_sharp_mse=long['sharp_check_mse'], partition_sharp_mse=part['sharp_check_mse'],
                gauss_sharp_mse=gauss['sharp_check_mse'],
                long_position_mse=long['position_mse'], partition_position_mse=part['position_mse'],
                gauss_position_mse=gauss['position_mse'],
                sharp_reduction_long_to_sequence=long['sharp_check_mse']/part['sharp_check_mse'],
                position_reduction_long_to_sequence=long['position_mse']/part['position_mse'],
                position_reduction_initial_to_sequence=case['initial']['position_mse']/part['position_mse'],
                exposure_reduction_long_to_sequence=long['exposure_check_mse']/part['exposure_check_mse'],
                gauss_better_than_partition=gauss['sharp_check_mse'] < part['sharp_check_mse']))
        out['sequence_caveat'] = 'Additional known-exposure observations; not an equal-data integrator advantage.'
    destination = directory / 'joint_summary.json'
    destination.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path(__file__).parent / 'results' / 'joint_audit')
    summarize(parser.parse_args().directory)
