"""Recompute paired summaries from saved trials, preserving rejected cases."""
import json
from pathlib import Path
import numpy as np


def main():
    directory = Path(__file__).parent/'results'
    confirmations = [json.loads((directory/'confirmation_balanced'/f'{i:03d}.json').read_text()) for i in range(20, 84)]
    summary = {'independent_seeds': [20, 83], 'integration': {}, 'temporal_recovery': []}
    for gradient in [False, True]:
        ratios, excluded, costs, reference_changes = [], [], [], []
        for case in confirmations:
            rows = [r for r in case['rows'] if r['gradient'] == gradient]
            if not all(r['within_budget'] for r in rows):
                excluded.append(case['seed']); continue
            ours = next(r for r in rows if r['method'] == 'partition')
            best = min(r['mse'] for r in rows if r['method'] != 'partition')
            ratios.append(ours['mse']/max(best, 1e-30)); costs.append(ours['seconds'])
            reference_changes.append(case['reference_change'][int(gradient)])
        ratios = np.asarray(ratios)
        rng = np.random.default_rng(923)
        bootstrap = ratios[rng.integers(len(ratios), size=(4096, len(ratios)))].mean(1)
        summary['integration']['gradient' if gradient else 'image'] = dict(eligible=len(ratios), excluded=excluded,
            mean_ratio=float(ratios.mean()), median_ratio=float(np.median(ratios)),
            max_ratio=float(ratios.max()), p95_ratio=float(np.quantile(ratios, .95)),
            bootstrap_mean_ratio_95=np.quantile(bootstrap, [.025, .975]).tolist(),
            median_seconds=float(np.median(costs)), max_seconds=max(costs),
            max_reference_mse_change=max(reference_changes), wins=int((ratios < 1).sum()))
    for seed in [3, 4, 5]:
        rows = []
        for calls in [32, 64, 128]:
            suffix = '' if calls == 32 else f'_calls{calls}'
            case = json.loads((directory/f'train_probe_{seed}{suffix}_temporal.json').read_text())
            rows += [dict(r, calls=calls) for r in case['rows']]
        ours = next(r for r in rows if r['method'] == 'partition')
        simple = [r for r in rows if r['method'] != 'partition']
        best = min(simple, key=lambda r: r['reference_check_mse'])
        best_param = min(r['log_scale_mse'] for r in simple)
        summary['temporal_recovery'].append(dict(seed=seed, best_simple=best['method'], best_calls=best['calls'],
            best_mse=best['reference_check_mse'], partition_mse=ours['reference_check_mse'],
            mse_reduction_factor=best['reference_check_mse']/ours['reference_check_mse'],
            best_log_scale_mse=best_param, partition_log_scale_mse=ours['log_scale_mse'],
            parameter_reduction_factor=best_param/ours['log_scale_mse'],
            psnr_gain=ours['check_psnr']-best['check_psnr']))
    passes = (all(r['eligible'] >= 60 and r['max_ratio'] < .8 for r in summary['integration'].values())
              and all(r['mse_reduction_factor'] >= 5 and r['parameter_reduction_factor'] >= 5
                      for r in summary['temporal_recovery']))
    summary['decision'] = ('CONFIRMED_CONDITIONAL_BENEFIT_FOR_TEMPORAL_INTEGRATION_AND_KNOWN_GEOMETRY_TEMPORAL_FIT'
                           if passes else 'NOT_CONFIRMED')
    summary['limits'] = 'Joint geometry fitting inconsistent; no real-dataset gain established; no global unbiased gradient claim.'
    (directory/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
