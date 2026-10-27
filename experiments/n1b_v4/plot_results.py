"""Reproduce the full-test figure from the compact metrics tracked in Git."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    folder = Path(__file__).parent/'results'
    report = json.loads((folder/'full_confirmation.json').read_text())
    assert report['complete'], 'Wait for all three predeclared seeds'
    records = {r['run']:r for r in json.loads((folder/'development.json').read_text())}
    fig, ax = plt.subplots(figsize=(9,4.7))
    colors = ['#0072B2','#D55E00','#009E73']
    all_deltas = []
    for pair, color in zip(report['pairs'], colors):
        seed = pair['seed']
        before = records[f'full_s{seed}_native']['views']
        after = records[f'full_s{seed}_candidate']['views']
        assert len(before) == len(after) == 300
        assert [v['image'] for v in before] == [v['image'] for v in after]
        delta = np.array([y['psnr']-x['psnr'] for x,y in zip(before,after)])
        assert abs(delta.mean()-pair['delta']['psnr']) < 1e-9
        all_deltas.extend(delta)
        ax.plot(np.arange(300), delta, color=color, linewidth=1.4,
                label=f'Seed {seed}: mean {delta.mean():+.3f} dB')
    ax.axhline(0, color='#555555', linewidth=1., linestyle='--')
    ax.set(xlim=(0,299), ylim=(min(-.01,min(all_deltas)-.01),max(.02,max(all_deltas)+.02)),
           xlabel='Official test frame', ylabel='PSNR gain over paired native baseline (dB)',
           title='Coffee Martini: full native settings, fixed local acceleration')
    ax.grid(axis='y',alpha=.18)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,fontsize=9)
    fig.text(.5,.02,'30,000 total steps | 300 test frames | Three paired seeds; adjacent frames are correlated',
             ha='center',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.05,1,1))
    path = folder/'full_psnr_deltas.png'
    fig.savefig(path,dpi=220)
    plt.close(fig)
    print(path.resolve())

if __name__ == '__main__':
    main()
