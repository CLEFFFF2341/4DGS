"""Read-only aggregation of every completed v3 run; never choose checkpoints."""
import json
from pathlib import Path


def main():
    for path in sorted(Path('output/n1b_v3').glob('*/results.json')):
        result=json.loads(path.read_text())
        if 'arms' not in result:continue
        baseline=result['arms'].get('native:1:0') or result['arms'].get('native:8:0')
        if baseline is None or 'metrics' not in baseline:continue
        base=baseline['metrics']['psnr']
        print(path.parent.name, 'baseline',round(base,5))
        for name,r in result['arms'].items():
            if 'metrics' not in r:continue
            print(' ',name, 'delta',round(r['metrics']['psnr']-base,5),
                  'PSNR',round(r['metrics']['psnr'],5),'SSIM',round(r['metrics']['ssim'],6),
                  'steps',r['steps'],'seconds',round(r['training_seconds'],3))


if __name__=='__main__':main()
