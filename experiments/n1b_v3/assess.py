"""Apply the predeclared confirmation gate and create fixed-view comparisons."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

BASE='native:1:0';CAND='affine_spline:1:0.00003'


def main():
    root=Path('output/n1b_v3');dest=Path('experiments/n1b_v3/results')
    audits=json.loads((dest/'confirmation01_audit.json').read_text())['rows']
    byrun={Path(r['run']).name:r for r in audits};rows=[]
    for path in sorted(root.glob('confirm01_*/results.json')):
        r=json.loads(path.read_text());n=r['arms'][BASE];c=r['arms'][CAND];audit=byrun[path.parent.name]
        row={'run':path.parent.name,'sequence':r['dataset']['sequence'],'start':r['previous_args']['start'],
             'seed':r['previous_args']['seed'],'budget':r['budget'],'baseline':{},'candidate':{},'delta':{}}
        for label,arm,item in [('baseline',BASE,n),('candidate',CAND,c)]:
            row[label]={k:item['metrics'][k] for k in ('psnr','ssim','roi_psnr')}
            row[label].update({'lpips':audit['arms'][arm]['lpips'],'seconds':item['training_seconds'],
                               'steps':item['steps'],'renders':item['renders'],'parameters':item['parameter_count'],
                               'quadrature_mse':audit['arms'][arm]['quadrature_mse_32_vs_64'],
                               'checkpoint_hash_ok':audit['arms'][arm]['hash_matches_training_record']})
        for metric in ('psnr','ssim','roi_psnr','lpips'):
            row['delta'][metric]=row['candidate'][metric]-row['baseline'][metric]
        rows.append(row)
    summary={};complete=len(rows)==18
    for budget in ('renders','time'):
        group=[r for r in rows if r['budget']==budget]
        summary[budget]={'pairs':len(group),'positive_psnr':sum(r['delta']['psnr']>0 for r in group),
                         'mean_delta':{k:float(np.mean([r['delta'][k] for r in group])) for k in ('psnr','ssim','roi_psnr','lpips')},
                         'min_delta_psnr':min(r['delta']['psnr'] for r in group),
                         'max_delta_psnr':max(r['delta']['psnr'] for r in group),'clips':{}}
        for sequence in sorted(set(r['sequence'] for r in group)):
            clip=[r for r in group if r['sequence']==sequence]
            summary[budget]['clips'][sequence]={
                label:{k:float(np.mean([r[label][k] for r in clip])) for k in ('psnr','ssim','lpips','roi_psnr')}
                for label in ('baseline','candidate','delta')}
    renders=summary['renders'];time=summary['time']
    gate={'complete':complete,'mean_gain_at_least_0_2':renders['mean_delta']['psnr']>=.2,
          'at_least_8_positive_pairs':renders['positive_psnr']>=8,
          'all_clip_mean_psnr_positive':all(v['delta']['psnr']>0 for v in renders['clips'].values()),
          'all_clip_mean_ssim_nondecreasing':all(v['delta']['ssim']>=0 for v in renders['clips'].values()),
          'all_clip_mean_lpips_nonincreasing':all(v['delta']['lpips']<=0 for v in renders['clips'].values()),
          'time_all_clip_mean_psnr_positive':all(v['delta']['psnr']>0 for v in time['clips'].values()),
          'time_overall_psnr_positive':time['mean_delta']['psnr']>0,
          'all_checkpoint_hashes_match':all(r[label]['checkpoint_hash_ok'] for r in rows for label in ('baseline','candidate'))}
    result={'summary':summary,'gate':gate,'gate_passed':all(gate.values()),'rows':rows}
    target=dest/'confirmation01_assessment.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2));print(json.dumps({'summary':summary,'gate':gate,'passed':all(gate.values())}),flush=True)
    # Predetermined fourth held-out frame, left camera, seed3, all clips.
    canvas=Image.new('RGB',(1080,942),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for j,label in enumerate(('Real recorded frame','Matched native Gauss1','Shared temporal field')):
        draw.text((j*360+8,8),label,fill='black',font=font)
    for i,short in enumerate(('c05','c08','n260')):
        folder=root/f'confirm01_{short}_s3_renders';r=json.loads((folder/'results.json').read_text())
        frame=r['dataset']['test_frame_indices'][3];y=40+i*300
        draw.text((8,y+3),f"{r['dataset']['sequence']} / frame {frame} / seed 3",fill='black',font=font)
        target=Image.open(Path(r['previous_args']['root'])/'left'/f'{frame:06d}.png').convert('RGB').resize((360,270),Image.Resampling.LANCZOS)
        images=[target]+[Image.open(folder/arm.replace(':','_')/f'0_{frame:06d}.png') for arm in (BASE,CAND)]
        for j,im in enumerate(images):canvas.paste(im,(j*360,y+30))
    canvas.save(dest/'confirmation01_comparison.png')


if __name__=='__main__':main()
