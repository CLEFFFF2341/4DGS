"""Independent LPIPS, integration-convergence and motion audit of frozen runs."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from experiments.n1b.core import rotation
from experiments.n1b_v2.real_dynamic import DynamicData,predict


def main():
    out=Path('experiments/n1b_v2/results/real_dynamic');out.mkdir(parents=True,exist_ok=True)
    runs=[]
    for p in sorted(Path('output').glob('dsec_*/results.json')):
        r=json.loads(p.read_text())
        if (r['args']['points'],r['args']['warmup'],r['args']['steps'])==(32768,1200,800):runs.append((p,r))
    assert len(runs)==9,len(runs)
    assert len({(r['dataset']['sequence'],r['args']['start'],r['args']['seed']) for _,r in runs})==9
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='vgg',normalize=True).cuda().eval()
    audits=[]
    with torch.no_grad():
        for p,r in runs:
            d=DynamicData(r['args']['root'],r['args']['start'],r['args']['count'],r['args']['width'])
            a={'sequence':d.root.name,'start':d.start,'seed':r['args']['seed'],'output':str(p.parent.resolve()),'arms':{}}
            initial=torch.load(p.parent/'shared_initial.pt',weights_only=True)
            for name in ['initial','original','modified','gauss']:
                path=p.parent/('shared_initial.pt' if name=='initial' else name+'.pt');raw=torch.load(path,weights_only=True)
                scores=[];quad=[];roi_quad=[];instant=[]
                for s,i in d.test:
                    full,_=predict(raw,d,s,i,32,'gauss');coarse,_=predict(raw,d,s,i,16,'gauss')
                    fine,_=predict(raw,d,s,i,64,'gauss')
                    score=lpips(full.clamp(0,1)[None],d.images[s][i][None]);scores.append(float(score));lpips.reset()
                    quad.append(float((full-fine).square().mean()));roi_quad.append(float((full[:,d.roi]-fine[:,d.roi]).square().mean()))
                R=rotation(raw['left_isoclinic_rotations'],raw['right_isoclinic_rotations'])
                var=(2*torch.cat((raw['spatial_scales'],raw['temporal_scales']),-1)).exp()
                cov=(R*var[:,None,:])@R.transpose(-1,-2)
                velocity=cov[:,:3,3]/cov[:,3:4,3]/d.span
                a['arms'][name]={'lpips_vgg':float(np.mean(scores)),'per_view_lpips':scores,
                    'quadrature_mse_32_vs_64':float(np.mean(quad)),'roi_quadrature_mse_32_vs_64':float(np.mean(roi_quad)),
                    'mean_parameter_position_displacement':float((raw['spatial_means']-initial['spatial_means']).norm(dim=-1).mean()),
                    'median_parameter_velocity_rig_units_per_second':float(velocity.norm(dim=-1).median()),
                    'median_temporal_sigma_seconds':float(cov[:,3,3].sqrt().median()*d.span),
                    'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            model_a=torch.load(p.parent/'modified.pt',weights_only=True);model_b=torch.load(p.parent/'gauss.pt',weights_only=True)
            a['max_parameter_difference_modified_gauss']={k:float((model_a[k]-model_b[k]).abs().max()) for k in model_a}
            audits.append(a)
            (out/f"{d.root.name}_f{d.start}_s{a['seed']}.json").write_text(json.dumps(r,indent=2));print('Audited',d.root.name,d.start,a['seed'],flush=True)
    aggregate={}
    for name in ['initial','original','modified','gauss']:
        metrics=[r['initial'] if name=='initial' else r['arms'][name]['metrics'] for _,r in runs]
        aggregate[name]={key:float(np.mean([m[key] for m in metrics])) for key in ['mean_psnr','mean_ssim','mean_roi_psnr']}
        aggregate[name]['lpips_vgg']=float(np.mean([a['arms'][name]['lpips_vgg'] for a in audits]))
    env={'torch':torch.__version__,'cuda':torch.version.cuda,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'source_sha256':{n:hashlib.sha256((Path('experiments/n1b_v2')/n).read_bytes()).hexdigest() for n in ['real_dynamic.py','real_dynamic_audit.py','joint_optimization.py','partition.py','real_cdavis.py','download_dsec.py']}}
    (out/'audit.json').write_text(json.dumps({'aggregate':aggregate,'environment':env,'runs':audits},indent=2));print(json.dumps(aggregate),flush=True)
    # Fixed center held-out time, left camera, seed zero for all three clips.
    chosen=[(p,r) for p,r in runs if r['args']['seed']==0]
    canvas=Image.new('RGB',(360*4,36+len(chosen)*300),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    for col,title in enumerate(['Real finite-exposure target','Original N1b settings','Modified N1b','Gauss + same optimizer']):draw.text((col*360+5,8),title,fill='black',font=font)
    for row,(p,r) in enumerate(chosen):
        frame=r['dataset']['test_frame_indices'][3];y=36+row*300;draw.text((5,y+5),f"{r['dataset']['sequence']} / {frame} / seed 0",fill='black',font=font)
        target=Image.open(Path(r['args']['root'])/'left'/f'{frame:06d}.png').resize((360,270),Image.Resampling.LANCZOS)
        canvas.paste(target,(0,y+30))
        for col,name in enumerate(['original','modified','gauss'],1):canvas.paste(Image.open(p.parent/name/f'0_{frame:06d}.png'),(col*360,y+30))
    canvas.save(out/'comparison.png')


if __name__=='__main__':main()
