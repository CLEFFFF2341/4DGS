"""Audit original-core checkpoints against the frozen exposure baselines."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from experiments.n1b_v2.real_dynamic import DynamicData,predict
from experiments.n1b_v2.real_fastergs import midpoint


@torch.no_grad()
def main():
    out=Path('experiments/n1b_v2/results/fastergs_core');out.mkdir(parents=True,exist_ok=True)
    runs=[]
    for p in sorted(Path('output').glob('fastergs_core_*/results.json')):
        r=json.loads(p.read_text())
        if set(r['checkpoints'])=={'800','6400'}:runs.append((p,r))
    assert len(runs)==9,len(runs)
    perceptual=LearnedPerceptualImagePatchSimilarity(net_type='vgg',normalize=True).cuda().eval();rows=[]
    for p,r in runs:
        previous=json.loads((Path(r['source_run'])/'results.json').read_text());args=previous['args']
        d=DynamicData(args['root'],args['start'],args['count'],args['width'])
        assert r['shared_initial_sha256']==hashlib.sha256((Path(r['source_run'])/'shared_initial.pt').read_bytes()).hexdigest()
        item={'sequence':d.root.name,'start':d.start,'seed':r['seed'],'run':str(p.parent.resolve()),'conditions':{}}
        for key,filename,metrics in [('original_800','step800.pt',r['checkpoints']['800']),('original_6400','step6400.pt',r['checkpoints']['6400']),('midpoint_matched','midpoint_matched.pt',r['midpoint_matched_control'])]:
            raw=torch.load(p.parent/filename,weights_only=True);lp_mid=[];lp_exp=[];quad=[]
            for s,i in d.test:
                mid=midpoint(raw,d,s,i);exp,_=predict(raw,d,s,i,32,'gauss');fine,_=predict(raw,d,s,i,64,'gauss')
                lp_mid.append(float(perceptual(mid[None].clamp(0,1),d.images[s][i][None])));perceptual.reset()
                lp_exp.append(float(perceptual(exp[None].clamp(0,1),d.images[s][i][None])));perceptual.reset()
                quad.append(float((exp-fine).square().mean()))
            item['conditions'][key]={'midpoint':metrics['midpoint'],'exposure_gauss32':metrics['exposure_gauss32'],
                                    'lpips_midpoint':float(np.mean(lp_mid)),'lpips_exposure':float(np.mean(lp_exp)),
                                    'quadrature_mse_32_vs_64':float(np.mean(quad)),
                                    'checkpoint_sha256':hashlib.sha256((p.parent/filename).read_bytes()).hexdigest()}
        rows.append(item);(out/(p.parent.name+'.json')).write_text(json.dumps(r,indent=2));print('Audited',p.parent.name,flush=True)
    aggregate={}
    for key in ['original_800','original_6400','midpoint_matched']:
        aggregate[key]={}
        for mode in ['midpoint','exposure_gauss32']:
            aggregate[key][mode]={name:float(np.mean([r['conditions'][key][mode][name] for r in rows])) for name in ['mean_psnr','mean_ssim','mean_roi_psnr']}
        aggregate[key]['lpips_midpoint']=float(np.mean([r['conditions'][key]['lpips_midpoint'] for r in rows]))
        aggregate[key]['lpips_exposure']=float(np.mean([r['conditions'][key]['lpips_exposure'] for r in rows]))
    environment={'torch':torch.__version__,'cuda':torch.version.cuda,'source_sha256':{n:hashlib.sha256((Path('experiments/n1b_v2')/n).read_bytes()).hexdigest() for n in ['real_fastergs.py','real_fastergs_audit.py','real_dynamic.py']},
                 'full_trainer_status':'blocked at MortonEncoding._C import by Windows Application Control; no policy changes or blocked-extension substitution',
                 'adaptation':'fixed-capacity original native rasterizer, FusedAdam and unmodified original loss source'}
    (out/'audit.json').write_text(json.dumps({'aggregate':aggregate,'environment':environment,'runs':rows},indent=2));print(json.dumps(aggregate),flush=True)
    chosen=[(p,r) for p,r in runs if r['seed']==0]
    canvas=Image.new('RGB',(360*4,36+len(chosen)*300),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for j,title in enumerate(['Real target','Previous modified N1b / 800','FasterGS core / 800','FasterGS core / 6400']):draw.text((j*360+5,8),title,fill='black',font=font)
    for row,(p,r) in enumerate(chosen):
        prior=json.loads((Path(r['source_run'])/'results.json').read_text());frame=prior['dataset']['test_frame_indices'][3];y=36+row*300
        draw.text((5,y+5),f"{r['dataset']['sequence']} / {frame} / seed 0",fill='black',font=font)
        target=Image.open(Path(prior['args']['root'])/'left'/f'{frame:06d}.png').resize((360,270),Image.Resampling.LANCZOS)
        images=[target,Image.open(Path(r['source_run'])/'modified'/f'0_{frame:06d}.png'),Image.open(p.parent/'step800_midpoint'/f'0_{frame:06d}.png'),Image.open(p.parent/'step6400_midpoint'/f'0_{frame:06d}.png')]
        for j,im in enumerate(images):canvas.paste(im,(j*360,y+30))
    canvas.save(out/'comparison.png')


if __name__=='__main__':main()
