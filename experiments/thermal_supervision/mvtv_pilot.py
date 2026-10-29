"""Real MVTV development pilot using explicitly approximate author alignment.

This deliberately does not masquerade as the strict linear_sensor manifest.
COLMAP calibration is transductive; JPEG IR is processed intensity, not Kelvin
or radiance. RGB PNG is used as distributed, with a training-only fixed scale.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch

from .core import Frame, GateConfig, sensor_loss
from .runtime import Camera, ExposureSample, Observation, ThermalModel
from .profiled import ProfileConfig, ProfiledTransferEngine


def unpack(stream, fmt):
    return struct.unpack('<'+fmt, stream.read(struct.calcsize('<'+fmt)))


def read_colmap(root):
    cameras, images, points = {}, {}, []
    with (root/'cameras.bin').open('rb') as stream:
        for _ in range(unpack(stream, 'Q')[0]):
            key, model, width, height = unpack(stream, 'iiQQ')
            counts = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}
            if model not in counts:
                raise ValueError(f'unsupported COLMAP camera model {model}')
            cameras[key] = (model, width, height, unpack(stream, 'd'*counts[model]))
    with (root/'images.bin').open('rb') as stream:
        for _ in range(unpack(stream, 'Q')[0]):
            key, *values = unpack(stream, 'i7di')
            name = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError('truncated COLMAP filename')
                if char == b'\0':
                    break
                name.extend(char)
            count = unpack(stream, 'Q')[0]
            xy = np.frombuffer(stream.read(count*24), dtype=[('x','<f8'),('y','<f8'),('id','<i8')]).copy()
            w, x, y, z = values[:4]
            rotation = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])
            matrix = np.eye(4)
            matrix[:3,:3], matrix[:3,3] = rotation, values[4:7]
            images[key] = {'name': name.decode(), 'w2c': matrix, 'camera': values[7], 'xy': xy}
    with (root/'points3D.bin').open('rb') as stream:
        for _ in range(unpack(stream, 'Q')[0]):
            key, x, y, z, r, g, b, error = unpack(stream, 'QdddBBBd')
            count = unpack(stream, 'Q')[0]
            tracks = np.frombuffer(stream.read(count*8), dtype='<i4').reshape(-1,2).copy()
            points.append((key, np.array([x,y,z]), error, tracks))
    return cameras, images, points


def camera_parameters(spec):
    model, width, height, p = spec
    if model == 4:
        fx, fy, cx, cy, k1, k2, p1, p2 = p
    elif model == 1:
        fx, fy, cx, cy = p
        k1 = k2 = p1 = p2 = 0.
    else:
        fx, cx, cy = p[:3]
        fy = fx
        k1 = p[3] if model in (2,3) else 0.
        k2 = p[4] if model == 3 else 0.
        p1 = p2 = 0.
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]), np.array([k1,k2,p1,p2,0.]), int(width), int(height)


def prepare(scene, width, max_points, seed):
    cameras, images, points = read_colmap(scene/'sparse/0')
    train = {Path(n).stem for n in (scene/'train_filenames.txt').read_text().split()}
    test = {Path(n).stem for n in (scene/'eval_filenames.txt').read_text().split()}
    if train & test or not test or len(train) < 3:
        raise ValueError('invalid author split')
    by_stem = {Path(value['name']).stem: (key,value) for key,value in images.items()}
    if not (train|test) <= by_stem.keys():
        raise ValueError('split contains unregistered camera')
    raw_images = {}
    for stem in sorted(train):
        array = cv2.imread(str(scene/'images'/f'{stem}.png'), cv2.IMREAD_UNCHANGED)
        if array is None or array.dtype != np.uint16 or array.ndim != 3:
            raise ValueError('expected distributed uint16 RGB PNG')
        raw_images[stem] = array[:,:,::-1].astype(np.float32)/65535.
    # Exposure normalization is fixed from training inputs, never long/test GT.
    rgb_scale = float(np.quantile(np.concatenate([a[::8,::8].ravel() for a in raw_images.values()]), .99))
    if not math.isfinite(rgb_scale) or rgb_scale <= 0:
        raise ValueError('invalid training exposure scale')
    centers = np.stack([np.linalg.inv(by_stem[s][1]['w2c'])[:3,3] for s in sorted(train)])
    origin = centers.mean(0)
    extent = float(np.linalg.norm(centers-origin, axis=1).max())
    if extent <= 0:
        raise ValueError('no camera baseline')
    train_ids = {by_stem[s][0] for s in train}
    xyz, colors, point_ids = [], [], []
    for key, position, error, tracks in points:
        usable = [(int(i),int(j)) for i,j in tracks if i in train_ids]
        if error > 3 or len(usable) < 2:
            continue
        samples = []
        for image_id, index in usable:
            img = images[image_id]
            coords = img['xy'][index]
            array = raw_images[Path(img['name']).stem]
            u, v = int(round(float(coords['x']))), int(round(float(coords['y'])))
            if 0 <= v < array.shape[0] and 0 <= u < array.shape[1]:
                samples.append(array[v,u]/rgb_scale)
        if samples:
            xyz.append((position-origin)/extent)
            colors.append(np.median(samples, axis=0).clip(0,1))
            point_ids.append(key)
    if len(xyz) < 100:
        raise ValueError('insufficient training-supported initialization')
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(xyz), min(max_points,len(xyz)), replace=False))
    xyz, colors = np.asarray(xyz)[chosen], np.asarray(colors)[chosen]
    distances, _ = cKDTree(xyz).query(xyz, k=4)
    scale = np.sqrt(np.mean(distances[:,1:]**2,axis=1)).clip(.001,.1)
    n = len(xyz)
    q = np.zeros((n,4),np.float32)
    q[:,0] = 1
    arrays = {'spatial_means': xyz, 'temporal_means': np.full((n,1),.5),
        'spatial_scales': np.log(scale[:,None].repeat(3,1)), 'temporal_scales': np.full((n,1),math.log(100.)),
        'left_isoclinic_rotations': q, 'right_isoclinic_rotations': q,
        'opacities': np.full((n,1),-1.386294361),
        'sh_coefficients_0': ((colors-.5)/.28209479177387814)[:,None,:],
        'sh_coefficients_rest': np.zeros((n,15,3))}
    model = ThermalModel({k: torch.tensor(v,dtype=torch.float32,device='cuda') for k,v in arrays.items()})
    observations = {'train': [], 'test': [], 'warmup': []}
    for split, stems in [('train',train),('test',test)]:
        for stem in sorted(stems):
            _, item = by_stem[stem]
            K, distortion, original_w, original_h = camera_parameters(cameras[item['camera']])
            if stem in raw_images:
                rgb = raw_images[stem]/rgb_scale
            else:
                rgb = cv2.imread(str(scene/'images'/f'{stem}.png'),-1)[:,:,::-1].astype(np.float32)/(65535*rgb_scale)
            thermal = cv2.imread(str(scene/'thermals'/f'{stem}.jpg'),cv2.IMREAD_GRAYSCALE)
            if rgb.shape[:2] != (original_h,original_w) or thermal is None or thermal.shape != rgb.shape[:2]:
                raise ValueError('COLMAP/image/thermal dimensions disagree')
            target_w = min(width,original_w)
            target_h = round(original_h*target_w/original_w)
            valid = cv2.undistort(np.ones((original_h,original_w),np.float32),K,distortion)
            def rectify(array):
                return cv2.resize(cv2.undistort(array,K,distortion),(target_w,target_h),interpolation=cv2.INTER_AREA)
            # Ordinary RGB fitting retains dark and bright pixels. Only invalid
            # rectification and near-digital-white source pixels are excluded.
            unsaturated = rectify((rgb.max(2)*rgb_scale < .999).astype(np.float32)) > .999
            rgb, ir = rectify(rgb), rectify(thermal.astype(np.float32)/255.)
            valid = cv2.resize(valid,(target_w,target_h),interpolation=cv2.INTER_AREA) > .999
            w2c = item['w2c'].copy()
            w2c[:3,3] = (w2c[:3,:3]@origin+w2c[:3,3])/extent
            camera = Camera(torch.tensor(w2c,dtype=torch.float32,device='cuda'),target_w,target_h,
                K[0,0]*target_w/original_w,K[1,1]*target_h/original_h,K[0,2]*target_w/original_w,K[1,2]*target_h/original_h,.001,100.)
            pair = []
            for sensor, array, noise in [('rgb',rgb,.05),('thermal',ir[:,:,None],.04)]:
                weight = valid.copy()
                if sensor == 'rgb':
                    weight &= (array.mean(2) > .01) & (array.max(2) < 2.)
                else:
                    weight &= (array[:,:,0] > 1/255.) & (array[:,:,0] < 254/255.)
                obs = Observation(Frame(stem+'_'+sensor,stem,0.,split,True), sensor,
                    (ExposureSample(camera,.5,1.),),torch.tensor(array.transpose(2,0,1).copy(),device='cuda'),
                    torch.tensor(weight[None].astype(np.float32),device='cuda'),torch.tensor(noise,device='cuda'))
                obs.validate()
                pair.append(obs)
                if split == 'train' and sensor == 'rgb':
                    observations['warmup'].append(replace(obs,weight=torch.tensor(
                        (valid & unsaturated)[None].astype(np.float32),device='cuda')))
            observations[split].append(pair)
    audit = {'dataset':str(scene.resolve()),'protocol':'author_homography_proxy_development_pilot',
        'rgb_units':'distributed uint16 RAW-derived PNG / 65535 / train-only p99; unverified black level',
        'thermal_units':'processed JPEG intensity / 255, NOT calibrated radiance',
        'projection':'author aligned thermal uses RGB camera proxy; common OPENCV undistortion',
        'frame_calibrated_meaning':'audited finite author camera proxy, NOT independent dual-sensor calibration',
        'exposure':'static midpoint approximation, no motion/deblurring claim',
        'noise':'fixed heuristic robust-loss scales, not sensor noise calibration',
        'rgb_fit_mask':'undistortion valid and original RAW <0.999 digital white; includes dark/bright pixels',
        'rgb_guard_mask':'undistortion valid, scaled RAW mean>0.01 and max<2; heuristic reliable range only',
        'sfm':'author all-view COLMAP (transductive); point colors sampled from training PNG only',
        'train':sorted(train),'test':sorted(test),'rgb_scale':rgb_scale,'origin':origin.tolist(),'extent':extent,
        'points':n,'selected_point_ids':[int(point_ids[i]) for i in chosen],
        'metadata_sha256':{str(p.relative_to(scene)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [scene/'transforms.json',scene/'train_filenames.txt',scene/'eval_filenames.txt',*sorted((scene/'sparse/0').glob('*.bin'))]}}
    return model, observations, audit


@torch.no_grad()
def evaluate(model, pairs, rgb_scale):
    # Reporting only; never used to select training steps or hyperparameters.
    values = []
    for rgb, _ in pairs:
        pred = model.render(rgb)
        mse = float((pred-rgb.target).square().mean())
        masked_mse = float(((pred-rgb.target).square()*rgb.weight).sum()/(3*rgb.weight.sum()))
        values.append({'view':rgb.frame.view_id,'mse_scaled_raw':mse,
                       'psnr_scaled_raw':-10*math.log10(max(mse,1e-15)),
                       'mse_distributed_raw_01':mse*rgb_scale**2,
                       'mse_fixed_reliable_mask_scaled_raw':masked_mse,
                       'reliable_fraction':float((rgb.weight>0).float().mean())})
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--allow-author-proxy', action='store_true', required=True)
    parser.add_argument('--rgb-steps',type=int,default=3000)
    parser.add_argument('--thermal-steps',type=int,default=500)
    parser.add_argument('--transactions',type=int,default=100)
    parser.add_argument('--width',type=int,default=284)
    parser.add_argument('--points',type=int,default=12000)
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    if min(args.rgb_steps,args.thermal_steps,args.transactions,args.width,args.points) < 1:
        raise ValueError('positive budgets required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    def log(row):
        row['seconds'] = time.perf_counter()-started
        row['cuda_allocated_mb'] = torch.cuda.memory_allocated()/2**20
        with (out/'progress.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    model, obs, audit = prepare(Path(args.scene),args.width,args.points,args.seed)
    audit['arguments'] = vars(args)
    audit['source_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    train = obs['train']
    generator = torch.Generator().manual_seed(args.seed)
    rgb_names = {'spatial_means':.0005,'spatial_scales':.002,'opacities':.01,'sh_coefficients_0':.0025}
    for name, parameter in model.raw.items():
        parameter.requires_grad_(name in rgb_names)
    optimizer = torch.optim.Adam([{'params':[model.raw[k]],'lr':lr,'name':k} for k,lr in rgb_names.items()])
    log({'stage':'initialized','train_views':len(train),'test_views':len(obs['test']),'points':len(model.point_ids)})
    for step in range(1,args.rgb_steps+1):
        rgb = obs['warmup'][int(torch.randint(len(train),(),generator=generator))]
        optimizer.zero_grad(set_to_none=True)
        loss = sensor_loss(model.render(rgb,train_raw=True),rgb.target,rgb.weight,rgb.noise)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.raw['spatial_scales'].clamp_(math.log(.0005),math.log(.2))
            model.raw['opacities'].clamp_(-8,8)
        if step == 1 or step % 100 == 0:
            log({'stage':'rgb_warmup','step':step,'loss':float(loss.detach())})
        if step % 1000 == 0 or step == args.rgb_steps:
            model.save(out/'rgb_checkpoint.pt',optimizers={'rgb':optimizer},extra={'stage':'rgb_warmup','step':step})
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    with torch.no_grad():
        model.reference_scale.copy_(model.raw['spatial_scales'].exp())
    baseline = evaluate(model,obs['test'],audit['rgb_scale'])
    (out/'baseline_eval.json').write_text(json.dumps(baseline,indent=2))
    optimizer = torch.optim.Adam([model.thermal],lr=.02)
    for step in range(1,args.thermal_steps+1):
        thermal = train[int(torch.randint(len(train),(),generator=generator))][1]
        optimizer.zero_grad(set_to_none=True)
        loss = sensor_loss(model.render(thermal,thermal=model.thermal),thermal.target,thermal.weight,thermal.noise)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0:
            log({'stage':'thermal_head_fit','step':step,'loss':float(loss.detach())})
    optimizer.zero_grad(set_to_none=True)
    del optimizer  # no stale thermal Adam state survives functional transactions
    model.save(out/'prefit_checkpoint.pt',extra={'stage':'thermal_prefit'})
    engine = ProfiledTransferEngine(model,ProfileConfig(gate=GateConfig(max_rgb_loss_increase=0.,max_rgb_anchor_increase=0.)))
    # Establish anchors before any transfer, including guards not seen first.
    with torch.no_grad():
        for rgb,_ in train:
            engine.anchors[rgb.frame.view_id] = float(sensor_loss(model.render(rgb),rgb.target,rgb.weight,rgb.noise))
    accepted = 0
    for step in range(args.transactions):
        index = step % len(train)
        rgb, source = train[index]
        source_center = torch.linalg.inv(source.samples[0].camera.w2c)[:3,3]
        candidates = []
        for j, (_, other) in enumerate(train):
            if j != index:
                distance = float((torch.linalg.inv(other.samples[0].camera.w2c)[:3,3]-source_center).norm())
                if distance >= float(model.reference_scale.median())*.5:
                    candidates.append((distance,j))
        if not candidates:
            raise ValueError('no physical thermal witness baseline')
        _, j = min(candidates)
        guard, witness = train[j]
        record = engine.step(source,[witness],[rgb,guard])
        accepted += int(record['accepted'])
        log({'stage':'thermal_refine','step':step+1,**record})
        if (step+1) % 10 == 0 or step+1 == args.transactions:
            model.save(out/'refined_checkpoint.pt',extra={'stage':'thermal_refine','step':step+1,'engine':engine.state_dict()})
    final = evaluate(model,obs['test'],audit['rgb_scale'])
    (out/'final_eval.json').write_text(json.dumps(final,indent=2))
    log({'stage':'complete','accepted':accepted,'transactions':args.transactions,'baseline':baseline,'final':final})


if __name__ == '__main__':
    main()
