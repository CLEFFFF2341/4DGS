"""Native prefix checkpoints and resumed matched branches."""
import json
import time
import torch
from Methods.FasterGS4D.Trainer import FasterGS4DTrainer
from Methods.Base.utils import training_callback, post_training_callback
from Optim.Samplers.DatasetSamplers import DatasetSampler
from experiments.n1b_native.native import raw_state, tensor_hash


class Trainer(FasterGS4DTrainer):
    def __init__(self, *, prefix=False, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix
        self.rows = []
        self.started = time.perf_counter()

    @training_callback(priority=79)
    def update_residuals(self, *_):
        optimizer = self.model.gaussians.extra_optimizer
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    @training_callback(priority=2, iteration_stride=1000)
    def progress(self, iteration, _):
        torch.cuda.synchronize()
        row = {'step': iteration+1, 'gaussians': len(self.model.gaussians.spatial_means),
               'seconds': time.perf_counter()-self.started,
               'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
               'requires_empty_cache': self.requires_empty_cache}
        self.rows.append(row)
        with (self.output_directory/'progress.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def finalize(self, *args):
        if self.prefix:
            g = self.model.gaussians
            state = {'raw': raw_state(self.model), 'optimizer': g.optimizer.state_dict(),
                'steps': self.model.num_iterations_trained,
                'active_sh_degree': g.active_sh_degree, 'extent': g.training_cameras_extent,
                'sampler_indices': self.train_sampler.id_sampler.indices.cpu(),
                'sampler_cursor': self.train_sampler.id_sampler.current_id,
                'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state(),
                'densification_info': None if g.densification_info is None else g.densification_info.cpu()}
            torch.save(state, self.output_directory/'prefix.pt')
        else:
            super().finalize(*args)
            self.model.gaussians.extra_optimizer = None


def resume(trainer, dataset, checkpoint, kind, accel_lr=.01, color_lr=.001):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model, g = trainer.model, trainer.model.gaussians
    assert saved['steps'] >= trainer.DENSIFICATION_END_ITERATION
    for key, value in saved['raw'].items():
        setattr(g, '_'+key, torch.nn.Parameter(value.cuda()))
    g.active_sh_degree = saved['active_sh_degree']
    g.active_sh_bases = (g.active_sh_degree+1)**2
    g.training_setup(trainer, saved['extent'])
    g.optimizer.load_state_dict(saved['optimizer'])
    for state in g.optimizer.state.values():
        state['step'] = int(state['step'])
    g._densification_info = None if saved['densification_info'] is None else saved['densification_info'].cuda()
    model.num_iterations_trained = saved['steps']
    trainer.train_sampler = DatasetSampler(dataset.train(), random=True)
    trainer.train_sampler.id_sampler.indices = saved['sampler_indices'].cuda()
    trainer.train_sampler.id_sampler.current_id = saved['sampler_cursor']
    g.activate(kind, accel_lr, color_lr)
    torch.set_rng_state(saved['rng_cpu'])
    torch.cuda.set_rng_state(saved['rng_cuda'])
    return {'initial_tensor_sha256': tensor_hash(saved['raw']),
            'prefix_steps': saved['steps'], 'extent': saved['extent'],
            'extra_parameters': sum(p.numel() for p in g.residuals.values())}


assert Trainer.training_iteration is FasterGS4DTrainer.training_iteration
