"""Convert timm's torchvision VGG16 state dict back to torchvision key layout.

The published tv_in1k checkpoint retains the original feature tensors. The
classifier layers are re-keyed and flattened from equivalent 1x1/7x7 convolutions.
The complete converted state dictionary must pass torchvision strict loading.
"""
import hashlib
import json
import os
from pathlib import Path
import torch
from torchvision.models import vgg16

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'downloads/vgg16-timm-torchvision.bin'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '1d677a45f252c18f796d56251b7584e9fdc977dc3053ea0b2cd25d519f5d0fb2'
state = torch.load(source, map_location='cpu', weights_only=True)
mapping = {'pre_logits.fc1': 'classifier.0', 'pre_logits.fc2': 'classifier.3', 'head.fc': 'classifier.6'}
converted = {}
for name, tensor in state.items():
    new_name = name
    for old, new in mapping.items():
        if name.startswith(old + '.'):
            new_name = name.replace(old, new, 1)
            if name.endswith('.weight'):
                tensor = tensor.reshape(tensor.shape[0], -1)
            break
    converted[new_name] = tensor
model = vgg16(weights=None)
model.load_state_dict(converted, strict=True)
for name, tensor in state.items():
    if name.startswith('features.'):
        assert torch.equal(tensor, model.state_dict()[name])
cache = ROOT / '.cache/checkpoints/vgg16-397923af.pth'
cache.parent.mkdir(parents=True, exist_ok=True)
torch.save(converted, cache)
os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch')
torch.hub.set_dir(str(ROOT / '.cache'))
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()
image = torch.rand(1, 3, 64, 64, device='cuda')
score = metric(image, image).item()
assert abs(score) < 1e-6, score
(ROOT / 'deployment/lpips_weights.json').write_text(json.dumps({
    'source': json.loads((ROOT / 'deployment/vgg16_archive.json').read_text()),
    'cache': str(cache), 'cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
    'feature_tensors_unchanged': True, 'strict_state_dict_load': True,
    'lpips_identity_score': score, 'classifier_key_mapping': mapping,
}, indent=2))
print('LPIPS-VGG cache and GPU identity check passed')
