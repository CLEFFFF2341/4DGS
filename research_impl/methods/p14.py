from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..adapter import ModelAdapter, STATIC_BUFFERS, STATIC_PARAMETERS


@torch.no_grad()
def convert_candidates(adapter: ModelAdapter, probe: dict) -> tuple[ModelAdapter, dict]:
    candidate = probe["candidate"].bool().numpy()
    converted_rows = np.flatnonzero(candidate).astype(np.int64)
    retained_rows = np.flatnonzero(~candidate).astype(np.int64)
    transformed = adapter.gather(range(adapter.static_count), retained_rows)
    model = transformed.model
    source = adapter.model
    cidx = torch.as_tensor(converted_rows, dtype=torch.long, device="cuda")
    append = {
        "_xyz": probe["intercept"].cuda().index_select(0, cidx),
        "_features_dc": source._features_dc_motion.index_select(0, cidx),
        "_features_rest": source._features_rest_motion.index_select(0, cidx),
        "_scaling": source._scaling_motion.index_select(0, cidx),
        "_rotation": probe["reference_rotation"].cuda().index_select(0, cidx),
        "_opacity": source._opacity_motion.index_select(0, cidx),
        "_xyz_disp": probe["displacement"].cuda().index_select(0, cidx),
    }
    old_static_count = model._xyz.shape[0]
    for name in STATIC_PARAMETERS:
        original = getattr(model, name)
        setattr(model, name, nn.Parameter(torch.cat((original, append[name]), dim=0).detach().clone().requires_grad_(original.requires_grad)))
    for name in STATIC_BUFFERS:
        value = getattr(model, name)
        if value.ndim and value.shape[0] == old_static_count:
            padding = torch.zeros((len(converted_rows), *value.shape[1:]), dtype=value.dtype, device=value.device)
            setattr(model, name, torch.cat((value, padding), dim=0))
    transformed.static_rows = np.concatenate((adapter.static_rows, -(converted_rows + 1)))
    mapping = {
        "converted_dynamic_rows": converted_rows.tolist(),
        "retained_dynamic_rows": retained_rows.tolist(),
        "negative_static_row_encoding": "-(original_dynamic_row+1); parent kind remains dynamic",
    }
    return transformed, mapping
