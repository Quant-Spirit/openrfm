"""Saved memory modules: {schema, config, state_dict}."""
from dataclasses import asdict, fields

import torch

from openrfm.models.memory.config import MemoryConfig
from openrfm.models.memory.module import MemoryModule

SCHEMA = 'memory_v1'


def memory_payload(module):
    return dict(schema=SCHEMA, config=asdict(module.config),
                state_dict={k: v.detach().cpu().clone() for k, v in module.state_dict().items()})


def save_memory(module, path):
    torch.save(memory_payload(module), path)


def memory_from_payload(payload):
    """The memory module of a payload saved by `save_memory` or of the released checkpoint."""
    payload = payload['trainer']['temporal_memory'] if 'trainer' in payload else payload
    if payload['schema'] != SCHEMA:
        raise ValueError(f"unknown memory schema {payload['schema']!r}")
    stored = payload['config']
    if stored.get('history', 'cells') != 'cells' or not stored.get('context_tokens', True):
        raise ValueError('the payload needs a memory variant this module does not implement')
    names = {f.name for f in fields(MemoryConfig)}
    config = {k: v for k, v in stored.items() if k in names}
    config['target_depths'] = tuple(config.get('target_depths', MemoryConfig.target_depths))
    module = MemoryModule(MemoryConfig(**config))
    module.load_state_dict(payload['state_dict'], strict=True)
    return module


def load_memory(path, map_location='cpu'):
    """A memory module saved by `save_memory`, or the released checkpoint."""
    return memory_from_payload(torch.load(path, map_location=map_location, weights_only=True))
