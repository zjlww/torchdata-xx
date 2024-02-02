"""
Save Item Shards with TorchScript.
"""

import torch
from torch import nn
from torch.jit import ScriptModule
from torchdataxx import ShardType, PathType, normalize_path


def gen_test_shard(n: int = 500, m: int = 20000) -> ShardType:
    shard = dict()
    for idx in range(n):
        shard[str(idx)] = {"x": torch.randn(m)}
    return shard


def encode_shard(shard: ShardType) -> ScriptModule:
    shard_module = nn.Module()
    for item_key, item in shard.items():
        item_module = nn.Module()
        for key, value in item.items():
            setattr(item_module, key, value)
        setattr(shard_module, item_key, item_module)
    return torch.jit.script(shard_module)


def save_shard(shard: ShardType, path: PathType) -> None:
    path = normalize_path(path)
    m = encode_shard(shard)
    m.save(str(path))
