"""
Read in raw waveforms and store them in shards at the target location.
A generated shard flooks like the following:
In file 001.pt: {
    "0000_000001_0000_0001": { "wave": Int16Tensor },
    ...
}
"""

import os
import random
import torch
from math import ceil
from tqdm import tqdm
from typing import List

from torchdataxx import UnionedDatasets
from torchdataxx.path import GlobPathDataset
from torchdataxx.shard import save_shard
from torchdataxx.audio import wav_load_mono
from .config import DatasetConfig as dcfg

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def process_shard(args) -> None:
    # shard: ShardType, shard_path: PathType
    shard, shard_path = args
    for key, item in tqdm(
        shard.items(), total=len(shard), desc=f"Worker {os.getpid()}"
    ):
        wave, sr = wav_load_mono(item["path"])
        shard[key].update(wave=wave, sr=sr)
    save_shard(shard, shard_path)


def process(shuffle_seed: int, n_chunk: int):
    dcfg.path_shard.mkdir(exist_ok=True, parents=True)
    gpd = UnionedDatasets(
        [
            GlobPathDataset(
                dcfg.pattern_wav_glob, dcfg.path_train_clean_100, dcfg.path_to_key
            ),
            GlobPathDataset(
                dcfg.pattern_wav_glob, dcfg.path_train_clean_360, dcfg.path_to_key
            ),
            GlobPathDataset(
                dcfg.pattern_wav_glob, dcfg.path_train_other_500, dcfg.path_to_key
            ),
        ],
        add_prefix=False,
    )

    items = {}
    for key in gpd:
        item = gpd[key]
        items[key] = {
            "path": str(item["path"]),
            "key": key,
            "speaker": int(key.split("_")[0]),
        }

    rng = random.Random(shuffle_seed)
    keys = list(items.keys())
    rng.shuffle(keys)

    n_total = len(keys)
    chunk_size = ceil(n_total / n_chunk)

    shards = []
    for cid in range(n_chunk):
        shard = {}
        a = cid * chunk_size
        b = a + chunk_size
        for key in keys[a:b]:
            shard[key] = items[key]
        shards.append((shard, dcfg.path_shard / f"{cid:04d}.pt"))

    pool = ProcessPoolExecutor(max_workers=8)
    pool.map(process_shard, shards)


if __name__ == "__main__":
    process(0, 128)
