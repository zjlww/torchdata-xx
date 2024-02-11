from functools import partial
import torch
from torchdataxx.path import GlobPathDataset
from torchdataxx.memory_buffer import MemoryBufferedSampler
from torchdataxx.functional import MappedSampler, Sampler
from torchdataxx import extension
from torchdataxx.utils import send_item

from .config import DatasetConfig as dcfg


def get_segment_sampler(
    N_batch: int,
    N_samples: int,
    S_io_queue: int,
    S_io_thread: int,
    S_device_queue: int,
    device: torch.device = torch.device("cpu"),
    same_speaker: bool = True,
    samples_per_shard: int = 256,
) -> Sampler:
    ## Now switching to sharded dataset.
    # gpd = GlobPathDataset(dcfg.pattern_wav_glob, dcfg.path_dataset, dcfg.path_to_key)
    # paths = {}
    # for key, item in gpd.key_to_path.items():
    #     paths[key] = {
    #         "path": str(item["path"]),
    #         "key": key,
    #         "speaker": int(key.split("_")[0]),
    #     }
    # paths = extension.immediateDataset(paths)
    # sampler = paths.sample()
    # sampler = sampler.map(
    #     extension.functional.readAudioTransform("path", "wave", "sampling_rate", True)
    # )

    ## Loading pre-sharded dataset:
    shard_paths = list(dcfg.path_shard.glob("**/*.pt"))
    _d = {}
    for shard_path in shard_paths:
        _d[shard_path.stem] = {
            "shard_path": str(shard_path),
            "shard_id": int(shard_path.stem),
        }
    sampler = (
        extension.immediateDataset(_d)
        .permuteSample()
        .sampleShard("shard_path", "shard_id", samples_per_shard)
    )

    sampler = sampler.queue(S_io_thread, S_io_queue)
    if same_speaker:
        sampler = sampler.segmentClasswise("wave", "speaker", N_samples, 0)
    else:
        sampler = sampler.segment("wave", N_samples, 0)
    sampler = sampler.batch(N_batch)
    sampler = sampler.stack()
    sampler = MappedSampler(sampler, partial(send_item, keys=["wave"], device=device))
    sampler = MemoryBufferedSampler(sampler, 1, S_device_queue)
    return sampler
