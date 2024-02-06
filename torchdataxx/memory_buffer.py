from . import Sampler, ItemType, Dataset
from threading import Thread
from queue import Queue
from typing import Iterator


def push_queue_forever(sampler: Sampler, queue: Queue) -> None:
    while True:
        try:
            item = sampler.sample()
            queue.put(item)
        except Exception as e:
            raise e


class MemoryBufferedSampler(Sampler):
    """
    A multithreading sampling buffer.

    This class accepts a sampler and buffers the sampling process. The buffer
    samples in prior to fetch to reduce I/O delay.

    Args:
        sampler (Sampler): The sampler to buffer.
        n_threads (int): The number of threads to use for sampling.
        queue_size (int): The maximum size of the buffer queue.

    Note:
        The user needs to ensure that the sampler can handle being passed as a
        function parameter in Threads.
    """

    def __init__(self, sampler: Sampler, n_threads: int, queue_size: int):
        super().__init__()
        self.queue = Queue(maxsize=queue_size)
        self.threads = [
            Thread(target=push_queue_forever, args=(sampler, self.queue), daemon=True)
            for _ in range(n_threads)
        ]
        for thread in self.threads:
            thread.start()

    def sample(self) -> ItemType:
        return self.queue.get()

    def __len__(self) -> int:
        return self.queue.qsize()


class MemoryBufferedDataset(Dataset):
    """
    A cache layer on a dataset. Key access will cache the loaded item in
    memory.
    Warning: This cache does not limit total memory consumption.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key: str) -> ItemType:
        if key in self.cache:
            return self.cache[key]
        else:
            item = self.dataset[key]
            self.cache[key] = item
            return item

    def __iter__(self) -> Iterator[str]:
        for item in self.dataset:
            yield item

    def __contain__(self, key: str) -> bool:
        return key in self.dataset
