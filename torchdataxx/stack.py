from typing import Iterable, Optional
from . import ItemType, Sampler
import torch
from .utils import stack_items


class BatchAndStack(Sampler):
    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        arr_keys: Iterable[str],
        int_keys: Iterable[str],
        device: Optional[torch.device] = None,
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.arr_keys = arr_keys
        self.int_keys = int_keys
        self.device = device

    def sample(self) -> ItemType:
        items = [self.sampler.sample() for _ in range(self.batch_size)]
        return stack_items(items, self.arr_keys, self.int_keys, self.device)


class Stack(Sampler):
    """
    Stack a list of item into a single item. The output of self.sampler should
    be a list of items.
    """

    def __init__(
        self,
        sampler: Sampler,
        arr_keys: Iterable[str],
        int_keys: Iterable[str],
        device: torch.device,
    ):
        self.sampler = sampler
        self.arr_keys = arr_keys
        self.int_keys = int_keys
        self.device = device

    def sample(self) -> ItemType:
        items = self.sampler.sample()
        return stack_items(items, self.arr_keys, self.int_keys, self.device)
