from . import Dataset, Sampler, ItemType
from typing import Callable, Iterator, Union


class FilteredDataset(Dataset):
    """Filter dataset by filtering keys.
    Args:
        dataset: The dataset to be filtered.
        filter_fn: Callable that takes a single value and returns Boolean.
        filter_item: Defaults to False. When not False, the items will be loaded
        from dataset first, then the items will be passed to filter_fn instead
        of the keys.

    Warning: Setting filter_item when dataset requires file system I/O can be
    very slow.
    """

    def __init__(
        self,
        dataset: Dataset,
        filter_fn: Callable[[Union[str, ItemType]], bool],
        filter_item: bool = False,
    ):
        self.dataset = dataset
        if filter_item:
            self.filtered_keys = []
            for key in self.dataset:
                item = self.dataset[key]
                if filter_fn(item):
                    self.filtered_keys.append(key)
        else:
            self.filtered_keys = list(filter(filter_fn, dataset))

    def __len__(self) -> int:
        return len(self.filtered_keys)

    def __getitem__(self, key: str) -> ItemType:
        if key not in self.filtered_keys:
            raise KeyError(f"Key {key} does not exists.")
        else:
            return self.dataset[key]

    def __contain__(self, key: str) -> bool:
        return key in self.filtered_keys

    def __iter__(self) -> Iterator[str]:
        return iter(self.filtered_keys)


class MappedDataset(Dataset):
    """
    Apply some function on all items in a single dataset. (Lazy Evaluation)
    Args:
        dataset: The dataset to be modified.
        func: The modification function. The function can contain additional information
            on `drop_int_keys`, `drop_arr_keys`, `add_int_keys`, `add_arr_keys`.
    """

    def __init__(self, dataset: Dataset, func: Callable[[ItemType], ItemType]):
        super().__init__()
        self.func = func
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key: str) -> ItemType:
        return self.func(self.dataset[key].copy())

    def __contain__(self, key: str) -> bool:
        return key in self.dataset

    def __iter__(self) -> Iterator[str]:
        return iter(self.dataset)


class MappedSampler(Sampler):
    """
    Apply some function on the sampled item.
    Args:
        sampler: The sampler to be modified.
        func: The modification function. The function can contain additional information
            on `drop_int_keys`, `drop_arr_keys`, `add_int_keys`, `add_arr_keys`.
    """

    def __init__(self, sampler: Sampler, func: Callable[[ItemType], ItemType]):
        super().__init__()
        self.func = func
        self.sampler = sampler

    def sample(self) -> ItemType:
        item = self.sampler.sample()
        return self.func(item.copy())


class FilteredSampler(Sampler):
    """
    Sample repeatedly until a sample passes the test.
    It is the user's responsibility to ensure that some item can pass the test
    with high probability.
    """

    def __init__(
        self,
        sampler: Sampler,
        pred: Callable[[ItemType], bool],
        max_retry: int = 65536,
    ):
        assert isinstance(sampler, Sampler)
        self.pred = pred
        self.sampler = sampler
        self.max_retry = max_retry

    def sample(self) -> ItemType:
        for t in range(self.max_retry):
            item = self.sampler.sample()
            if self.pred(item):
                return item
        raise RuntimeError(
            f"FilteredSampler failed to sample after {self.max_retry} tries."
        )
