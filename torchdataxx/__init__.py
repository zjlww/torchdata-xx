"""
A custom data loading library.
An `Item` is a dictionary from strings to Any type.
ItemType, Dataset, and Sampler are fundamental types in the library.
"""

from pathlib import Path
from os.path import expanduser
from random import choice, choices, shuffle
from functools import reduce
from abc import ABC, abstractmethod
from torch import Tensor
from typing import (
    List,
    Tuple,
    Iterator,
    Any,
    Union,
    Dict,
    Iterable,
    Callable,
    TYPE_CHECKING,
)
from threading import RLock


ItemType = Dict[str, Union[bool, int, float, str, Tensor]]
ShardType = Dict[str, ItemType]
ItemList = Iterable[ItemType]
PathType = Union[str, Path]
Partition = List[Tuple[int, int, int]]


if TYPE_CHECKING:
    from .functional import (
        MappedSampler,
        MappedDataset,
        FilteredSampler,
        FilteredDataset,
    )


class Dataset(ABC):
    """Dataset abstract base class."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> ItemType:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Iter through keys."""
        pass

    @abstractmethod
    def __contain__(self, key: str) -> bool:
        """Test whether key is within the existing key set."""
        pass

    @property
    def keys(self) -> List[str]:
        return list(self)

    def map(self, func: Callable[[ItemType], ItemType]) -> "MappedDataset":
        from .functional import MappedDataset

        return MappedDataset(self, func)

    def filter(self, key_pred: Callable[[str], bool]) -> "FilteredDataset":
        from .functional import FilteredDataset

        return FilteredDataset(self, key_pred, filter_item=False)


class Sampler(ABC):
    """Sampler provides a sample method."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def sample(self) -> ItemType:
        pass

    def map(self, func: Callable[[ItemType], ItemType]) -> "MappedSampler":
        from .functional import MappedSampler

        return MappedSampler(self, func)

    def filter(
        self, item_pred: Callable[[ItemType], bool], max_retry: int = 65536
    ) -> "FilteredSampler":
        from .functional import FilteredSampler

        return FilteredSampler(self, item_pred, max_retry)


class BatchSampler(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def sample(self) -> ItemList:
        pass


def normalize_path(path: PathType) -> Path:
    """Normalize path, as relative paths and tilde expansion does not work
    for some libraries.
    """
    return Path(expanduser(str(path)))


class SampledDataset(Sampler):
    """To sample from a dataset."""

    def __init__(self, dataset: Dataset, sequential: bool = False) -> None:
        self.dataset = dataset
        self.keys = [key for key in self.dataset]

        self.sequential = sequential
        self.seq_lock = RLock()
        self.seq_state = []

    def sample(self) -> ItemType:
        if self.sequential:
            with self.seq_lock:
                if len(self.seq_state) == 0:
                    self.seq_state = self.keys.copy()
                    shuffle(self.seq_state)
                return self.dataset[self.seq_state.pop()]
        else:
            return self.dataset[choice(self.keys)]


class SampledSamplers(Sampler):
    """
    Sample from multiple samplers, with given weight with replacement.
    Args:
        samplers: Samplers.
        sampler_ids: IDs of samplers, inorder to identify the source of
            samples.
        weights: Weights of each sampler, can be any positive numbers.
    """

    def __init__(
        self,
        samplers: Iterable[Sampler],
        sampler_ids: Iterable[str],
        weights: Iterable[Union[int, float]],
    ):
        self.id_samplers = [
            (idx, sampler) for idx, sampler in zip(sampler_ids, samplers)
        ]
        self.weights = list(weights)

    def sample(self) -> ItemType:
        """Get a sample from sampled samplers.
        Example sample: {
            ...(other fields),
            "sampler_id": "someid",
        }
        """
        sampler_id, sampler = choices(self.id_samplers, self.weights, k=1)[0]
        item = sampler.sample()
        item.update(sampler_id=sampler_id)
        return item


class ZippedDatasets(Dataset):
    """Union the items in finite different datasets that have the same keys.
    If a key does not appear in all datasets, it is dropped.
    This wrapper uses lazy evaluation, items are merged when __getitem__()
    is called.
    """

    def __init__(self, datasets: Iterable[Dataset]):
        self.datasets = list(datasets)
        self.key_sets = [set(dataset) for dataset in datasets]
        self.common_key_set = reduce(lambda x, y: x & y, self.key_sets)

    def __len__(self) -> int:
        return len(self.common_key_set)

    def __getitem__(self, key: str) -> ItemType:
        if key not in self.common_key_set:
            raise KeyError("Input key is not a common key of given datasets")
        else:
            item = {}
            for dataset in self.datasets:
                item.update(dataset[key])
            return item

    def __contain__(self, key: str) -> bool:
        return key in self.common_key_set

    def __iter__(self) -> Iterator[ItemType]:
        return iter(self.common_key_set)


class UnionedDatasets(Dataset):
    """Union datasets with disjoint keys.
    This wrapper uses lazy evaluation, values are merged when __getitem__()
    is called.
    """

    def __init__(self, datasets: Iterable[Dataset], add_prefix: bool = False):
        """
        Args:
            add_prefix (bool, optional): add prefix to keys in different datasets.
                This is useful when input datasets may contain same keys.
        """
        self.datasets = list(datasets)
        self.key_to_dataset = {}
        self.add_prefix = add_prefix
        for idx, dataset in enumerate(datasets):
            for key in dataset:
                if add_prefix:
                    self.key_to_dataset[f"{idx:016d}_" + key] = dataset
                else:
                    self.key_to_dataset[key] = dataset

    def __len__(self) -> int:
        return len(self.key_to_dataset)

    def __getitem__(self, key: str) -> ItemType:
        if key not in self.key_to_dataset:
            raise KeyError("Input key is not a common key of given datasets")
        elif self.add_prefix:
            return self.key_to_dataset[key][key[17:]]
        else:
            return self.key_to_dataset[key][key]

    def __contain__(self, key: str) -> bool:
        return key in self.key_to_dataset

    def __iter__(self) -> Iterator[ItemType]:
        return iter(self.key_to_dataset.keys())


class DictDataset(Dataset, dict):
    """Create a dataset from a dictionary containing key -> item pairs."""

    def __init__(self, data: Dict[str, ItemType]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> ItemType:
        if key not in self.data:
            raise KeyError("Input key not found in data.")
        else:
            return self.data[key]

    def __contain__(self, key: str) -> bool:
        return key in self.data

    def __iter__(self) -> Iterator[str]:
        return iter(self.data.keys())

    def keys(self) -> Iterator[str]:
        return iter(self.data.keys())


class SavedDataset(DictDataset):
    """SavedDataset(path) is equivalent to DictDataset(torch.load(path))."""

    def __init__(self, path: PathType, map_location="cpu"):
        import torch

        path = normalize_path(path)
        if path.is_file():
            super().__init__(torch.load(path, map_location=map_location))
        else:
            super().__init__({})
