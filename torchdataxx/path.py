from pathlib import Path
from . import normalize_path, Dataset, ItemType
from typing import Union, Callable, Iterator, Dict


class GlobPathDataset(Dataset):
    """Create a dataset with a root path and a glob pattern.
    Args:
        glob_pattern: See Path.glob() for more details.
        root_path: Root where the globbing happens.
        path_to_key_function: Function that converts a Path object to str.
                              A key is used to uniquely identify an item.
    {
        "somekey": {
            "path": "/some/path"
            "key": "somekey"
        }, ...
    }
    """

    def __init__(
        self,
        glob_pattern: str,
        root_path: Union[str, Path],
        path_to_key_function: Callable[[Path], str],
    ) -> None:
        self.root_path = normalize_path(root_path)
        self.matched_paths = self.root_path.glob(glob_pattern)
        self.key_to_path = {}
        for matched_path in self.matched_paths:
            key = path_to_key_function(matched_path)
            self.key_to_path[key] = {"path": matched_path, "key": key}

    def __len__(self) -> int:
        return len(self.key_to_path)

    def __getitem__(self, key: str) -> ItemType:
        return self.key_to_path[key]

    def __contain__(self, key: str) -> bool:
        return key in self.key_to_path

    def __iter__(self) -> Iterator[str]:
        return iter(self.key_to_path)
