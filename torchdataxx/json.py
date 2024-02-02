"""Handling JSONified Datasets.
A JSON dataset is the JSON serialization of { "key": { item }, ... }.
"""
import orjson
from . import PathType, normalize_path, Dataset, ItemType
from typing import Iterator


class JSONDataset(Dataset):
    """Load a dictionary stored in json of following form:
    { "key_1": {item_1}, ... }.
    The file should be loadable by the orjson library.
    Warning:
        If the file does not exists, the dataset will be empty.
        This may lead to confusing missing key errors.
    """

    def __init__(self, json_path: PathType):
        self.json_path = normalize_path(json_path)
        if self.json_path.is_file():
            with open(self.json_path, "rb") as f:
                self.data = orjson.loads(f.read())
        else:
            self.data = {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> ItemType:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        for key in self.data:
            yield key

    def __contain__(self, key: str) -> bool:
        return key in self.data


def union_json(a: PathType, b: PathType, c: PathType):
    """Union two dataset stored as json files into a single json file.
    a + b => c.
    """
    from orjson import dumps, OPT_INDENT_2

    a, b, c = tuple(normalize_path(p) for p in (a, b, c))

    assert a.is_file() and b.is_file()
    A = JSONDataset(a)
    B = JSONDataset(b)
    C = A.data
    C.update(B.data)
    json = dumps(C, option=OPT_INDENT_2)
    with open(c, "wb") as f:
        f.write(json)
