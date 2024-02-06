from os import makedirs
from os.path import expanduser
from pathlib import Path
from typing import Callable, Dict, Optional, Text, Union
import orjson
from hashlib import md5

PathType = Union[str, Path]


def normalize_path(path: PathType) -> Path:
    """Normalize path, as relative paths and tilde expansion does not work
    for some libraries.
    """
    return Path(expanduser(str(path))).resolve()


def json_safe(obj) -> bool:
    try:
        orjson.dumps(obj)
    except:
        return False
    return True


class Config:
    """The config base class. As python classes are basically dictionaries. A
    static autocomplete engine can read and complete items in python class
    syntax.
    """

    @classmethod
    def parameters(cls):
        """Dump attributes to a dictionary."""
        d = dict()
        mro = list(cls.mro())
        mro.reverse()
        for some_cls in mro:
            for key, value in some_cls.__dict__.items():
                if "__" not in key:
                    if json_safe(value):
                        d[key] = value
                    if isinstance(value, Path):
                        d[key] = value.name
                    else:
                        try:
                            d[key] = value.parameters()
                        except:
                            pass
        return d

    @classmethod
    def paths(cls):
        """Dump paths to a dictionary."""
        d = dict()
        mro = list(cls.mro())
        mro.reverse()
        for some_cls in mro:
            for key, value in some_cls.__dict__.items():
                if "__" not in key and isinstance(value, Path):
                    d[key] = value
        return d

    @classmethod
    def create_paths(cls):
        """
        Automatically create paths for paths that does not exist.
        The rule here is that the key must contain word `path`, and
        its value must be a Path object.
        """
        for key, value in cls.paths().items():
            if "path" in key and isinstance(value, Path):
                value: Path
                if not value.exists():
                    makedirs(expanduser(str(value)))

    @classmethod
    def info_string(cls):
        out = ""
        p = cls.parameters()
        for key, value in p.items():
            if isinstance(value, Callable):
                out = out + f"\n{key:25s} : callable {value.__name__}"
            else:
                out = out + f"\n{key:25s} : {value}"
        return out


def dictionary_to_config(config_dict: Dict, class_name: Optional[Text] = "DictConfig"):
    """Create a config class from a dictionary."""
    for k, v in config_dict.items():
        if isinstance(v, dict):
            config_dict[k] = dictionary_to_config(v)
    return type(class_name, (Config,), config_dict)


def dumps_config(config: Config) -> bytes:
    """dumps config.parameters() with json."""
    return orjson.dumps(
        config.parameters(), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
    )


def hash_config(config: Config) -> str:
    """Compute md5 digest of given config."""
    return md5(dumps_config(config)).hexdigest()


def save_config(config: Config, path: PathType):
    path = normalize_path(path)
    with open(path, "wb") as f:
        code = dumps_config(config)
        f.write(code)


def load_config(path: PathType):
    path = normalize_path(path)
    with open(path, "rb") as f:
        d = orjson.loads(f.read())
    return dictionary_to_config(d)


def time_string() -> str:
    import time

    return time.strftime("%D_%I:%M:%S")
