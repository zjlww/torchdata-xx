import torch
from typing import Tuple, List, Iterable

from . import ItemType


from torch.nn.utils.rnn import pad_sequence
import torch


def geometric_partition(
    min_len: int, max_len: int, p=0.3, min_gap: int = 16
) -> List[Tuple[int, int]]:
    """Generate a partition of interval [min_len, max_len), such that in each part
    [a, b), we have (b - a) / b is approximately p.
    """
    r = 1 / (1 - p)
    result = []
    a = min_len
    while a < max_len:
        b = min(int(a * r), max_len)
        if b <= a + min_gap:
            b = a + min_gap
        result.append((a, b))
        a = b
    return result


def send_item(
    item: ItemType, keys: Iterable[str], device: torch.device = None
) -> ItemType:
    """Sending PyTorch tensors in an item to a device.
    Args:
        item: a dictionary containing the item to be sent.
        keys: the keys of the item to be sent.
        device: if not None, the tensor will be pushed to the device.
    Returns:
        A dictionary containing the sent item.
    """
    if device is not None:
        for key in keys:
            item[key] = item[key].pin_memory().to(device=device, non_blocking=True)

    return item


def stack_items(
    items: Tuple[ItemType],
    arr_keys: Iterable[str],
    int_keys: Iterable[str],
    device: torch.device,
) -> ItemType:
    """Merging a tuple of items into a single item. Various keys are merged in
    different ways.
    Args:
        items: a tuple of items to be merged.
        arr_keys: these keys are merged with pad_sequence. NOTE: Currently
            only supporting 1D arrays.
        int_keys: these keys are concatenated into a IntTensor.
        device: where to store the int tensors.
    Returns:
        A dictionary containing the merged items.
    """
    item = {}

    for key in arr_keys:
        arrs = [torch.as_tensor(item[key]) for item in items]
        # [len_i, ...] -> [n_batch, max_len, ...] padding zero on the right.
        item[key] = pad_sequence(arrs, batch_first=True)

    for key in int_keys:
        item[key] = torch.as_tensor(
            [item[key] for item in items], device=device, dtype=torch.int32
        )

    return item
