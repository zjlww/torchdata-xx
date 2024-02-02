import torch
from torch import Tensor


class TensorBuffer:
    """
    A buffer that stores 1D tensors of different lengths and chunks them to a given size.

    This buffer is not thread safe, so locks should be added accordingly.

    Examples:
        >>> buffer = TensorBuffer()
        >>> a = torch.ones(3)
        >>> b = torch.zeros(4)
        >>> buffer.push(a)
        >>> buffer.push(b)
        >>> c = buffer.pop(5)
        >>> print(c)
        Tensor([1, 1, 1, 0, 0])
        >>> print(buffer.buf)
        Tensor([0, 0])

    Attributes:
        buf (torch.Tensor): The buffer that stores the tensors.
    """
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.buf = torch.empty([0], dtype=dtype)

    def push(self, tensor: Tensor) -> None:
        """
        Pushes a tensor into the buffer.

        Args:
            tensor (torch.Tensor): The tensor to be pushed into the buffer.
        """
        self.buf = torch.cat([self.buf, tensor])

    def remain(self, segment_size: int) -> int:
        """
        Returns the number of remaining segments in the buffer.

        Args:
            segment_size (int): The size of each segment.

        Returns:
            int: The number of remaining segments in the buffer.
        """
        return len(self.buf) // segment_size

    def pop(self, segment_size: int) -> Tensor:
        """
        Pops a tensor of the given segment size from the buffer.

        Args:
            segment_size (int): The size of the segment to be popped.

        Returns:
            torch.Tensor: The popped tensor.

        Raises:
            RuntimeError: If the buffer is not long enough to chunk out the given segment size.
        """
        if self.remain(segment_size) > 0:
            segment = self.buf[:segment_size]
            self.buf = self.buf[segment_size:]
            return segment
        else:
            raise RuntimeError(f"Buffer not long enough to chunk out {segment_size} elements.")
