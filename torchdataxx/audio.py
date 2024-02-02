from torch import Tensor, FloatTensor
from typing import Tuple


def resample(x: Tensor, src_sr: int, tgt_sr: int) -> Tensor:
    """Wrapper around resample in torchaudio. With additional
        type conversions.
    Args:
        x (Tensor): [n_samples].
    Returns:
        y (Tensor): [ceil(n_samples / src_sr * tgt_sr)].
    """
    from torchaudio.functional import resample

    y = resample(x.float(), src_sr, tgt_sr).to(x.dtype)
    return y


def wav_load_mono(path: str) -> Tuple[Tensor, int]:
    """Wrapper around sox_load in torchaudio.
    See https://pytorch.org/audio/stable/backend.html#load for details.
    Returns:
        x (Tensor): [n_samples].
        sr (int): sampling rate.
    Warning: This function does not handle resampling.
    """
    from torchaudio.backend._sox_io_backend import load as sox_load

    x, sr = sox_load(path, normalize=False, channels_first=True)
    return x.view(-1), sr


def wav_save_pcm_16_mono(path: str, x: FloatTensor, sampling_rate: int):
    """Save to PCM 16 Mono-channel WAV file.
    Args:
        filepath (str): target path with wav extension;
        x (Tensor): Tensor of any shape in range [-1, 1].
            Note that this function does not do normalization.
    """
    from torchaudio.backend._sox_io_backend import save as sox_save

    x = x.detach().cpu().view(-1).unsqueeze(0)
    # [1, n_sample]
    sox_save(
        path,
        x,
        sampling_rate,
        channels_first=True,
        encoding="PCM_S",
        bits_per_sample=16,
    )
