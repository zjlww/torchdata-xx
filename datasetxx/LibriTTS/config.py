from torchdataxx.config import normalize_path, Config, Path


class DatasetConfig(Config):
    frame_size = 256
    # Raw Audio Format:
    sampling_rate = 24000
    path_dataset = normalize_path("/bulk/datasets/LibriTTS")

    pattern_wav_glob = "**/*.wav"

    def path_to_key(path: Path) -> str:
        return path.stem.split(".")[0]
