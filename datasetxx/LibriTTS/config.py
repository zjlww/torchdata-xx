from torchdataxx.config import normalize_path, Config, Path


class DatasetConfig(Config):
    frame_size = 256
    # Raw Audio Format:
    sampling_rate = 24000

    path_dataset = normalize_path("/bulk/datasets/LibriTTS")
    path_train_clean_100 = path_dataset / "train-clean-100"
    path_train_clean_360 = path_dataset / "train-clean-360"
    path_train_other_500 = path_dataset / "train-other-500"
    path_test_clean = path_dataset / "test-clean"
    path_test_other = path_dataset / "test-other"
    path_dev_clean = path_dataset / "dev-clean"
    path_dev_other = path_dataset / "dev-other"

    pattern_wav_glob = "**/*.wav"

    def path_to_key(path: Path) -> str:
        return path.stem.split(".")[0]

    path_shard = normalize_path("/titan/LibriTTS-Shards/")
