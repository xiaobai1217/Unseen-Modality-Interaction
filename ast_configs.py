

def get_audio_configs(target_length=128):
    norm_stats = [-5.388, 2.94]

    train_config = {
        "num_mel_bins": 128,
        "target_length": target_length,
        "freqm": 48,
        "timem": 192,
        "mixup": 0.5,
        "dataset": "Kinetics400",
        "mode": "train",
        "mean": norm_stats[0],
        "std": norm_stats[1],
        "noise": True,
    }
    val_config = {
        "num_mel_bins": 128,
        "target_length": target_length,
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "dataset": "Kinetics400",
        "mode": "evaluation",
        "mean": norm_stats[0],
        "std": norm_stats[1],
        "noise": False,
    }

    return train_config, val_config