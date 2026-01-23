"""
Minimal config for local Vid2Seq training (PyTorch path).
Avoids Scenic/TF dependencies. Override via CLI if needed.
"""

from types import SimpleNamespace


def get_config():
    return {
        "dataset_configs": {
            "base_dir": "dataset/breakfast/breakfast_5",
        },
        "batch_size": 4,
        "num_training_epochs": 10,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "max_events": 20,
        "train_stage": "1",  # '1', '2', 'mixed'
        "mask_ratio": 0.3,
        "mixed_stage2_ratio": 0.5,
        "caption_loss_weight": 1.0,
    }
