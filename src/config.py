# This file not used in current pipeline
# Most funcitons incldued in argparse in command line interface in gpt.py

from dataclasses import dataclass
from pathlib import Path

import torch


# The main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # File paths
    raw_csv_path: Path = PROJECT_ROOT / "data" / "raw" / "recipes.csv"
    processed_text_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "recipes.txt"
    )
    checkpoint_path: Path = (
        PROJECT_ROOT / "checkpoints" / "recipe_transformer.pt"
    )

    # Data settings
    train_split: float = 0.9

    # Training settings
    batch_size: int = 16
    context_size: int = 128
    learning_rate: float = 1e-3
    max_steps: int = 5000

    # Evaluation settings
    eval_interval: int = 500
    eval_iters: int = 100

    # Transformer settings
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2

    # Reproducibility
    seed: int = 1337

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

