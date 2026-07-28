# This file is not used in current pipeline
# Alternative data loading and batching file

from pathlib import Path

import torch

from tokenizer import RecipeTokenizer


def load_token_data(
    text_path: Path,
    tokenizer: RecipeTokenizer,
    train_split: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read the recipe text file, tokenize it, and divide it into
    training and validation data.
    """

    if not text_path.exists():
        raise FileNotFoundError(
            f"Processed text file was not found: {text_path}\n"
            "Run scripts/preprocess.py first."
        )

    text = text_path.read_text(encoding="utf-8")

    token_ids = tokenizer.encode(text)

    data = torch.tensor(
        token_ids,
        dtype=torch.long,
    )

    split_index = int(len(data) * train_split)

    train_data = data[:split_index]
    validation_data = data[split_index:]

    return train_data, validation_data


def get_batch(
    split: str,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    batch_size: int,
    context_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create one batch of input sequences and target sequences.
    """

    if split == "train":
        data = train_data
    elif split == "validation":
        data = validation_data
    else:
        raise ValueError(
            "split must be either 'train' or 'validation'"
        )

    if len(data) <= context_size:
        raise ValueError(
            f"The {split} data has only {len(data)} tokens, "
            f"but context_size is {context_size}."
        )

    starting_positions = torch.randint(
        low=0,
        high=len(data) - context_size,
        size=(batch_size,),
    )

    # Input sequences
    x = torch.stack(
        [
            data[position : position + context_size]
            for position in starting_positions
        ]
    )

    # Target sequences are shifted forward by one token
    y = torch.stack(
        [
            data[position + 1 : position + context_size + 1]
            for position in starting_positions
        ]
    )

    return x.to(device), y.to(device)

    print("file ran successul")
