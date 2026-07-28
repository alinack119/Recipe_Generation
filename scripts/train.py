# Build and train the model

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.model import GPT
from src.tokenizer import RecipeTokenizer
from src.training_utils import setup_data, train


# Train the recipe transformer using settings received from the command line interface
def train_model(args):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    torch.manual_seed(args.seed)

    # Processed data
    with open(
        args.input,
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    tokenizer = RecipeTokenizer()

    # Encoding
    encoded_text = tokenizer.encode(text)

    data = torch.tensor(
        encoded_text,
        dtype=torch.long
    )

    # Split into training and validation data
    split_index = int(0.9 * len(data))

    train_data = data[:split_index]
    val_data = data[split_index:]

    # Stores tensors and selected device inside training_utils.py
    setup_data(
        train_data,
        val_data,
        device
    )

    # transformer model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=args.n_embd,
        context_size=args.context_size,
        n_head=args.n_head,
        n_layer=args.n_layer
    )

    model = model.to(device)

    print("=" * 20, "TRAINING", "=" * 20)
    print(f"Device: {device}")
    print(f"Input file: {args.input}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training tokens: {len(train_data)}")
    print(f"Validation tokens: {len(val_data)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context size: {args.context_size}")
    print(f"Embedding size: {args.n_embd}")
    print(f"Attention heads: {args.n_head}")
    print(f"Layers: {args.n_layer}")
    print(f"Training steps: {args.steps}")

    # Train model
    train(
        model=model,
        steps=args.steps,
        batch_size=args.batch_size,
        context_size=args.context_size,
        learning_rate=args.lr,
        report_frequency=args.report
    )

    checkpoint_path = Path(args.save)

    # Create checkpoint folder
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save model and hyperparameters
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "context_size": args.context_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer
        },
        checkpoint_path
    )

    print(f"Model saved to: {checkpoint_path}")
