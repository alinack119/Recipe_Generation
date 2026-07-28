#Command line interface

import argparse

from train import train_model
from generate import generate_model


parser = argparse.ArgumentParser(
    description="Train or evaluate the recipe transformer."
)


# Shared argument
parser.add_argument(
    "--input",
    type=str,
    default="data/processed/recipes.txt",
    help="Path to the processed recipe text file."
)

parser.add_argument(
    "--seed",
    type=int,
    default=1337,
    help="Random seed."
)


# Create train and eval commands
subparsers = parser.add_subparsers(
    dest="command",
    required=True
)


# train arguments
train_parser = subparsers.add_parser(
    "train",
    help="Train the recipe transformer."
)

train_parser.add_argument(
    "--save",
    type=str,
    default="checkpoints/recipe_transformer.pt",
    help="Path for the saved model checkpoint."
)

train_parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
    help="Training batch size."
)

train_parser.add_argument(
    "--context-size",
    type=int,
    default=256,
    help="Maximum context length."
)

train_parser.add_argument(
    "--n-embd",
    type=int,
    default=128,
    help="Embedding size."
)

train_parser.add_argument(
    "--n-head",
    type=int,
    default=4,
    help="Number of attention heads."
)

train_parser.add_argument(
    "--n-layer",
    type=int,
    default=4,
    help="Number of transformer layers."
)

train_parser.add_argument(
    "--steps",
    type=int,
    default=5000,
    help="Number of training steps."
)

train_parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Learning rate."
)

train_parser.add_argument(
    "--report",
    type=int,
    default=500,
    help="How often loss is reported."
)


# eval arguments
eval_parser = subparsers.add_parser(
    "eval",
    help="Generate text from a trained model."
)

eval_parser.add_argument(
    "--load",
    type=str,
    default="checkpoints/recipe_transformer.pt",
    help="Path to the saved model checkpoint."
)

eval_parser.add_argument(
    "--prompt",
    type=str,
    default="<RECIPE>\n<TITLE>",
    help="Starting prompt for generation."
)

eval_parser.add_argument(
    "--token-count",
    type=int,
    default=400,
    help="Maximum number of new tokens."
)

eval_parser.add_argument(
    "--interactive",
    action="store_true",
    help="Ask the user for recipe information interactively."
)


args = parser.parse_args()


if args.command == "train":
    train_model(args)

elif args.command == "eval":
    generate_model(args)
