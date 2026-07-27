import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.model import GPT
from src.tokenizer import RecipeTokenizer
from src.training_utils import generate


def ask_for_ingredients():
    """
    Ask the user to enter ingredients separated by commas.
    """

    print("\nEnter the ingredients you want to use.")
    print("Separate ingredients with commas.")
    print("You may include measurements if you know them.")

    user_input = input("\nIngredients: ").strip()

    ingredients = [
        ingredient.strip()
        for ingredient in user_input.split(",")
        if ingredient.strip()
    ]

    return ingredients


def create_default_title(ingredients):
    """
    Create a basic title from the first three ingredients.
    """

    main_ingredients = ingredients[:3]

    title_words = [
        ingredient.title()
        for ingredient in main_ingredients
    ]

    return " and ".join(title_words) + " Recipe"


def build_recipe_prompt(title, ingredients):
    """
    Build a prompt matching the format used during training.
    """

    ingredient_text = "\n".join(
        ingredients
    )

    return (
        "<RECIPE>\n"
        "<TITLE>\n"
        f"{title}\n"
        "<INGREDIENTS>\n"
        f"{ingredient_text}\n"
        "<DIRECTIONS>\n"
    )


def generate_model(args):
    """
    Load a checkpoint and generate a recipe.
    """

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    checkpoint_path = Path(args.load)

    tokenizer = RecipeTokenizer()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model = GPT(
        vocab_size=checkpoint["vocab_size"],
        context_size=checkpoint["context_size"],
        n_embd=checkpoint["n_embd"],
        n_head=checkpoint["n_head"],
        n_layer=checkpoint["n_layer"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    if args.interactive:
        ingredients = ask_for_ingredients()

        if not ingredients:
            print("No ingredients were entered.")
            return

        print("\nEnter a title, or press Enter to create one automatically.")

        title = input("Title: ").strip()

        if not title:
            title = create_default_title(
                ingredients
            )

        prompt = build_recipe_prompt(
            title=title,
            ingredients=ingredients
        )

    elif args.prompt:
        prompt = args.prompt

    else:
        prompt = (
            "<RECIPE>\n"
            "<TITLE>\n"
        )

    print("=" * 20, "EVALUATION", "=" * 20)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print("\nPrompt:\n")
    print(prompt)

    prompt_tokens = tokenizer.encode(
        prompt
    )

    start_idx = torch.tensor(
        prompt_tokens,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    end_recipe_id = tokenizer.token_id(
        "<END_RECIPE>"
    )

    with torch.no_grad():
        generated_tokens = generate(
            model=model,
            context_size=checkpoint["context_size"],
            start_idx=start_idx,
            number_of_tokens=args.token_count,
            stop_token_id=end_recipe_id
        )

    generated_recipe = tokenizer.decode(
        generated_tokens[0].tolist()
    )

    print("\nGenerated recipe:\n")
    print(generated_recipe)

    