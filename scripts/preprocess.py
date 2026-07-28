# Transforms raw dataset into consistent strcuted text

import ast
from pathlib import Path

import pandas as pd


# Finds the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input and output file locations
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "1_Recipe_csv.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "Recipe_Clean.csv"
OUTPUT_TEXT_PATH = PROJECT_ROOT / "data" / "processed" / "recipes.txt"


# Ensures ingredients and directions into strings
def clean_list_text(value):
   

    try:
        items = ast.literal_eval(value)

        if isinstance(items, list):
            return "\n".join(
                str(item).strip()
                for item in items
                if str(item).strip()
            )

    except (ValueError, SyntaxError, TypeError):
        pass

    return str(value).strip()

# Replace Unicode fraction symbols with regular text fractions
def normalize_fractions(value):

    fraction_map = {
        "¼": "1/4",
        "½": "1/2",
        "¾": "3/4",
        "⅓": "1/3",
        "⅔": "2/3",
        "⅛": "1/8",
        "⅜": "3/8",
        "⅝": "5/8",
        "⅞": "7/8",
    }

    value = str(value)

    for fraction, replacement in fraction_map.items():
        value = value.replace(
            fraction,
            replacement,
        )

    return value.strip()
   

# Convert one dataframe row into the structured recipe format used for model training
def format_recipe(row):

    return (
        "<RECIPE>\n"
        "<TITLE>\n"
        f"{row['recipe_title']}\n"
        "<INGREDIENTS>\n"
        f"{row['ingredients']}\n"
        "<DIRECTIONS>\n"
        f"{row['directions']}"
    )


# Coordinates preprocessing
def main():
    df = pd.read_csv(INPUT_PATH)
    df = df.fillna("")
   
    df["ingredients"] = (
        df["ingredients"]
        .apply(clean_list_text)
        .apply(normalize_fractions)
    )

    df["directions"] = (
        df["directions"]
        .apply(clean_list_text)
        .apply(normalize_fractions)
    )

    df["recipe_title"] = (
        df["recipe_title"]
        .apply(normalize_fractions)
    )

    df["text"] = df.apply(
        format_recipe,
        axis=1,
    )

    text = "\n\n<END_RECIPE>\n\n".join(
        df["text"].tolist()
    )

    text = text + "\n\n<END_RECIPE>"

    # Remove unwanted symbols
    replacements = {
        "®": "",
        "™": "",
        "‘": "'",
        "’": "'",
        "℉": "°F",
        "\\": "",
        "{": "",
        "}": "",
    }

    for old_character, new_character in replacements.items():
        text = text.replace(
            old_character,
            new_character,
        )

    # Make sure output folders exist
    OUTPUT_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the cleaned dataframe
    df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
    )

    # Save the text used to train the model
    OUTPUT_TEXT_PATH.write_text(
        text,
        encoding="utf-8",
    )

    print(f"Number of recipes: {len(df):,}")
    print(f"Saved cleaned CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved training text to: {OUTPUT_TEXT_PATH}")
    print("\nSpecial-token counts:")

    for token in [
        "<RECIPE>",
        "<TITLE>",
        "<INGREDIENTS>",
        "<DIRECTIONS>",
        "<END_RECIPE>",
    ]:
        print(
            f"{token}: {text.count(token):,}"
        )

    # Show only a sample
    print("\nSample processed text:\n")
    print(text[:1000])


if __name__ == "__main__":
    main()
