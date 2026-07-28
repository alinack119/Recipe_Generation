# Converts text into model input
# Model cannot understand strings only integers

from typing import Iterable

import tiktoken


SPECIAL_TOKENS = {
    "<RECIPE>": 50257,
    "<TITLE>": 50258,
    "<INGREDIENTS>": 50259,
    "<DIRECTIONS>": 50260,
    "<END_RECIPE>": 50261,
}


class RecipeTokenizer:
    def __init__(self) -> None:
        base_encoding = tiktoken.get_encoding("gpt2")

        # Create a new tokenizer that includes recipe special tokens
        self.encoding = tiktoken.Encoding(
            name="gpt2_recipe",
            pat_str=base_encoding._pat_str,
            mergeable_ranks=base_encoding._mergeable_ranks, #byte-pair merge ranks
            special_tokens={
                **base_encoding._special_tokens,
                **SPECIAL_TOKENS,
            },
        )

        self.allowed_special = set(SPECIAL_TOKENS.keys())
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> list[int]:

        return self.encoding.encode(
            text,
            allowed_special=self.allowed_special,
        )

    def decode(self, token_ids: Iterable[int]) -> str:

        return self.encoding.decode(list(token_ids))

    def token_id(self, token: str) -> int: # Special tokens

        if token not in SPECIAL_TOKENS:
            raise KeyError(f"Unknown special token: {token}")

        return SPECIAL_TOKENS[token]

