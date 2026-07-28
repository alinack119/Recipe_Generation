# Core Transformer Architecture

import torch
from torch import nn
import torch.nn.functional as F

# One self-attention head (which earliter tokens are relevent to current token)
class Head(nn.Module):
    '''
    Query: what this position is looking for.
    Key: what this position offers for matching.
    Value: the information that can be passed forward.
    '''
    def __init__(
        self,
        head_size,
        n_embd,
        context_size
    ):
        super().__init__()

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    context_size,
                    context_size
                )
            )
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = (
            q @ k.transpose(-2, -1)
        ) * (k.shape[-1] ** -0.5)

        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(
            weights,
            dim=-1
        )

        v = self.value(x)

        output = weights @ v

        return output

# Create Multiple Attention Heads
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_size,
        n_embd,
        context_size
    ):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(
                head_size,
                n_embd,
                context_size
            )
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )


# Transforms the information independently at every token position
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

# Multihead self attention + Feed-forward neural network 
class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        context_size
    ):
        super().__init__()

        head_size = n_embd // n_head

        self.self_attention = MultiHeadAttention(
            n_head,
            head_size,
            n_embd,
            context_size
        )

        self.feed_forward = FeedForward(
            n_embd
        )

    def forward(self, x):
        x = x + self.self_attention(x)
        x = x + self.feed_forward(x)

        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=32,
        context_size=8,
        n_head=4,
        n_layer=4
    ):
        super().__init__()

        self.context_size = context_size

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            context_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_head=n_head,
                    context_size=context_size
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = (
            self.token_embedding_table(idx)
        )

        position_embeddings = (
            self.position_embedding_table(
                torch.arange(
                    T,
                    device=idx.device
                )
            )
        )

        x = (
            token_embeddings
            + position_embeddings
        )

        x = self.blocks(x)

        logits = self.ln_head(x)

        # Loss calculations
        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)

            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        else:
            loss = None

        return logits, loss
