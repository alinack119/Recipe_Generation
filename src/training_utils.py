# Code for batches, loss reporting, training, and generation
# training and inference utilities

import torch
import torch.nn.functional as F


# These will be assigned by train.py 
train_data = None
val_data = None
device = None


def setup_data(training_data, validation_data, selected_device):
    
    global train_data
    global val_data
    global device

    train_data = training_data
    val_data = validation_data
    device = selected_device


# Generate a small batch of input and target sequences
def get_batch(split, batch_size, context_size):

    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - context_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + context_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + context_size + 1]
        for i in ix
    ])

    x = x.to(device)
    y = y.to(device)

    return x, y


# Calculate average loss on training and validation data
@torch.no_grad()
def estimate_loss(
    model,
    batch_size,
    context_size,
    eval_iters=100
):


    out = {}

    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(
                split,
                batch_size,
                context_size
            )

            logits, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out

# Main optimization loop training the model
def train(
    model,
    steps,
    batch_size,
    context_size,
    learning_rate=1e-3,
    report_frequency=1000
):


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    for step in range(steps):
        # Get random training batch
        xb, yb = get_batch(
            "train",
            batch_size,
            context_size
        )

        # Run model
        logits, loss = model(xb, yb)

        # Reset old gradients
        optimizer.zero_grad(
            set_to_none=True
        )

        # Calculate new gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        if (
            step % report_frequency == 0
            or step == steps - 1
        ):
            losses = estimate_loss(
                model,
                batch_size,
                context_size
            )

            print(
                f"Step {step}, "
                f"train loss: {losses['train']:.4f}, "
                f"val loss: {losses['val']:.4f}"
            )


# Generate new tokens one at a time
@torch.no_grad()
def generate(
    model,
    context_size,
    start_idx,
    number_of_tokens,
    stop_token_id=None
):

    idx = start_idx

    for _ in range(number_of_tokens):
        idx_cond = idx[:, -context_size:]

        # Get model predictions
        logits, loss = model(idx_cond)

        # Prediction for final position
        logits = logits[:, -1, :]

        # Convert predictions to probabilities
        probabilities = F.softmax(
            logits,
            dim=1
        )

        # Randomly select the next token
        idx_next = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Add the token to the generated sequence
        idx = torch.cat(
            (idx, idx_next),
            dim=1
        )

        # Stop if <END_RECIPE> is generated
        if stop_token_id is not None:
            if idx_next.item() == stop_token_id:
                break

    return idx
