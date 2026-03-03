import torch
from torch.cuda.amp import autocast, GradScaler
from tokenizer import GPTTokenizer
from dataset import create_dataloader_from_huggingface
from model import GPTModel
from datasets import load_dataset
import os
import logging
from datetime import datetime


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def calc_loss_batch(input_batch, target_batch, model, device, use_amp=False):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    with autocast(enabled=use_amp):
        logits = model(input_batch)
        logits_flat = logits.flatten(0, 1)
        targets_flat = target_batch.flatten()
        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

    return loss


def evaluate_model(model, train_loader, device, num_batches=None):
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(train_loader):
            if num_batches and i >= num_batches:
                break
            loss = calc_loss_batch(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device,
            )
            loss_sum += loss.item()
    if num_batches is None:
        return loss_sum / len(train_loader)
    return loss_sum / num_batches


def train_model(
    model,
    train_loader,
    optimizer,
    device,
    num_epochs,
    print_every=10,
    logger=None,
    save_dir="checkpoints",
    use_amp=True,
):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)

    scaler = GradScaler() if use_amp else None

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()

    global_step = 0
    history = {"epoch": [], "train_loss": [], "lr": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = datetime.now()

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            loss = calc_loss_batch(input_batch, target_batch, model, device, use_amp)

            optimizer.zero_grad()

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            global_step += 1
            epoch_loss += loss.item()

            if global_step % print_every == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                lr = optimizer.param_groups[0]["lr"]
                msg = f"Epoch {epoch + 1}/{num_epochs} | Step {global_step} | Batch loss: {loss.item():.4f} | Avg loss: {avg_loss:.4f} | LR: {lr:.2e}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)

        avg_epoch_loss = epoch_loss / len(train_loader)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_epoch_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        msg = f"Epoch {epoch + 1}/{num_epochs} completed | Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.1f}s"
        if logger:
            logger.info(msg)
        else:
            print(msg)

        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss,
                "history": history,
            },
            checkpoint_path,
        )
        if logger:
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    print("Training finished.")
    return model, history


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("=" * 50)
    logger.info("Starting training pipeline")
    logger.info("=" * 50)

    logger.info("Loading TinyStories dataset from HuggingFace...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    max_samples = None  # Use None for full dataset, or set to integer for subset
    dataset_subset = (
        dataset if max_samples is None else dataset.select(range(max_samples))
    )
    logger.info(
        f"Using {'full dataset' if max_samples is None else f'first {max_samples} samples'}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    logger.info("Initializing tokenizer...")
    tokenizer = GPTTokenizer()
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocabulary size: {vocab_size}")

    train_loader = create_dataloader_from_huggingface(
        dataset_subset,
        tokenizer,
        batch_size=8,
        max_length=1024,
        stride=128,
        shuffle=True,
    )

    model = GPTModel(
        vocab_size=vocab_size,
        emb_dim=768,
        context_length=1024,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        qkv_bias=True,
    )
    logger.info(
        f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    logger.info(f"Optimizer: AdamW (lr=3e-4, weight_decay=0.1)")

    num_epochs = 5
    print_every = 50

    logger.info(f"Training config: epochs={num_epochs}, batch_size=8, max_length=1024")
    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        print_every=print_every,
        logger=logger,
    )

    torch.save(model.state_dict(), "gpt_model.pt")
    logger.info("Final model saved to gpt_model.pt")

    logger.info(f"Training history: {history}")
