import torch
from torch.amp import autocast, GradScaler
from tokenizer import GPTTokenizer
from dataset import create_dataloader_from_huggingface
from model import GPTModel
from datasets import load_dataset
import os
import logging
from datetime import datetime
import json


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

    with autocast("cuda", enabled=use_amp):
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
    gradient_accumulation_steps=1,
    start_epoch=0,
):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)

    scaler = GradScaler("cuda") if use_amp else None

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()

    global_step = 0
    history = {"epoch": [], "train_loss": [], "lr": []}

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = datetime.now()
        optimizer.zero_grad()

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            loss = calc_loss_batch(input_batch, target_batch, model, device, use_amp)
            loss = loss / gradient_accumulation_steps

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item() * gradient_accumulation_steps

            if global_step % print_every == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                lr = optimizer.param_groups[0]["lr"]
                msg = f"Epoch {epoch + 1}/{num_epochs} | Step {global_step} | Batch loss: {loss.item() * gradient_accumulation_steps:.4f} | Avg loss: {avg_loss:.4f} | LR: {lr:.2e}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)

        avg_epoch_loss = epoch_loss / max(batch_idx + 1, 1)
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
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    logger.info("Dataset loaded in streaming mode")

    max_samples = None  # Use None for full dataset, or set to integer for subset

    use_streaming = True
    if use_streaming:
        logger.info("Using streaming mode for memory efficiency")

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
        dataset,
        tokenizer,
        batch_size=1,  # Small batch for memory
        max_length=1024,
        stride=128,
        shuffle=True,
        streaming=True,
        max_samples=100000,
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

    # Initialize optimizer first
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    logger.info(f"Optimizer: AdamW (lr=3e-4, weight_decay=0.1)")

    # Check for existing checkpoint to resume training
    save_dir = "checkpoints"
    resume_checkpoint = None
    if os.path.exists(save_dir):
        checkpoints = [
            f for f in os.listdir(save_dir) if f.startswith("checkpoint_epoch_")
        ]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            resume_checkpoint = os.path.join(save_dir, latest_checkpoint)

    start_epoch = 0
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Move optimizer state tensors to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resumed from epoch {start_epoch}")
    else:
        logger.info("Starting training from scratch")

    num_epochs = 10
    print_every = 100
    gradient_accumulation_steps = 8  # Effective batch = 1 × 8 = 8

    logger.info(
        f"Training config: epochs={num_epochs}, batch_size=1, grad_accum={gradient_accumulation_steps}, max_length=1024"
    )
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
        start_epoch=start_epoch,
    )

    torch.save(model.state_dict(), "gpt_model.pt")
    logger.info("Final model saved to gpt_model.pt")

    logger.info(f"Training history: {history}")
