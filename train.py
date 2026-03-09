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
from huggingface_hub import snapshot_download
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


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
    save_every_steps=500,
    is_distributed=False,
    rank=0,
):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[rank])

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
                if logger and rank == 0:
                    logger.info(msg)
                elif rank == 0:
                    print(msg)

            if global_step % save_every_steps == 0 and rank == 0:
                checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "model_state_dict": model.module.state_dict() if is_distributed else model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item() * gradient_accumulation_steps,
                        "history": history,
                    },
                    checkpoint_path,
                )
                if logger:
                    logger.info(f"Intermediate checkpoint saved at step {global_step}")

        avg_epoch_loss = epoch_loss / max(batch_idx + 1, 1)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_epoch_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        msg = f"Epoch {epoch + 1}/{num_epochs} completed | Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.1f}s"
        if logger and rank == 0:
            logger.info(msg)
        elif rank == 0:
            print(msg)

        if rank == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict() if is_distributed else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "history": history,
                },
                checkpoint_path,
            )
            if logger:
                logger.info(f"Checkpoint saved: {checkpoint_path}")

    if rank == 0:
        print("Training finished.")
    return model, history


if __name__ == "__main__":
    # Check for DDP environment variables
    is_distributed = "WORLD_SIZE" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logging() if rank == 0 else None
    
    if rank == 0:
        logger.info("=" * 50)
        logger.info("Starting training pipeline")
        if is_distributed:
            logger.info(f"Distributed training enabled with {world_size} GPUs")
        logger.info("=" * 50)

        logger.info("Loading TinyStories dataset from HuggingFace...")
        
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    if rank == 0:
        logger.info("Dataset loaded in streaming mode")

    max_samples = None  # Use None for full dataset, or set to integer for subset

    use_streaming = True
    if use_streaming and rank == 0:
        logger.info("Using streaming mode for memory efficiency")

    if rank == 0:
        logger.info(f"Using device: {device}")
        if device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

    tokenizer = GPTTokenizer()
    vocab_size = tokenizer.vocab_size
    if rank == 0:
        logger.info(f"Vocabulary size: {vocab_size}")

    train_loader = create_dataloader_from_huggingface(
        dataset,
        tokenizer,
        batch_size=1,  # Small batch for memory
        max_length=1024,
        stride=128,
        shuffle=(not is_distributed), # DistributedSampler handles shuffling
        streaming=True,
        max_samples=100000,
        sampler=DistributedSampler(dataset) if is_distributed else None,
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
    if rank == 0:
        logger.info(
            f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters"
        )

    # Initialize optimizer first
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    # Check for existing checkpoint to resume training
    save_dir = "checkpoints"
    resume_checkpoint = None
    
    if rank == 0:
        if os.path.exists(save_dir):
            checkpoints = sorted([
                f for f in os.listdir(save_dir) if f.startswith("checkpoint_") and f.endswith(".pt")
            ])
            if checkpoints:
                # Prioritize 'checkpoint_latest.pt' if it exists
                if "checkpoint_latest.pt" in checkpoints:
                    resume_checkpoint = os.path.join(save_dir, "checkpoint_latest.pt")
                else:
                    resume_checkpoint = os.path.join(save_dir, checkpoints[-1])

    # Broadcast checkpoint path to all processes
    if is_distributed:
        checkpoint_path_list = [resume_checkpoint]
        dist.broadcast_object_list(checkpoint_path_list, src=0)
        resume_checkpoint = checkpoint_path_list[0]

    start_epoch = 0
    if resume_checkpoint:
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except Exception as e:
                if rank == 0:
                    logger.warning(f"Failed to load optimizer state: {e}. Starting with fresh optimizer.")
        
        start_epoch = checkpoint.get("epoch", 0)
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}")
    else:
        if rank == 0:
            logger.info("No local checkpoint found. Attempting to fetch base weights from Hugging Face...")
        hf_token = os.getenv("HF_TOKEN")
        try:
            if rank == 0:
                model_path = snapshot_download(
                    repo_id="CuriousDragon/gpt-tinystories",
                    allow_patterns=["pytorch_model.pt"],
                    token=hf_token,
                )
                hf_weights_path = f"{model_path}/pytorch_model.pt"
            else:
                hf_weights_path = None
            
            if is_distributed:
                path_list = [hf_weights_path]
                dist.broadcast_object_list(path_list, src=0)
                hf_weights_path = path_list[0]

            state_payload = torch.load(hf_weights_path, map_location=device)
            if "model_state_dict" in state_payload:
                model.load_state_dict(state_payload["model_state_dict"])
            else:
                model.load_state_dict(state_payload)
                
            if rank == 0:
                logger.info("Successfully loaded base weights from Hugging Face!")
        except Exception as e:
            if rank == 0:
                logger.warning(f"Failed to load weights from Hugging Face: {e}")
                logger.info("Starting training entirely from scratch.")

    num_epochs = 10
    print_every = 100
    gradient_accumulation_steps = 8  # Effective batch = 1 × 8 × GPUs

    if rank == 0:
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
        is_distributed=is_distributed,
        rank=rank,
    )

    if rank == 0:
        torch.save(model.state_dict(), "gpt_model.pt")
        logger.info("Final model saved to gpt_model.pt")
        logger.info(f"Training history: {history}")

    if is_distributed:
        dist.destroy_process_group()
