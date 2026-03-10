import argparse
import logging
import os
import random
from datetime import datetime

import torch
import torch.distributed as dist
from datasets import interleave_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from huggingface_hub import snapshot_download
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

from model import GPTModel
from tokenizer import GPTTokenizer


DEFAULT_HF_REPO = "CuriousDragon/gpt-tinystories"
DEFAULT_HF_FILE = "gpt_model_v2.pt"


def setup_logging(log_dir, rank):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"bridge_train_rank_{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"bridge_train_{timestamp}.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def init_distributed():
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
    return is_distributed, rank, world_size, device


def strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key.replace("module.", "", 1)] = value
        else:
            cleaned[key] = value
    return cleaned


def load_initial_checkpoint(args, device, is_distributed, rank, logger):
    checkpoint_path = args.resume_from

    if not checkpoint_path and args.load_from_hf:
        hf_path = None
        if rank == 0:
            logger.info(
                "No local checkpoint provided. Fetching %s from %s",
                args.hf_filename,
                args.hf_repo,
            )
            model_dir = snapshot_download(
                repo_id=args.hf_repo,
                allow_patterns=[args.hf_filename],
                token=os.getenv("HF_TOKEN"),
            )
            hf_path = os.path.join(model_dir, args.hf_filename)

        if is_distributed:
            path_list = [hf_path]
            dist.broadcast_object_list(path_list, src=0)
            hf_path = path_list[0]

        checkpoint_path = hf_path

    if not checkpoint_path:
        return None

    payload = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in payload:
        payload["model_state_dict"] = strip_module_prefix(payload["model_state_dict"])
    elif isinstance(payload, dict):
        payload = strip_module_prefix(payload)
    return payload


def get_text(example):
    for key in ("text", "article", "content", "markdown"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class MixedTextChunkDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, seq_len):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eot_id = tokenizer.special_tokens["<|endoftext|>"]

    def __iter__(self):
        buffer = []

        for example in self.hf_dataset:
            text = get_text(example)
            if not text:
                continue

            token_ids = self.tokenizer.encode(text)
            if not token_ids:
                continue

            buffer.extend(token_ids)
            buffer.append(self.eot_id)

            while len(buffer) > self.seq_len:
                chunk = buffer[: self.seq_len + 1]
                if len(chunk) < self.seq_len + 1:
                    break
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, labels
                buffer = buffer[self.seq_len :]


def build_bridge_stream(seed, rank, world_size, logger):
    stories = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    simplewiki = load_dataset(
        "HuggingFaceTB/simplewiki-pruned-350k", split="train", streaming=True
    )
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    stories = stories.shuffle(seed=seed, buffer_size=5_000)
    simplewiki = simplewiki.shuffle(seed=seed + 1, buffer_size=5_000)
    fineweb = fineweb.shuffle(seed=seed + 2, buffer_size=10_000)

    blended = interleave_datasets(
        [simplewiki, fineweb, stories],
        probabilities=[0.50, 0.30, 0.20],
        seed=seed,
        stopping_strategy="first_exhausted",
    )

    if world_size > 1:
        blended = split_dataset_by_node(blended, rank=rank, world_size=world_size)

    if rank == 0:
        logger.info(
            "Bridge mix configured: 50%% SimpleWiki, 30%% FineWeb-Edu sample-10BT, 20%% TinyStories replay"
        )

    return blended


def calc_loss(model, input_ids, labels, device, use_amp):
    input_ids = input_ids.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    with autocast("cuda", enabled=use_amp and device.type == "cuda"):
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            labels.flatten(),
        )
    return loss


def save_checkpoint(model, optimizer, args, save_name, step, logger, rank):
    if rank != 0:
        return

    os.makedirs(args.save_dir, exist_ok=True)
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    checkpoint = {
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
    }
    path = os.path.join(args.save_dir, save_name)
    torch.save(checkpoint, path)
    logger.info("Saved checkpoint: %s", path)


def build_model(tokenizer):
    return GPTModel(
        vocab_size=tokenizer.vocab_size,
        emb_dim=768,
        context_length=1024,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        qkv_bias=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Bridge pretraining for general knowledge.")
    parser.add_argument("--save-dir", default="checkpoints_bridge")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--load-from-hf", action="store_true")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-filename", default=DEFAULT_HF_FILE)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--use-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    is_distributed, rank, world_size, device = init_distributed()
    logger = setup_logging(args.log_dir, rank)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    tokenizer = GPTTokenizer()
    model = build_model(tokenizer)

    initial_payload = load_initial_checkpoint(args, device, is_distributed, rank, logger)
    if initial_payload is not None:
        state_dict = (
            initial_payload["model_state_dict"]
            if "model_state_dict" in initial_payload
            else initial_payload
        )
        model.load_state_dict(state_dict)
        if rank == 0:
            logger.info("Loaded initial weights successfully.")

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if (
        initial_payload is not None
        and isinstance(initial_payload, dict)
        and "optimizer_state_dict" in initial_payload
        and args.resume_from
    ):
        try:
            optimizer.load_state_dict(initial_payload["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
            if rank == 0:
                logger.info("Loaded optimizer state from resume checkpoint.")
        except Exception as exc:
            if rank == 0:
                logger.warning("Skipping optimizer state restore: %s", exc)

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    stream = build_bridge_stream(args.seed, rank, world_size, logger)
    train_dataset = MixedTextChunkDataset(stream, tokenizer, seq_len=args.seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    scaler = GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    micro_step = 0
    train_step = 0
    running_loss = 0.0

    if rank == 0:
        logger.info("Starting bridge pretraining on %d device(s)", world_size)
        logger.info(
            "Config: seq_len=%d batch_size=%d grad_accum=%d max_steps=%d lr=%g",
            args.seq_len,
            args.batch_size,
            args.grad_accum,
            args.max_steps,
            args.learning_rate,
        )

    while train_step < args.max_steps:
        for input_ids, labels in train_loader:
            loss = calc_loss(model, input_ids, labels, device, args.use_amp)
            loss = loss / args.grad_accum
            running_loss += loss.item()

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_step += 1
            if micro_step % args.grad_accum != 0:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            train_step += 1

            if rank == 0 and train_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                logger.info("Bridge step %d/%d | loss=%.4f", train_step, args.max_steps, avg_loss)
                running_loss = 0.0

            if train_step % args.save_every == 0:
                save_checkpoint(model, optimizer, args, "bridge_latest.pt", train_step, logger, rank)

            if train_step >= args.max_steps:
                break

        if train_step >= args.max_steps:
            break

    save_checkpoint(model, optimizer, args, "bridge_latest.pt", train_step, logger, rank)
    if rank == 0:
        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        final_path = os.path.join(args.save_dir, "gpt_model_bridge.pt")
        torch.save(model_state, final_path)
        logger.info("Saved final bridge model to %s", final_path)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
