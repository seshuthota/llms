import argparse
import logging
import os
import random
from datetime import datetime

import torch
import torch.distributed as dist
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from model import GPTModel
from tokenizer import GPTTokenizer


DEFAULT_BRIDGE_REPO = "CuriousDragon/gpt-tinystories"
DEFAULT_BRIDGE_FILE = "gpt_model_v2.pt"


def setup_logging(log_dir, rank):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"sft_train_rank_{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"sft_train_{timestamp}.log")
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


def load_initial_checkpoint(args, device, is_distributed, rank, logger):
    checkpoint_path = args.resume_from

    if not checkpoint_path and args.load_from_hf:
        hf_path = None
        if rank == 0:
            logger.info(
                "Fetching initial SFT weights %s from %s",
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


def format_squad_prompt(context, question):
    return (
        "### Instruction:\n"
        "Answer the question using only the context. If the answer is not stated, reply with I don't know.\n\n"
        f"### Context:\n{context.strip()}\n\n"
        f"### Question:\n{question.strip()}\n\n"
        "### Response:\n"
    )


def format_dolly_prompt(instruction, context):
    parts = [
        "### Instruction:",
        instruction.strip(),
        "",
    ]
    if context and context.strip():
        parts.extend(["### Input:", context.strip(), ""])
    parts.extend(["### Response:", ""])
    return "\n".join(parts)


def build_sft_examples(max_squad_samples, max_dolly_samples, seed, logger, rank):
    random.seed(seed)

    squad = load_dataset("rajpurkar/squad_v2", split="train")
    squad = squad.shuffle(seed=seed)
    if max_squad_samples:
        squad = squad.select(range(min(max_squad_samples, len(squad))))

    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    allowed_categories = {
        "closed_qa",
        "open_qa",
        "information_extraction",
        "summarization",
    }
    dolly = dolly.filter(lambda row: row["category"] in allowed_categories)
    dolly = dolly.shuffle(seed=seed + 1)
    if max_dolly_samples:
        dolly = dolly.select(range(min(max_dolly_samples, len(dolly))))

    examples = []
    squad_count = 0
    dolly_count = 0

    for row in squad:
        answers = row.get("answers", {}).get("text", [])
        answer = answers[0].strip() if answers else "I don't know."
        examples.append(
            {
                "prompt": format_squad_prompt(row["context"], row["question"]),
                "response": answer,
                "source": "squad",
            }
        )
        squad_count += 1

    for row in dolly:
        instruction = row.get("instruction", "").strip()
        response_text = row.get("response", "").strip()
        if not instruction or not response_text:
            continue

        examples.append(
            {
                "prompt": format_dolly_prompt(instruction, row.get("context", "")),
                "response": response_text,
                "source": "dolly",
            }
        )
        dolly_count += 1

    random.shuffle(examples)
    if rank == 0:
        logger.info(
            "Prepared %d SFT examples (%d SQuAD v2, %d filtered Dolly)",
            len(examples),
            squad_count,
            dolly_count,
        )
    return examples


class InstructionDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eot_id = tokenizer.special_tokens["<|endoftext|>"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        prompt_tokens = self.tokenizer.encode(example["prompt"])
        response_tokens = self.tokenizer.encode(example["response"]) + [self.eot_id]

        max_response_tokens = max(self.max_length // 2, 64)
        if len(response_tokens) > max_response_tokens:
            response_tokens = response_tokens[:max_response_tokens]

        available_prompt = max(self.max_length - len(response_tokens), 1)
        if len(prompt_tokens) > available_prompt:
            prompt_tokens = prompt_tokens[-available_prompt:]

        input_ids = prompt_tokens + response_tokens
        labels = ([-100] * len(prompt_tokens)) + response_tokens

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_batch(batch, pad_token_id):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        padded_input = torch.nn.functional.pad(
            item["input_ids"], (0, pad_len), value=pad_token_id
        )
        padded_labels = torch.nn.functional.pad(
            item["labels"], (0, pad_len), value=-100
        )
        input_ids.append(padded_input)
        labels.append(padded_labels)

    return torch.stack(input_ids), torch.stack(labels)


def calc_loss(model, input_ids, labels, device, use_amp):
    input_ids = input_ids.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    with autocast("cuda", enabled=use_amp and device.type == "cuda"):
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            labels.flatten(),
            ignore_index=-100,
        )
    return loss


def save_checkpoint(model, optimizer, args, save_name, epoch, logger, rank):
    if rank != 0:
        return

    os.makedirs(args.save_dir, exist_ok=True)
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
    }
    path = os.path.join(args.save_dir, save_name)
    torch.save(checkpoint, path)
    logger.info("Saved checkpoint: %s", path)


def parse_args():
    parser = argparse.ArgumentParser(description="Instruction SFT for the TinyStories GPT model.")
    parser.add_argument("--save-dir", default="checkpoints_sft")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--load-from-hf", action="store_true")
    parser.add_argument("--hf-repo", default=DEFAULT_BRIDGE_REPO)
    parser.add_argument("--hf-filename", default=DEFAULT_BRIDGE_FILE)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-squad-samples", type=int, default=12000)
    parser.add_argument("--max-dolly-samples", type=int, default=6000)
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
            logger.info("Loaded initial weights successfully for SFT.")

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    examples = build_sft_examples(
        max_squad_samples=args.max_squad_samples,
        max_dolly_samples=args.max_dolly_samples,
        seed=args.seed,
        logger=logger,
        rank=rank,
    )
    train_dataset = InstructionDataset(examples, tokenizer, max_length=args.max_length)
    sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=lambda batch: collate_batch(
            batch, tokenizer.special_tokens["<|endoftext|>"]
        ),
    )

    scaler = GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    if rank == 0:
        logger.info("Starting SFT on %d device(s)", world_size)
        logger.info(
            "Config: max_length=%d batch_size=%d grad_accum=%d epochs=%d lr=%g",
            args.max_length,
            args.batch_size,
            args.grad_accum,
            args.epochs,
            args.learning_rate,
        )

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        micro_step = 0

        for batch_idx, (input_ids, labels) in enumerate(train_loader, start=1):
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

            if rank == 0 and batch_idx % 50 == 0:
                avg_loss = running_loss / 50
                logger.info(
                    "SFT epoch %d/%d | batch %d/%d | loss=%.4f",
                    epoch + 1,
                    args.epochs,
                    batch_idx,
                    len(train_loader),
                    avg_loss,
                )
                running_loss = 0.0

        if args.save_every_epoch:
            save_checkpoint(model, optimizer, args, f"sft_epoch_{epoch + 1}.pt", epoch + 1, logger, rank)

    save_checkpoint(model, optimizer, args, "sft_latest.pt", args.epochs, logger, rank)
    if rank == 0:
        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        final_path = os.path.join(args.save_dir, "gpt_model_sft.pt")
        torch.save(model_state, final_path)
        logger.info("Saved final SFT model to %s", final_path)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
