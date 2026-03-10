import argparse
import json
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from model import GPTModel
from tokenizer import GPTTokenizer


MODEL_SPECS = {
    "base": {
        "repo_id": "CuriousDragon/gpt-tinystories",
        "filename": "gpt_model_v2.pt",
    },
    "bridge": {
        "repo_id": "CuriousDragon/gpt-tinystories-bridge",
        "filename": "gpt_model_bridge.pt",
    },
    "instruct": {
        "repo_id": "CuriousDragon/gpt-tinystories-instruct",
        "filename": "gpt_model_sft.pt",
    },
}


PROMPTS = [
    {
        "id": "story_continuation",
        "description": "Plain continuation prompt to see what story fluency survived.",
        "prompt": "Once upon a time in a small village, a child found a strange glowing key",
        "max_new_tokens": 100,
        "temperature": 0.8,
        "top_k": 40,
    },
    {
        "id": "grounded_qa",
        "description": "Instruction-style grounded QA prompt aligned with SFT formatting.",
        "prompt": (
            "### Instruction:\n"
            "Answer the question using only the context. If the answer is not stated, reply with I don't know.\n\n"
            "### Context:\n"
            "Mercury is the closest planet to the Sun. Venus is the hottest planet because its atmosphere traps heat.\n\n"
            "### Question:\n"
            "Which planet is closest to the Sun?\n\n"
            "### Response:\n"
        ),
        "max_new_tokens": 60,
        "temperature": 0.2,
        "top_k": 20,
    },
    {
        "id": "summarization",
        "description": "Simple summarization prompt aligned with the SFT template.",
        "prompt": (
            "### Instruction:\n"
            "Summarize the text in one short sentence.\n\n"
            "### Input:\n"
            "Ravi planted tomato seeds in a small garden behind his house. He watered them every morning, removed weeds, and protected the plants from pests. After two months, the plants produced bright red tomatoes that he shared with his neighbors.\n\n"
            "### Response:\n"
        ),
        "max_new_tokens": 60,
        "temperature": 0.2,
        "top_k": 20,
    },
    {
        "id": "unanswerable_qa",
        "description": "Checks whether the model can decline when the context is insufficient.",
        "prompt": (
            "### Instruction:\n"
            "Answer the question using only the context. If the answer is not stated, reply with I don't know.\n\n"
            "### Context:\n"
            "The museum opens at 9 AM and closes at 5 PM from Tuesday to Sunday.\n\n"
            "### Question:\n"
            "Who founded the museum?\n\n"
            "### Response:\n"
        ),
        "max_new_tokens": 40,
        "temperature": 0.2,
        "top_k": 20,
    },
]


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


def clean_state_dict(payload):
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key.replace("module.", "", 1)] = value
        else:
            cleaned[key] = value
    return cleaned


def load_model(model_key, tokenizer, device, token):
    spec = MODEL_SPECS[model_key]
    path = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        token=token,
    )
    payload = torch.load(path, map_location=device)
    model = build_model(tokenizer)
    model.load_state_dict(clean_state_dict(payload))
    model.to(device)
    model.eval()
    return model, path


def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k):
    eot_id = tokenizer.special_tokens["<|endoftext|>"]
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_tensor)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < values[:, [-1]]] = float("-inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

            if next_token.item() == eot_id:
                break

    full_text = tokenizer.decode(input_tensor[0].tolist())
    completion = full_text[len(prompt):]
    return completion.strip()


def run_evaluation(device_name, output_path):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = GPTTokenizer()

    print(f"Using device: {device}")
    print(f"Using HF token: {'yes' if hf_token else 'no'}")

    models = {}
    for key in MODEL_SPECS:
        print(f"Loading {key} model from {MODEL_SPECS[key]['repo_id']}/{MODEL_SPECS[key]['filename']}")
        model, local_path = load_model(key, tokenizer, device, hf_token)
        models[key] = {"model": model, "path": local_path}

    results = {
        "device": str(device),
        "models": {key: {"repo_id": MODEL_SPECS[key]["repo_id"], "filename": MODEL_SPECS[key]["filename"], "local_path": value["path"]} for key, value in models.items()},
        "prompts": [],
    }

    for prompt_cfg in PROMPTS:
        print(f"\n=== Prompt: {prompt_cfg['id']} ===")
        prompt_result = {
            "id": prompt_cfg["id"],
            "description": prompt_cfg["description"],
            "prompt": prompt_cfg["prompt"],
            "outputs": {},
        }

        for model_key, bundle in models.items():
            start = time.time()
            completion = generate_text(
                model=bundle["model"],
                tokenizer=tokenizer,
                prompt=prompt_cfg["prompt"],
                max_new_tokens=prompt_cfg["max_new_tokens"],
                temperature=prompt_cfg["temperature"],
                top_k=prompt_cfg["top_k"],
            )
            duration = time.time() - start
            prompt_result["outputs"][model_key] = {
                "text": completion,
                "seconds": round(duration, 2),
            }
            print(f"\n[{model_key}] ({duration:.2f}s)\n{completion}\n")

        results["prompts"].append(prompt_result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare base, bridge, and instruct models from Hugging Face.")
    parser.add_argument("--device", default="", help="Device override, for example 'cpu' or 'cuda'.")
    parser.add_argument(
        "--output",
        default="eval_results/model_comparison.json",
        help="Where to save the JSON results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.device, Path(args.output))
