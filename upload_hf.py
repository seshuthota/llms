from huggingface_hf import HfApi, login
import os
import torch
from model import GPTModel
from tokenizer import GPTTokenizer

# Login with your token (get from https://huggingface.co/settings/tokens)
token = "YOUR_HF_TOKEN"  # Replace with your token
login(token)

api = HfApi()

# Your repo name
repo_id = "seshuthota/gpt-tinystories"  # Change to your desired repo name

# Create repo if not exists
try:
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"Repo created/verified: {repo_id}")
except Exception as e:
    print(f"Error creating repo: {e}")

# Upload model files
api.upload_file(
    path_or_fileobj="gpt_model.pt",
    path_in_repo="pytorch_model.pt",
    repo_id=repo_id,
    repo_type="model",
)
print(f"Model uploaded to https://huggingface.co/{repo_id}")

# Also upload config info
config = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "context_length": 1024,
    "num_heads": 12,
    "num_layers": 12,
    "dropout": 0.1,
}

import json

with open("config.json", "w") as f:
    json.dump(config, f)

api.upload_file(
    path_or_fileobj="config.json",
    path_in_repo="config.json",
    repo_id=repo_id,
    repo_type="model",
)
print(f"Config uploaded!")
print(f"\nModel available at: https://huggingface.co/{repo_id}")
