import torch
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os
import time

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in .env file!")

try:
    from model import GPTModel
    from tokenizer import GPTTokenizer
    import generate
except ImportError:
    print(
        "Please run this folder inside the exact directory containing model.py and tokenizer.py"
    )
    exit(1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing environment utilizing: {device.type.upper()}")

    print("\n--- Initializing Blank Model Architecture ---")
    tokenizer = GPTTokenizer()
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        emb_dim=768,
        context_length=1024,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        qkv_bias=True,
    )

    print("--- Fetching checkpoint_epoch_3.pt via HF API ---")
    print(f"Using token: {'Yes' if HF_TOKEN else 'No (anonymous - will be slow)'}")

    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        repo_id="CuriousDragon/gpt-tinystories",
        allow_patterns=["checkpoint_epoch_3.pt"],
        token=HF_TOKEN,
    )
    path = f"{model_path}/checkpoint_epoch_3.pt"

    print("\n--- Binding Weights... ---")
    state_payload = torch.load(path, map_location=device)
    # Flexible Dict-mapping to handle any saved config structure
    if "model_state_dict" in state_payload:
        model.load_state_dict(state_payload["model_state_dict"])
    else:
        model.load_state_dict(state_payload)

    model.to(device)
    model.eval()

    prompts = ["Once upon a time", "The little dog", "She went to the park and"]

    print("\n --- Starting Generation Inference ---")
    for prompt_txt in prompts:
        print(f"\nPrompt: {prompt_txt}")
        print("-" * 50)
        t_start = time.time()

        # Triggering the native imported functional generation methodology
        result_text = generate.generate_text(
            model,
            tokenizer,
            prompt=prompt_txt,
            max_new_tokens=80,
            temperature=0.7,
            top_k=40,
        )

        t_duration = time.time() - t_start
        print(result_text)
        print(f"\n>>> [Fast Inferred in {t_duration:.2f}s]")


if __name__ == "__main__":
    # Wrap standard PyTorch Warnings temporarily!
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
