import torch
from model import GPTModel
from tokenizer import GPTTokenizer


def generate_text(
    model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None
):
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPTTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        emb_dim=768,
        context_length=1024,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        qkv_bias=True,
    )

    checkpoint_path = "gpt_model.pt"  # or checkpoints/checkpoint_epoch_X.pt

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Training a fresh model for demo...")

    model.to(device)

    prompt = "Once upon a time"
    print(f"\nGenerating text with prompt: '{prompt}'")
    print("-" * 50)

    generated_text = generate_text(
        model, tokenizer, prompt, max_new_tokens=50, temperature=0.8
    )
    print(generated_text)
