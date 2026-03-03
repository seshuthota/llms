import torch
import torch.nn as nn
from embeddings import TokenPosEmbedding
from transformer import TransformerBlock, LayerNorm


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        context_length,
        num_heads,
        num_layers,
        dropout,
        qkv_bias=False,
    ):
        super().__init__()

        self.embedding_layer = TokenPosEmbedding(
            vocab_size=vocab_size, emb_dim=emb_dim, max_length=context_length
        )

        self.drop_emb = nn.Dropout(dropout)

        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    emb_dim=emb_dim,
                    context_length=context_length,
                    num_heads=num_heads,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = LayerNorm(emb_dim)

        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding_layer(input_ids)
        x = self.drop_emb(x)

        x = self.trf_blocks(x)

        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits


if __name__ == "__main__":
    torch.manual_seed(123)

    model = GPTModel(
        vocab_size=50257,
        emb_dim=768,
        context_length=1024,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        qkv_bias=True,
    )

    x = torch.randint(0, 50257, (1, 10))
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
