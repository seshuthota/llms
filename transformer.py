import torch
import torch.nn as nn
from attention import MultiHeadAttention


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=True)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        hidden_dim = 4 * embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            GELU(),
            nn.Linear(hidden_dim, embed_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            dropdout=dropout,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.ffn = FeedForward(embed_dim=emb_dim)

        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim=emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.att(self.norm1(x))
        attn_out = self.dropout(attn_out)

        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        ffn_out = self.dropout(ffn_out)

        x = x + ffn_out
        return x


if __name__ == "__main__":
    torch.manual_seed(123)
    batch_size = 2
    context_length = 16
    emb_dim = 16
    num_heads = 4
    dropout = 0.1
    x = torch.randn(batch_size, context_length, emb_dim)

    block = TransformerBlock(
        emb_dim=emb_dim,
        context_length=context_length,
        num_heads=num_heads,
        dropout=dropout,
    )

    out = block(x)
    print(f"Output shape from Transformer Block: {out.shape}")
    print(f"Expected output shape: ({batch_size}, {context_length}, {emb_dim})")

    print("\nTransformer Block instantiated and executed successfully!")
    print(
        "Notice that the sequence length and embedding dimensions are completely preserved."
    )
