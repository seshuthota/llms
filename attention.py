import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropdout, num_heads, qkv_bias=False
    ):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimension of each head

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropdout)

    def forward(self, x, verbose=False):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys_reshaped = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries_reshaped = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values_reshaped = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys_transposed = keys_reshaped.transpose(1, 2)
        queries_transposed = queries_reshaped.transpose(1, 2)
        values_transposed = values_reshaped.transpose(1, 2)

        attn_scores = queries_transposed @ keys_transposed.transpose(2, 3)

        attn_scores = attn_scores / math.sqrt(self.head_dim)

        attn_scores = attn_scores.masked_fill(
            torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
                diagonal=1,
            ),
            -torch.inf,
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values_transposed

        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        if verbose:
            print(f"Final output shape: {context_vec.shape}\n")

        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)
    batch_size = 2
    context_length = 16
    d_in = 16
    d_out = 16
    num_heads = 8

    x = torch.randn(batch_size, context_length, d_in)

    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropdout=0.0,
        num_heads=num_heads,
    )

    context_vecs = mha(x, verbose=True)
    print("Multi-Head Attention created successfully.\n")
    print(f"Input shape (x): {x.shape}")
    print(f"Output shape (context_vecs): {context_vecs.shape}")
    print(f"Expected output shape: ({batch_size}, {context_length}, {d_out})")
    print(
        "\nWe can observe that the context vector shape matches the input sequence shape."
    )
