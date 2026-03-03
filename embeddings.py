import torch
import torch.nn as nn

class TokenPosEmbedding(nn.Module):

    def __init__(self, vocab_size ,emb_dim ,max_length):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size , emb_dim)
        self.pos_emb = nn.Embedding(max_length, emb_dim)

    
    def forward(self, input_ids):

        batch_size, seq_length = input_ids.shape

        tok_embeddings = self.token_emb(input_ids)

        pos_indices = torch.arange(seq_length, device=input_ids.device)

        pos_embeddings = self.pos_emb(pos_indices)

        final_embeddings = tok_embeddings + pos_embeddings
        return final_embeddings

if __name__ == "__main__":

    vocab_size = 50257
    emb_dim  = 256
    max_length = 1024

    embedding_layer = TokenPosEmbedding(vocab_size, emb_dim, max_length)

    dummy_input_ids = torch.tensor([
        [7382,32,12,34],
        [3543 ,23,546,25]
    ])

    embedding_vector = embedding_layer(dummy_input_ids)

    print(embedding_vector[0 , 0 , :5])