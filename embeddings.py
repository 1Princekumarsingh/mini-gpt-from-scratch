import torch
import torch.nn as nn

class GPTInputEmbedding(nn.Module):
    
    def __init__(self,vocab_size, d_model, max_seq_len):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape

        positions = torch.arange(seq_len, device=token_ids.device)

        token_vector = self.token_embedding(token_ids)
        position_vector = self.position_embedding(positions)

        x = token_vector + position_vector

        return x
