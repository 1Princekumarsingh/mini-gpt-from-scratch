import torch.nn as nn

from attention import causal_mask
from transformer_block import TransformerBlock
from embeddings import GPTInputEmbedding

class GPT(nn.Module):

    def __init__(self,vocab_size, max_seq_len, d_model, num_heads, d_ff, num_layers):
        super().__init__()

        # token + embedding
        self.embedding = GPTInputEmbedding(d_model = d_model, max_seq_len = max_seq_len ,vocab_size = vocab_size )

        # stack transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(
            d_model = d_model,
            num_heads = num_heads,
            d_ff = d_ff
            )
            for _ in range(num_layers)
        ])

        # final normalization layer
        self.final_norm = nn.LayerNorm(d_model)

        # logits/ model head
        self.model_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, mask = None):

        if mask is None:
            seq_len = token_ids.size(1)
            mask = causal_mask(seq_len, device=token_ids.device)

        # token + embedding 
        x = self.embedding(token_ids)

        # transformer blocks
        attention_weights = []

        for block in self.blocks:
            x, weights = block(x, mask = mask)

            attention_weights.append(weights)

        # final normalization layer
        x = self.final_norm(x)

        #  Predict model logits
        logits = self.model_head(x)

        return logits, attention_weights
        



        

