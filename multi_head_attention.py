import torch
import torch.nn as nn
from attention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def split_heads(self, x):

        batch_size, seq_len, _ = x.size()

        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        x = x.transpose(1,2)

        return x

    def combine_heads(self, x):

        batch_size, num_heads, seq_len, head_dim = x.size()

        x = x.transpose(1, 2)

        x = x.contiguous().view(batch_size, seq_len, self.d_model)

        return x

    def forward(self, x, mask = None):
        # project input into q, k , v
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # attention independently in each head
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # combine heads
        attention_output = self.combine_heads(attention_output)

        # final linear projection
        output = self.out_proj(attention_output)

        return output, attention_weights