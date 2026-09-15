import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward

class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model, d_ff)

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask = None):

        # self attention
        attention_output, attention_weights = self.attention(x, mask = mask)

        # residual connection + layer norm
        x = self.norm1(x + attention_output)

        # ffn
        ff_output = self.feed_forward(x)

        # residual connection + layer norm
        x = self.norm2(x + ff_output)

        return x, attention_weights


