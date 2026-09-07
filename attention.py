import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask = None):

    scores = Q@K.transpose(-2,-1)

    scores = scores/ math.sqrt(Q.size(-1)) 

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = F.softmax(scores, dim = -1)

    output = attention_weights @ V

    return output, attention_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 4
    d_k = 3
    d_v = 3

    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)

    print("\nAttention weights:")
    print(weights)

    print("\nOutput:")
    print(output)