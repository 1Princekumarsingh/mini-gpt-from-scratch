import math
import torch
import torch.nn.functional as F

def causal_mask(seq_len, device = None):
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

def scaled_dot_product_attention(Q, K, V, mask = None):

    scores = Q@K.transpose(-2,-1)

    scores = scores/ math.sqrt(Q.size(-1)) 

    if mask is not None:
        scores = scores.masked_fill(~mask , float("-inf"))

    attention_weights = F.softmax(scores, dim = -1)

    output = attention_weights @ V
    
    return output, attention_weights
