                  TOKEN IDs
                      │
                      ▼
            ┌─────────────────┐
            │ Token Embedding  │
            └────────┬────────┘
                     +
            Position Embedding
                     │
                     ▼
          ┌─────────────────────┐
          │ Transformer Block   │ × N
          │                     │
          │ Causal MHA          │
          │       ↓             │
          │ Residual + Norm     │
          │       ↓             │
          │ FFN                 │
          │       ↓             │
          │ Residual + Norm     │
          └──────────┬──────────┘
                     │
                     ▼
                Final Norm
                     │
                     ▼
                  LM Head
                     │
                     ▼
                   Logits
                     │
                     ▼
              Next-token scores