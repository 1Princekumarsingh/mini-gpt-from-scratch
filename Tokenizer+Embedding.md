## Tokenizer
- Tokenizer converts text into token IDs.

Eample:

"I love you"
      ↓
["I", "love", "you"]
      ↓
[15, 82, 341]

- Each token has a fixed ID in the vocabulary.

## Token Embedding
- token_embedding = nn.Embedding(vocab_size, d_model)
                  ↓
This creates a learnable token table:
[vocab_size, d_model] # d_model -> vector length

- During backpropagation, the values in this table are updated based on the loss.

## Position Embedding

- Tokens also need to know where they occur in the sequence.

- We create another learnable table:
position_embedding = nn.Embedding(max_seq_len, d_model)

- For example:
position_embedding.weight
[100, 8]

The rows represent positions:

position 0 → [8 numbers]
position 1 → [8 numbers]
position 2 → [8 numbers]
...
position 99 → [8 numbers]

- For a sequence of length 4:
positions = torch.arange(4)

gives: [0, 1, 2, 3]

- These are position IDs, not the vectors themselves.

Then: position_vectors = position_embedding(positions)
fetches the corresponding rows.
The same position table is shared across all sequences.

## Combine token + position
- x = token_vectors + position_vectors

- Now each token representation contains:

token information
      +
position information
      ↓
combined representation