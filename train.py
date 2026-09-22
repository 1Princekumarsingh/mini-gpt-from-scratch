import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tokenizer import CharacterTokenizer
from dataset import LanguageModelDataset
from transformer import GPT
from attention import causal_mask


def compute_loss(model, x, y):
    mask = causal_mask(x.size(1), device=x.device)
    logits, _ = model(x, mask = mask)
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(B*T, V), y.reshape(B*T))

# load text
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# tokenizer
tokenizer = CharacterTokenizer(text)
token_ids = tokenizer.encode(text=text)
vocab_size = tokenizer.vocab_size

# model configuration
max_seq_len = 32

d_model = 64
num_heads = 4
d_ff = 256
num_layers = 4

batch_size = 8
learning_rate = 3e-4
epochs = 3
val_split = 0.2

# dataset
split_idx = int(len(token_ids) * (1 - val_split))
min_train_tokens = max_seq_len + 1
max_train_tokens = len(token_ids) - (max_seq_len + 1)
split_idx = max(min_train_tokens, min(split_idx, max_train_tokens))

train_token_ids = token_ids[:split_idx]
val_token_ids = token_ids[split_idx:]

train_dataset = LanguageModelDataset(token_ids=train_token_ids, seq_len=max_seq_len)
val_dataset = LanguageModelDataset(token_ids=val_token_ids, seq_len=max_seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# model
model = GPT(
    vocab_size = vocab_size,
    max_seq_len = max_seq_len,
    d_model = d_model,
    num_heads = num_heads,
    d_ff = d_ff,
    num_layers = num_layers
).to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# training
for epoch in range(epochs):
    model.train()

    train_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        # cross entropy
        loss = compute_loss(model, x, y)

        # backprop
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()

    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            loss = compute_loss(model, x, y)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )
