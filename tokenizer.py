class CharacterTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))

        self.stoi = {ch : idx for idx, ch in enumerate(chars)}
        self.itos = {idx : ch for ch, idx in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[id] for id in ids)

    @property
    def vocab_size(self):
        return len(self.stoi)