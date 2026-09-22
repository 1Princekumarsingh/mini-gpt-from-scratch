import torch
from torch.utils.data import Dataset

class LanguageModelDataset(Dataset):

    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):

        x = self.token_ids[idx:idx + self.seq_len]
        y = self.token_ids[idx + 1:idx + self.seq_len + 1]

        return torch.tensor(x, dtype = torch.long), torch.tensor(y, dtype = torch.long) 
