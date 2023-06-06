import torch
from torch.utils.data import Dataset


class MusicDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (
            self.data[index : index + self.seq_length],
            self.data[index + 1 : index + self.seq_length + 1],
        )
