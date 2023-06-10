import torch
from torch.utils.data import Dataset


class MusicDataset(Dataset):
    def __init__(self, data, seq_length, transform=None):
        self.data = data
        self.seq_length = seq_length
        self.transform = transform

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        sample = self.data[index : index + self.seq_length + 1]

        if self.transform:
            sample = self.transform(sample)

        return (
            sample[:-1],
            sample[1:],
        )
