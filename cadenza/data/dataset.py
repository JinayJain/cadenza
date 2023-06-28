import torch
from torch.utils.data import Dataset


class MusicDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_length, transform=None):
        # separate the data into chunks of length seq_length
        batch_len = seq_length + 1
        self.batches = data.shape[0] // batch_len
        self.data = data[: self.batches * batch_len]
        self.data = self.data.reshape(self.batches, batch_len)
        self.transform = transform
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample[:-1], sample[1:]
