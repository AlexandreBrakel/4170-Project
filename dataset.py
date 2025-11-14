import torch
from torch.utils.data import Dataset

class NHLDataset(Dataset):
    def __init__(self, data_list, transform=None, target_transform=None):
        # TODO change how data is loaded
        self.x = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]
        self.y = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y