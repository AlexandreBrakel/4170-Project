import torch
from torch.utils.data import Dataset

class NHLDataset(Dataset):
    def __init__(self, input_features, output_labels, transform=None, target_transform=None):
        self.x = input_features
        self.y = output_labels
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