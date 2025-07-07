from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.Y = ...
        pass

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
