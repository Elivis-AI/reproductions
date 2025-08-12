from torch.utils.data import Dataset


class BookCorpus(Dataset):
    def __init__(self, file: str):
        super().__init__()
        with open(file, 'r') as f:
            self.s_list = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.s_list)
    
    def __getitem__(self, idx):
        return self.s_list[idx]