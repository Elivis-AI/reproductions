from torchvision import datasets, transforms
from torch.utils.data import Dataset

class MNIST(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.dataset = datasets.MNIST('~/.cache/', train=is_train, download=True)
        self.img_convert = transforms.Compose([
            transforms.PILToTensor(),
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        img, label = self.dataset[idx]
        return self.img_convert(img) / 255.0, label
    

class CIFAR10(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.dataset = datasets.CIFAR10('~/.cache/', train=is_train, download=True)
        self.img_convert = transforms.Compose([
            transforms.PILToTensor(),
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.img_convert(img) / 255.0, label