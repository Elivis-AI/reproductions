import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms
from typing import Tuple


class VOCDetectionForYOLO(Dataset):
    """
    YOLO system models detection as a regression problem. 
    It divides the image into an S × S grid and for each grid cell predicts B bounding boxes,
    confidence for those boxes, and C class probabilities. 
    These predictions are encoded as an S × S × (B ∗ 5 + C) tensor.
    """
    def __init__(self, image_set: str, image_size: int, n_grid: int, bbox_per_grid: int = 2, download=False):
        assert image_size % n_grid == 0
        self.voc = VOCDetection(root='~/.cache', year='2012', image_set=image_set, download=download)
        self.all_classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self.image_size = image_size
        self.n_grid = n_grid
        self.bbox_per_grid = bbox_per_grid
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.all_classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.all_classes)}
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self.voc[idx]
        # 每张图片的尺寸可能都不同
        x_scale = self.image_size / img.width
        y_scale = self.image_size / img.height
        grid_size = self.image_size // self.n_grid
        
        x = self.transform(img)
        # S × S × (B ∗ 5 + C)
        y = torch.zeros(self.n_grid, self.n_grid, self.bbox_per_grid * 5 + len(self.all_classes))
        
        for obj in label['annotation']['object']: # for each bbox
            bbox = obj['bndbox']
            classid = self.class_to_idx[obj['name']]
            
            # 转换尺寸后的绝对坐标 范围[0, self.image_size]
            xmin = int(bbox['xmin']) * x_scale
            xmax = int(bbox['xmax']) * x_scale
            ymin = int(bbox['ymin']) * y_scale
            ymax = int(bbox['ymax']) * y_scale
            xcenter = (xmin + xmax) / 2
            ycenter = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            # 只预测中心点所在的grid格子
            grid_i, grid_j = int(ycenter // grid_size), int(xcenter // grid_size)
            xcenter, ycenter = xcenter % grid_size / grid_size, ycenter % grid_size / grid_size
            width, height = width / self.image_size, height / self.image_size # normalized global bbox size
            # fill tensor
            grid_tensor = torch.tensor([xcenter, ycenter, width, height, 1.0])  # x,y,w,h,c, c=1.0 is the confidence
            class_tensor = F.one_hot(torch.tensor(classid), num_classes=len(self.all_classes)).float()
            y[grid_i, grid_j] = torch.cat([grid_tensor] * self.bbox_per_grid + [class_tensor])

        return x, y

    @property
    def n_classes(self):
        return len(self.all_classes)
    

if __name__ == '__main__':
    try:
        from torchvision.datasets import voc
        for key in voc.DATASET_YEAR_DICT.keys():
            voc.DATASET_YEAR_DICT[key]['url'] = 'http://data.brainchip.com/dataset-mirror/voc/' + voc.DATASET_YEAR_DICT[key]['filename']
        dataset = VOCDetectionForYOLO(image_set='train', image_size=224, n_grid=7, bbox_per_grid=2, download=True)
    except:
        dataset = VOCDetectionForYOLO(image_set='train', image_size=224, n_grid=7, bbox_per_grid=2, download=True)

    
    img, label = dataset[0]
    print(img.shape, label.shape)