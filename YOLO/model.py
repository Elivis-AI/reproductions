import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from utils import compute_iou_norm

class YOLOv1(nn.Module):
    def __init__(self, n_grid: int, n_classes: int, bbox_per_grid: int = 2):
        """ output: (B, n_grid, n_grid, dim) """
        super().__init__()
        self.n_grid = n_grid # S
        self.n_classes = n_classes
        self.bbox_per_grid = bbox_per_grid # B
        self.dim = bbox_per_grid * 5 + n_classes # B*5 + C
        
        # 用imagenet预训练过的backbone
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT) # (B, 2048, 14, 14)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2],
        ) # 去掉最后的平均池化层和全连接层
        for param in self.backbone.parameters():
            param.requires_grad=False
        
        # 拼接目标检测的分类头
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=2048,out_channels=1024,kernel_size=3,padding=1), # (batch,1024,14,14)
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1), # (batch,1024,7,7)
            nn.LeakyReLU(0.1),
            
            nn.Flatten(),
            nn.Linear(in_features=7*7*1024,out_features=4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096,out_features=n_grid*n_grid*self.dim), # (batch,n_grid*n_grid*dim)
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x) # (batch,1024,7,7)
        x = self.head(x) # (batch,S*S*(10+C))
        return x.view(-1, self.n_grid, self.n_grid, self.dim)
    

    def cal_loss(self, pred: torch.Tensor, target: torch.Tensor, a_coord: float = 5, a_noobj: float = .5) -> torch.Tensor:
        """
        pred: (B, n_grid, n_grid, dim)
        target: (B, n_grid, n_grid, dim)
        """
        obj_mask = target[..., [4]] # 每个块是否负责一个目标
        
        # 前 B*5 个通道是 bbox 预测 x,y,w,h,c
        pred_bbox_tensors = pred[..., :-self.n_classes].view(-1, self.n_grid, self.n_grid, self.bbox_per_grid, 5)
        target_bbox_tensors = target[..., :-self.n_classes].view(-1, self.n_grid, self.n_grid, self.bbox_per_grid, 5)
        max_iou, max_bbox_idx = compute_iou_norm(
            pred_bbox_tensors[..., :4],
            target_bbox_tensors[..., :4]
        ).max(dim=-1, keepdim=True)  # iou: (B, n_grid, n_grid, 1), bbox_idx: (B, n_grid, n_grid, 1)

        # 取出iou最大的bbox
        pred_max_bbox_tensor = torch.take_along_dim(
            pred_bbox_tensors, 
            max_bbox_idx.unsqueeze(-1).expand(-1, -1, -1, 1, 5),  # (B, n_grid, n_grid, 1, 5)
            dim=-2
        ).squeeze(-2) # (B, n_grid, n_grid, 5)
        target_max_bbox_tensor = target_bbox_tensors[..., 0, :] # (B, n_grid, n_grid, 5)

        # 坐标框损失
        wh_loss = F.mse_loss(pred_max_bbox_tensor[..., 2:4].sqrt(), target_max_bbox_tensor[..., 2:4].sqrt(), reduction='none').sum(-1, keepdim=True)
        center_loss = F.mse_loss(pred_max_bbox_tensor[..., :2], target_max_bbox_tensor[..., :2], reduction='none').sum(-1, keepdim=True)
        xywh_loss = wh_loss + center_loss

        # 类别损失
        # 原始yolov1论文用的mse，这里换成正常对分类任务作用的交叉熵(v2及以后采用)
        # class_loss = F.mse_loss(pred_class_tensor, target_class_tensor, reduction='none').sum(-1, keepdim=True)
        class_loss = F.binary_cross_entropy(pred_max_bbox_tensor[..., -self.n_classes:], target_max_bbox_tensor[..., -self.n_classes:], reduction='none').sum(-1, keepdim=True)

        # 置信度损失
        c_obj_loss = F.mse_loss(max_iou, pred_max_bbox_tensor[..., [4]], reduction='none')
        c_noobj_loss = torch.sum(
            pred_bbox_tensors[..., [4]] ** 2,
            dim=-2, keepdim=False
        ) - obj_mask * pred_max_bbox_tensor[..., [4]] ** 2
        
        # print(obj_mask.shape, class_loss.shape, c_loss.shape, wh_loss.shape, center_loss.shape) 
        loss = a_coord * obj_mask * xywh_loss \
            + obj_mask * c_obj_loss + a_noobj * c_noobj_loss \
                + class_loss * obj_mask
        return loss.sum()
    

if __name__ == '__main__':
    model = YOLOv1(n_grid=7, bbox_per_grid=2, n_classes=20)
    x = torch.randn(4, 3, 224, 224)  # batch size 4
    y = model(x)
    print(y.shape)  # (4, 7, 7, 30) 因为 B*5 + C = 2*5 + 20 = 30
    print(model.cal_loss(y, y))