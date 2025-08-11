import torch

def compute_iou_norm(xywh_a: torch.Tensor, xywh_b: torch.Tensor) -> torch.Tensor:
    """
    计算两个 YOLO 格式（根据长宽归一化到[0, 1]）的 bbox 的 IoU（torch版本）
    xywh_a: (B, 4) 或 (B, N, 4)  [x_center, y_center, w, h] 归一化坐标
    xywh_b: 同上，形状需和 xywh_a 一致
    返回: IoU 张量，形状与输入前两维一致
    """
    # 解包
    xa, ya, wa, ha = xywh_a.unbind(-1)
    xb, yb, wb, hb = xywh_b.unbind(-1)

    # 转成 xmin, ymin, xmax, ymax
    xmin_a, xmax_a = xa - wa / 2, xa + wa / 2
    ymin_a, ymax_a = ya - ha / 2, ya + ha / 2
    xmin_b, xmax_b = xb - wb / 2, xb + wb / 2
    ymin_b, ymax_b = yb - hb / 2, yb + hb / 2

    # 交集
    inter_xmin = torch.maximum(xmin_a, xmin_b)
    inter_xmax = torch.minimum(xmax_a, xmax_b)
    inter_ymin = torch.maximum(ymin_a, ymin_b)
    inter_ymax = torch.minimum(ymax_a, ymax_b)

    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_w * inter_h

    # 并集
    area_a = wa * ha
    area_b = wb * hb
    union_area = area_a + area_b - inter_area + 1e-6  # 防止除0

    return inter_area / union_area
