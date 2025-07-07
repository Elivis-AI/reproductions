import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import MyModel
from dataset import MyDataset

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aaa", type=str, default="aaa", help="aaa")
    parser.add_argument("--bbb", type=int, default=666, help="666")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset = MyDataset()
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    model.load_state_dict(torch.load("weights/tiny_mlp_epoch_300.pth", map_location=device))
    
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            out = model(bx).cpu().numpy()
            preds.append(out)
    preds = np.vstack(preds).squeeze()



if __name__ == "__main__":
    main()
