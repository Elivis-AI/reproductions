import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encode_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, encode_dim),
        )

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(encode_dim, input_dim // 4, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def train(data, Y, n_epochs=301, lr=0.01, weight_decay=5e-4, verbose=True, batch_size=128):
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoEncoder(data.shape[1], encode_dim=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 初始化交互绘图
    plt.ion()
    fig, ax = plt.subplots()
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_data, in loader:
            batch_data = batch_data.to(device)

            # 添加高斯噪声
            noisy_input = batch_data # + noise * torch.randn_like(batch_data)
            output = model(noisy_input)
            loss = criterion(output, batch_data)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch_data)

        epoch_loss /= len(data)
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Train MSE: {epoch_loss:.6f}")

        # 每100个epoch进行一次推理并可视化
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                encoded = model.encode(data.to(device)).cpu().numpy()
            ax.cla()
            ax.set_title(f"Encoded Representation at Epoch {epoch}")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.scatter(encoded[:, 0], encoded[:, 1], s=5, c=Y)
            plt.draw()
            plt.pause(0.001)
            model.train()

    # 最后一次获取编码输出，并关闭交互模式
    model.eval()
    with torch.no_grad():
        encoded = model.encode(data.to(device)).cpu().numpy()
    plt.ioff()
    return encoded


if __name__ == '__main__':
    iris = load_digits()
    X = iris.data
    Y = iris.target

    data_1 = train(X, Y, lr=1e-3, n_epochs=1000)
    data_2 = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title("my_AutoEncoder")
    plt.scatter(data_1[:, 0], data_1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_PCA")
    plt.scatter(data_2[:, 0], data_2[:, 1], c=Y)
    plt.show()
