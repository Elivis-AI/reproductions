import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=7, act='relu'):
        super().__init__()
        act_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
        }[act]
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), 
            act_fn,
            nn.Linear(256, 32),
            act_fn,
            # nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)  # (I_Gen, U_b)
        )

    def forward(self, x):
        return self.net(x)



class CNNRegressor(nn.Module):
    def __init__(self, in_dim=7, steps=9, out_dim=2):
        super().__init__()
        assert steps % 2 == 1
        # 使用 Conv1d 模拟滑动窗口提取特征
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=128,
                              kernel_size=steps, stride=1,
                              padding=(steps-1)//2, bias=False)

        self.mlp = nn.Sequential(
            # nn.ReLU(),
            # nn.Linear(128, 32),
            nn.ReLU(),
            # nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(128, out_dim)  # (I_Gen, U_b)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, N, d] -> [B, d, N]

        conv_out = self.conv(x)            # [B, 64, N]
        conv_out = conv_out.transpose(1, 2)  # [B, L', 64]

        out = self.mlp(conv_out)           # [B, L', 2]
        return out
