import torch 
import torch.nn as  nn 
import torch.nn.functional as F



#################
#     Text      #
#################

class TextEncoder(nn.Module):
    """a vanilla MLP text(class) encoder"""
    def __init__(self, dim_out=8):
        super().__init__()
        self.dim_out = dim_out
        self.embedding = nn.Embedding(10, 16)
        self.encoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, dim_out),
            nn.LayerNorm(dim_out),
        )

    def forward(self,x):
        x = self.embedding(x)
        x = self.encoder(x)
        return x.reshape(-1, self.dim_out) # x.ndim == 2
    

##################
#     Image      #
##################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1), # [b, ch_in, 28, 28] => [b, ch_out, 14, 14]
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),# keep dims: [b, ch_out, 13, 13]
            nn.BatchNorm2d(out_channels),
        )
        self.down_sample = nn.Conv2d(in_channels, out_channels, 1, 2, 0)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.model(x) + self.down_sample(x))


class ImageEncoder(nn.Module):
    def __init__(self, dim_out=8):
        super().__init__()
        self.res_blocks = nn.Sequential(
            ResidualBlock(in_channels=1,out_channels=16), # (batch,16,14,14)
            ResidualBlock(in_channels=16,out_channels=4), # (batch,4,7,7)
            ResidualBlock(in_channels=4,out_channels=1),  # (batch,1,4,4)
        )

        self.out_proj = nn.Sequential(
            nn.Linear(in_features=16,out_features=dim_out),
            nn.LayerNorm(dim_out)     
        )

        
    def forward(self,x):
        x = self.res_blocks(x)
        x = x.view(x.size(0),-1)
        x = self.out_proj(x)
        return x
