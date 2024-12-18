from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    """实现残差块
    """
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 是否使用 1x1 卷积层来适配尺寸
        if use_1x1conv:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.res_conv = None

    def forward(self, X):
        Y = self.seq(X)
        
        if self.res_conv:
            X = self.res_conv(X)
        
        Y += X
        return F.relu(Y)
