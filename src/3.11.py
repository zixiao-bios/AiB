from torch import nn

# 在全连接层使用 BN
layer = nn.Sequential(
    nn.Linear(128, 64),
    # 1D Batch norm，输入维度为 256
    nn.BatchNorm1d(64),
    nn.ReLU()
)

# 在卷积层使用 BN
layer = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=3),
    # 2D Batch norm，通道数为 6（卷积层的输出通道数）
    nn.BatchNorm2d(6),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 2, stride = 2)
)
