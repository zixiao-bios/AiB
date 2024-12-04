import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        # 第一层全连接层，输入784（28x28图像），输出256
        self.fc1 = nn.Linear(784, 256)

        # 第二层全连接层，输入256，输出128
        self.fc2 = nn.Linear(256, 128)

        # 第三层全连接层，输入128，输出10（类别数）
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # x.shape = (b, 1, 28, 28)
        
        # 将图像展平为一维向量
        x = torch.flatten(x, 1)
        # x.shape = (b, 784)

        # 第一层全连接后使用ReLU激活函数
        x = F.relu(self.fc1(x))

        # 第二层全连接后使用ReLU激活函数
        x = F.relu(self.fc2(x))

        # 第三层全连接输出
        x = self.fc3(x)
        # x.shape = (b, 10)

        # 在使用nn.CrossEntropyLoss时，不需要在这里应用Softmax
        return x

def main():
    # 加载 MNIST 训练集
    # 参数：数据集的本地路径、使用训练集还是测试集、是否自动下载数据集、数据预处理流程
    train_dataset = datasets.MNIST(
        root='./data',
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    
    # 从数据集创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SimpleMLP()
    
    # 分类问题，使用交叉熵损失函数
    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 设置网络为训练模式
    model.train()

    for epoch in range(5):  # 总共训练5轮
        # 本轮训练的平均loss
        train_loss = 0

        # enumerate用法：https://www.runoob.com/python/python-func-enumerate.html
        for batch_idx, (data, target) in enumerate(train_loader):
            # data.shape = (b, 1, 28, 28)
            # target.shape = (b, 10)

            # 将网络中的梯度清零
            optimizer.zero_grad()

            # 进行一次推理
            output = model(data)
            # output.shape = (b, 10)

            # 调用损失函数，计算损失值
            loss = loss_func(output, target)
            train_loss += loss.item()

            # 反向传播，计算loss的梯度
            loss.backward()

            # 使用网络中的梯度更新参数
            optimizer.step()

            # 每100次循环打印一次
            if batch_idx % 100 == 0:
                print(f"训练轮次: {epoch + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] 损失: {loss.item():.6f}")
        
        print(f"================ 训练轮次: {epoch + 1} 平均损失: {train_loss / len(train_loader):.6f} ================\n")


if __name__ == '__main__':
    main()
