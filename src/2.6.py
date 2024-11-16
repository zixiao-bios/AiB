import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ===================== 1. 构建人工数据集 =====================
# 自定义数据集类
class NonlinearDataset(Dataset):
    def __init__(self, num_samples=5000):
        super(NonlinearDataset, self).__init__()
        
        # 生成数据
        self.X = torch.linspace(-4, 4, num_samples).unsqueeze(1)  # (num_samples, 1)
        self.y = (
            torch.sin(2 * self.X)
            + torch.log(torch.abs(self.X) + 1)
            - 0.05 * self.X**3
            + torch.randn(num_samples, 1) * 0.2
        )  # (num_samples, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 实例化数据集
dataset = NonlinearDataset(num_samples=5000000)

# 创建 DataLoader，指定批量大小和是否打乱数据
batch_size = 1000000
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ===================== 2. 定义 MLP 模型 =====================
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        
        # 多层感知机，包含两层隐藏层
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),  # 输入维度为1，隐藏层维度为64
            nn.ReLU(),         # 激活函数 ReLU
            nn.Linear(64, 64), # 隐藏层维度为64
            nn.ReLU(),         # 激活函数 ReLU
            nn.Linear(64, 1),  # 输出维度为1
        )

    def forward(self, x):
        return self.mlp(x)

# 实例化模型
model = MLPModel()


# ===================== 3. 定义损失函数和优化器 =====================
# 均方误差作为损失函数
criterion = nn.MSELoss()

# Adam 作为优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 学习率 0.01


# ===================== 4. 训练模型 =====================
epochs = 20  # 训练回合数
for epoch in range(epochs):
    for X_batch, y_batch in dataloader:
        # 使用模型预测输出
        predictions = model(X_batch)
        
        # 计算损失
        loss = criterion(predictions, y_batch)
        
        # 梯度清零，防止累积
        optimizer.zero_grad()
        
        # 计算梯度
        loss.backward()
        
        # 更新参数
        optimizer.step()

    # 打印训练过程中的损失值
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# ===================== 5. 可视化结果 =====================
# 使用训练好的模型预测
with torch.no_grad():
    predictions = model(dataset.X)

# 原始数据
plt.scatter(dataset.X.numpy(), dataset.y.numpy(), label='Original Data', s=10)

# 预测数据
plt.plot(dataset.X.numpy(), predictions.numpy(), color='red', label='Fitted Curve')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title("Fitted Curve by MLP with Batches")
plt.show()
