import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from mlp import SimpleMLP

num_epochs = 100
lr = 0.01
batch_size = 64

hidden_dim = 2
hidden_num = 2
weight_decay = 1e-5


def main():
    # 读入处理后的数据
    print('\n================================== 读入处理后的数据 ==================================')
    df_train = pd.read_csv('dataset/train_processed.csv')
    df_test = pd.read_csv('dataset/test_processed.csv')
    
    df_train_features = df_train.drop(['Transported', 'PassengerId'], axis=1)
    df_train_target = df_train['Transported']
    df_test_features = df_test.drop(['Transported', 'PassengerId'], axis=1)
    print(df_train_features)
    print(df_train_target)
    
    # 将数据转换为 PyTorch 的 Tensor
    print('\n================================== 将数据转换为 PyTorch 的 Tensor ==================================')
    n_train = df_train.shape[0]
    n_test = df_test.shape[0]
    
    X_train = torch.tensor(df_train_features.values, dtype=torch.float32)
    y_train = torch.tensor(df_train_target.values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(df_test_features.values, dtype=torch.float32)
    print(X_train.shape)
    print(y_train.shape)
    
    # 构建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 定义模型、损失函数和优化器
    model = SimpleMLP(input_dim=X_train.shape[1], hidden_num=hidden_num, hidden_dim=hidden_dim, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 二元交叉熵损失函数(Binary Cross Entropy)，用于二分类任务
    loss = nn.BCELoss()
    
    # 训练模型
    print('\n================================== 训练模型 ==================================')
    for epoch in range(num_epochs):
        model.train()
        
        # 每个 epoch 的损失
        epoch_loss = 0
        
        # 预测正确的个数
        correct_num = 0
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            correct_num += torch.sum((y_pred > 0.5) == y_batch).item()
            
            l = loss(y_pred, y_batch)
            epoch_loss += l.item()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        
        print(f'Epoch: {epoch}, Epoch Loss: {epoch_loss}, Accuracy: {correct_num / n_train}')
    
    # 预测测试集
    print('\n================================== 预测测试集 ==================================')
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = (y_pred > 0.5).reshape(-1).cpu().numpy().astype(bool)
    sub = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Transported': y_pred})
    print(sub)
    sub.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
