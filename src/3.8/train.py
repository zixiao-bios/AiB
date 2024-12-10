from SimpleCNN import SimpleCNN

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 计算准确率、精确率、召回率、F1分数
def metrics(all_targets, all_preds):
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    return accuracy, precision, recall, f1

def train(model, device, train_loader, loss_func, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"训练轮次: {epoch + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] 损失: {loss.item():.6f}")
    
    print(f"Eposh {epoch + 1} 平均损失: {train_loss / len(train_loader):.6f}")

def test(model, device, test_loader, epoch):
    model.eval()
    all_targets, all_preds = [], []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            preds = output.argmax(dim=1)
            
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy, precision, recall, f1 = metrics(all_targets, all_preds)
    print(f"Epoch {epoch + 1} 测试集: 准确率={accuracy:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}, F1分数={f1:.4f}\n")       

def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Use GPU: {torch.cuda.get_device_name(0)}")
    elif torch.mps.is_available():
        device = 'mps'
        print("Use MPS")
    
    # 加载 MNIST 训练集
    # 参数：数据集的本地路径、使用训练集还是测试集、是否自动下载数据集、数据预处理流程
    train_dataset = datasets.MNIST(
        root='./data',
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 加载 MNIST 测试集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = SimpleCNN().to(device)
    
    # 分类问题，使用交叉熵损失函数
    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(5):
        train(model, device, train_loader, loss_func, optimizer, epoch)
        test(model, device, test_loader, epoch)

if __name__ == '__main__':
    main()
