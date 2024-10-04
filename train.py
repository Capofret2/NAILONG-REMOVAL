import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

early_stopping = EarlyStopping(patience=30, delta=0.005)

# 数据增强和转换
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    transforms.RandomGrayscale(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def train_model():
    # 数据集和数据加载器
    train_dataset = datasets.ImageFolder('./train', transform=train_transform)
    test_dataset = datasets.ImageFolder('./test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 检查是否有已训练的模型
    model_path = 'trained_resnet101.pth'
    if not os.path.isfile(model_path):
        print(f"{model_path} 文件不存在喵...(￣^￣) 重新炼吧！")
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        for param in model.parameters():
            param.requires_grad = False # 第一次训练，冻结预训练模型层
    else:
        print(f"{model_path} 文件存在喵～(≧▽≦) 接着炼！")
        model = models.resnet101(weights=None)
        for param in model.parameters():
            param.requires_grad = True

    # 修改最后的全连接层用于二分类
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)
    )

    # 加载上次保存的模型权重
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))

    # 分类头
    for param in model.fc.parameters():
        param.requires_grad = True

    # 设备与模型迁移
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 优化器、损失函数和学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = GradScaler()

    # 训练
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # 验证
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    torch.save(model.state_dict(), 'trained_resnet101.pth')
    print("model saved as 'trained_resnet101.pth'")

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    train_model()
