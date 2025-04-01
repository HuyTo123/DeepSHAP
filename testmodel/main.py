import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Định nghĩa các biến toàn cục
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Chuẩn bị dữ liệu
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),  # Ensures exact dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Chuẩn hóa dữ liệu về khoảng [-1,1]
])
# Tạo DataLoader cho tập dữ liệu huấn luyện và tập dữ liệu kiểm tra với ImageFolder tự động gán nhãn theo cấu trúc thư mục và thực hiện transform về định dạng chuẩn
train_dataset = torchvision.datasets.ImageFolder(root='training_set', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root='test_set', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Xây dựng mô hình CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128*16*16, 256)
        self.fc2 = nn.Linear(256,2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) #flatten layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
model = CNN().to(device='cuda')
criterion  = nn.CrossEntropyLoss()
 # Sử dụng thuật toán tối ưu Adam với learning rate = 0.001 và model parameters cung cấp trọng số cho mô hình
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Huấn luyện mô hình
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device='cuda')
        labels = labels.to(device='cuda')
        optimizer.zero_grad() # Xóa gradient cũ
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}')
torch.save(model.state_dict(), 'model.pth')
print('Finished Training')
# Đánh giá mô hình
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device='cuda')
        labels = labels.to(device='cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100*correct/total:.2f}%')