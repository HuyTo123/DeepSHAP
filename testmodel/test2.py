import shap
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
model = CNN()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()  # Đặt chế độ eval để tắt Dropout

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),  # Ensures exact dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Chuẩn hóa dữ liệu về khoảng [-1,1]
])

dataset = torchvision.datasets.ImageFolder(root='test_set', transform=transform)
# test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

X_explain = torch.stack([dataset[i][0] for i in range(10)])
X_reference = torch.stack([dataset[i][0] for i in range(10, 15)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Đưa mô hình lên GPU (nếu có)
X_explain = X_explain.to(device)
X_reference = X_reference.to(device)

explainer = shap.GradientExplainer(model, X_reference)
shap_values = explainer.shap_values(X_explain)

shap.image_plot(shap_values[0][0], -X_explain[0].cpu().numpy())