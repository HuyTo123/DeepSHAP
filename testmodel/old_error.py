
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import shap# Note: double underscores
# Define the CNN model
import os
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
    # Deep Explainer bị lỗi do khai báo trùng self.relu phải tạo nhiều object khác nhau self.relu1, 2, 3
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # x = torch.flatten(x, 1) 
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
model = CNN()
model.load_state_dict(torch.load('testmodel/model.pth', map_location=torch.device('cuda')))
model.eval()  # Đặt chế độ eval để tắt Dropout
# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),  # Ensures exact dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Chuẩn hóa dữ liệu về khoảng [-1,1]
])
def detatch(img_tensor):
    img_tensor = img_tensor.cpu().detach()
    img_tensor = img_tensor.numpy()
    img_tensor = np.transpose(img_tensor, (1,2,0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img_tensor = img_tensor*std+mean
    img_display = np.clip(img_tensor, 0, 1)
    return img_display
dataset = torchvision.datasets.ImageFolder(root='testmodel/test_set',transform=transform    )
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# X_explain = torch.stack([dataset[i][0] for i in range(100,103)])
# X_reference = torch.stack([dataset[i][0] for i in range(100)])
batch = next(iter(test_loader))
images, _ = batch

background = images[:5]
test_images = images[6:7]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

explainer = shap.DeepExplainer(model, background)
print('Đã khởi tạo xong DeepExplainer')
shap_values = explainer.shap_values(test_images.to(device)) # Chuyển đổi kích thước tensor về (1,3,128,128) để phù hợp với đầu vào của model
shap_values = np.sum(shap_values[0], axis = 0)
img_data = detatch(test_images[0])
print('Đã tính toán xong SHAP values')
plot_image = shap.image_plot(shap_values,img_data, show=True ) # Chuyển đổi về numpy array để vẽ hình ảnh

