from matplotlib import pyplot as plt
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
dataloader = DataLoader(dataset, batch_size=32, shuffle=False) # shuffle=False is important

# test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

image_to_explain, _ = dataset[4]
image_to_explain = image_to_explain.unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
image_to_explain = image_to_explain.to(device)
X_reference = next(iter(dataloader))[0][:5].to(device) # Lấy từ dataloader cho nhanh

# (6) Tính giá trị Shapley
explainer = shap.GradientExplainer(model, X_reference)
shap_values = explainer.shap_values(image_to_explain)

# (7) Trực quan hóa
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

image_to_explain_denorm = denormalize(image_to_explain[0].cpu().clone(),
                                      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Hiển thị ảnh gốc
plt.figure(figsize=(6, 10))  # Adjusted figure size for vertical layout

# Plot original image in row 1
plt.subplot(2, 1, 1)  # (2 rows, 1 column, first plot)
plt.imshow(image_to_explain_denorm.permute(1, 2, 0))
plt.axis('off')
plt.title("Original Image")

# Plot SHAP values in row 2
plt.subplot(2, 1, 2)  # (2 rows, 1 column, second plot)
shap.image_plot(shap_values[0][0], 
                -image_to_explain[0].cpu().numpy(),
                show=False)  # Prevent auto-display

# Adjust layout and display
plt.tight_layout()
plt.show()

