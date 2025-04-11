
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
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # Input: 3 x 32 x 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 16 x 16 x 16

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 32 x 8 x 8

        # Flatten: 32 * 8 * 8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes) # num_classes = 2 (cats, dogs)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1) # Flatten tất cả các chiều trừ batch
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load('testmodel/cats_dogs_model_32x32.pth', map_location=torch.device('cpu')))
model.eval()  # Đặt chế độ eval để tắt Dropout

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
def detatch(img_tensor):
    img_tensor = img_tensor.cpu().detach()
    img_tensor = img_tensor.numpy()
    img_tensor = np.transpose(img_tensor, (1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_tensor = img_tensor*std+mean
    img_display = np.clip(img_tensor, 0, 1)
    return img_display
dataset = torchvision.datasets.ImageFolder(root='testmodel/test_set', transform=transform)
test_loader = DataLoader(dataset, batch_size=64,  shuffle=False)

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
plot_image = shap.image_plot(shap_values,img_data, show=False ) # Chuyển đổi về numpy array để vẽ hình ảnh
print('Đã tính toán xong SHAP values')
plt.show()

