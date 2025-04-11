import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- Cấu hình ---
model_path = 'resnet50_catdog_finetuned.pth'
# image_path_to_explain = 'test_set/cats/cat.4001.jpg' # Không cần nữa nếu dùng dataset
train_dir = 'training_set' # Vẫn cần nếu dùng cho background, nhưng code hiện tại không dùng
num_classes = 2
class_names = ['cats', 'dogs']
# num_background_samples = 50 # Không dùng trong code mới
# batch_size_background = 16 # Không dùng trong code mới

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# --- Tải Model ---
model_explain = models.resnet50(weights=None)
num_ftrs = model_explain.fc.in_features
model_explain.fc = nn.Linear(num_ftrs, num_classes)

try:
    model_explain.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Trọng số model đã được tải từ {model_path}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file trọng số model: {model_path}")
    exit()
except Exception as e:
    print(f"Lỗi khi tải trọng số model: {e}")
    print("Kiểm tra lại kiến trúc model có khớp với file trọng số không.")
    exit()
for module in model_explain.modules():
    if isinstance(module, nn.ReLU):
        module.inplace = False
model_explain.to(device)
model_explain.eval()
print("Model đã sẵn sàng trên thiết bị và ở chế độ eval.")

# --- Chuẩn bị dữ liệu ---
explain_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Tải dataset để lấy ảnh và đường dẫn
try:
    dataset = torchvision.datasets.ImageFolder(root='test_set', transform=explain_transform)
    if len(dataset) < 15: # Kiểm tra xem có đủ ảnh không
         print(f"Lỗi: Thư mục 'test_set' cần ít nhất 15 ảnh.")
         exit()
    # Lấy 10 ảnh để giải thích (ví dụ)
    X_explain = torch.stack([dataset[i][0] for i in range(10)])
    # Lấy 5 ảnh làm tham chiếu (ví dụ)
    X_reference = torch.stack([dataset[i][0] for i in range(10, 15)])
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy thư mục test_set")
    exit()



X_explain = X_explain.to(device)
X_reference = X_reference.to(device)

# --- Tính SHAP values ---
print("Đang tính SHAP values...")
explainer = shap.DeepExplainer(model_explain, X_reference)
print("Tạo explainer xong.")
shap_values = explainer.shap_values(X_explain) # list[array(batch, C, H, W)]
print("Tính SHAP values xong.")




shap_data_to_plot = shap_values[0][0] # SHAP lớp 0, ảnh 0 (Channels First)
img_data_to_plot = X_explain[0].cpu().numpy() # Ảnh 0 (Channels First, Normalized)
shap.image_plot(
        shap_data_to_plot,
        img_data_to_plot,
        show=True # Để plt.show() quản lý việc hiển thị
    )
