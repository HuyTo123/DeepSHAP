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
model_path = 'testmodel/rest_net/resnet50_catdog_finetuned.pth'
# image_path_to_explain = 'test_set/cats/cat.4001.jpg' # Không cần nữa nếu dùng dataset
train_dir = 'testmodel/training_set' # Vẫn cần nếu dùng cho background, nhưng code hiện tại không dùng
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
def detach(img_tensor):
    img_tensor = img_tensor.cpu().detach()
    img_tensor = img_tensor.numpy()
    img_tensor = np.transpose(img_tensor, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_tensor = img_tensor*std+mean
    img_display = np.clip(img_tensor, 0, 1)
    return img_display

# Tải dataset để lấy ảnh và đường dẫn
try:
    dataset = torchvision.datasets.ImageFolder(root='testmodel/test_set', transform=explain_transform)
    if len(dataset) < 15: # Kiểm tra xem có đủ ảnh không
         print(f"Lỗi: Thư mục 'test_set' cần ít nhất 15 ảnh.")
         exit()
    # Lấy 10 ảnh để giải thích (ví dụ)
    X_explain = torch.stack([dataset[i][0] for i in range(10)])
    # Lấy 5 ảnh làm tham chiếu (ví dụ)
    X_reference = torch.stack([dataset[i][0] for i in range(10, 12)])
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy thư mục test_set")
    exit()


X_explain = X_explain.to(device)
X_reference = X_reference.to(device)

# --- Tính SHAP values ---
print("Đang tính SHAP values...")
explainer = shap.GradientExplainer(model_explain, X_reference)
shap_values,indexes = explainer.shap_values(X_explain, ranked_outputs=2) # list[array(batch, C, H, W)]
print("Tính SHAP values xong.")



# --- Hiển thị SHAP plot (Theo đúng dòng lệnh yêu cầu) ---
# Lấy dữ liệu cho plot như bạn chỉ định
shap_data = shap_values[0] # SHAP lớp 0, ảnh 0 (Channels First)
img_data_to_plot = detach(X_explain[0]) # Ảnh 0 (Channels First, Normalized)
shap_data_to_plot = np.sum(shap_data, axis = 0)
index_names = np.vectorize(lambda x: class_names[x])(indexes.cpu())
# Gọi hàm plot gốc của bạn.
# shap.image_plot sẽ tự tạo Figure 2. Đặt show=False để plt.show() cuối cùng quản lý.
# Cảnh báo Clipping và màu sai sẽ xuất hiện ở đây.
try:
    shap.image_plot(
        shap_data_to_plot,
        img_data_to_plot,
        index_names,
        show = True

    )
    print("Đã gọi shap.image_plot cho Figure 2.")
except Exception as e:
     print(f"Lỗi khi gọi shap.image_plot: {e}")

# # --- 3. Hiển thị cả hai cửa sổ đồ thị ---


