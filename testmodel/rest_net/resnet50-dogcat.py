import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import os
import copy
import matplotlib.pyplot as plt
from multiprocessing import freeze_support # Thêm dòng này

# --- Cấu hình ---
# ... (giữ nguyên phần cấu hình) ...
data_dir = '.'
train_dir = os.path.join(data_dir, 'training_set')
test_dir = os.path.join(data_dir, 'test_set')
num_classes = 2
batch_size = 32
num_epochs = 15
learning_rate = 0.001
momentum = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Chuẩn bị dữ liệu ---
# ... (giữ nguyên phần data_transforms) ...
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Xây dựng và chỉnh sửa Model ResNet-50 ---
# ... (giữ nguyên phần tải và sửa model) ...
weights = models.ResNet50_Weights.IMAGENET1K_V2
model_ft = models.resnet50(weights=weights)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
# model_ft = model_ft.to(device) # Sẽ chuyển vào trong if __name__ == '__main__'

# --- Định nghĩa Loss Function và Optimizer ---
# ... (giữ nguyên phần criterion, optimizer, scheduler) ...
criterion = nn.CrossEntropyLoss()
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate) # Sẽ tạo trong if __name__ == '__main__'
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) # Sẽ tạo trong if __name__ == '__main__'


# --- Hàm Huấn luyện và Đánh giá ---
# ... (giữ nguyên hàm train_model và plot_history) ...
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    # ... (Nội dung hàm giữ nguyên) ...
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train': scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'*** Lưu model tốt nhất với Val Acc: {best_acc:.4f} ***')
        print()
    time_elapsed = time.time() - since
    print(f'Huấn luyện hoàn thành trong {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Accuracy tốt nhất trên tập Validation: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history

def plot_history(history):
    # ... (Nội dung hàm giữ nguyên) ...
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===============================================
# === BẢO VỆ PHẦN THỰC THI CHÍNH BẰNG if __name__ == '__main__': ===
# ===============================================
if __name__ == '__main__':
    # Gọi freeze_support() ngay đầu khối if __name__ == '__main__' trên Windows
    # Quan trọng khi dùng multiprocessing (DataLoader với num_workers > 0)
    freeze_support()

    print(f"Sử dụng thiết bị: {device}")

    # Kiểm tra thư mục dữ liệu
    if not os.path.isdir(train_dir):
        print(f"Lỗi: Không tìm thấy thư mục training: {train_dir}")
        exit()
    if not os.path.isdir(test_dir):
        print(f"Lỗi: Không tìm thấy thư mục test: {test_dir}")
        exit()

    # Load datasets với ImageFolder (trong khối main)
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(test_dir, data_transforms['val'])
    }

    # Tạo DataLoaders (trong khối main)
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4) # num_workers=4 có thể gây lỗi nếu không có if __name__ == '__main__'
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"Tìm thấy các lớp: {', '.join(class_names)}")
    if len(class_names) != num_classes:
        print(f"Cảnh báo: Số lớp tìm thấy ({len(class_names)}) khác với num_classes ({num_classes}) đã đặt!")

    print(f"Kích thước tập huấn luyện: {dataset_sizes['train']}")
    print(f"Kích thước tập validation/test: {dataset_sizes['val']}")

    # Chuyển model lên device (trong khối main)
    model_ft = model_ft.to(device)

    # Tạo optimizer và scheduler (trong khối main, sau khi model lên device)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # --- Bắt đầu Huấn luyện ---
    print("Bắt đầu huấn luyện...")
    # Truyền device, dataloaders, dataset_sizes vào hàm train_model
    model_ft, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                    dataloaders, dataset_sizes, device, num_epochs=num_epochs)

    # --- Lưu Model đã huấn luyện ---
    model_save_path = 'resnet50_catdog_finetuned.pth'
    torch.save(model_ft.state_dict(), model_save_path)
    print(f"Model đã được lưu tại: {model_save_path}")

    # --- (Tùy chọn) Vẽ đồ thị Loss và Accuracy ---
    # Chuyển accuracy tensors trong history sang CPU để plot nếu cần
    history_plot = {k: [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in v] for k, v in history.items()}
    plot_history(history_plot)

    print("Hoàn tất!")