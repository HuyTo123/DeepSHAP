import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms # bỏ models vì không dùng đến trong code này
import os
import time

print(f"PyTorch version: {torch.__version__}")

# --- Configuration ---
TRAIN_DIR = 'testmodel/training_set'
TEST_DIR = 'testmodel/test_set'
IMAGE_SIZE = 32
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15 # Có thể điều chỉnh số lượng epochs

# --- Define the Model (Simple CNN) ---
# Đặt định nghĩa lớp ra ngoài khối if __name__ == '__main__'
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

# --- Main Execution Block ---
# Khối này VẪN CẦN THIẾT để dùng num_workers > 0 trên Windows
if __name__ == '__main__':

    # Kiểm tra xem có GPU không và chọn device tương ứng
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Preprocessing ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- Load Data ---
    print("Loading datasets...")
    try:
        image_datasets = {
            'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
            'test': datasets.ImageFolder(TEST_DIR, data_transforms['test'])
        }
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục {TRAIN_DIR} hoặc {TEST_DIR}.")
        print("Hãy đảm bảo bạn đã tạo đúng cấu trúc thư mục và chạy script từ thư mục cha của 'testmodel'.")
        exit() # Thoát chương trình nếu không tìm thấy dữ liệu

    # Tạo DataLoaders
    # Sử dụng num_workers > 0 để tăng tốc độ tải dữ liệu (yêu cầu if __name__ == '__main__')
    # Nếu vẫn gặp lỗi liên quan đến multiprocessing, thử giảm num_workers hoặc đặt = 0
    num_workers_to_use = 4 if device == torch.device('cuda') else 0 # Dùng workers nếu có GPU, không thì dùng tiến trình chính
    print(f"Using num_workers = {num_workers_to_use}")
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers_to_use),
        'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers_to_use)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    # Kiểm tra nếu không có dữ liệu
    if dataset_sizes['train'] == 0 or dataset_sizes['test'] == 0:
        print("Lỗi: Một trong các tập dữ liệu (train/test) không chứa ảnh nào.")
        print(f"Kiểm tra lại các thư mục con 'cats' và 'dogs' trong {TRAIN_DIR} và {TEST_DIR}.")
        exit()

    print(f"Training set size: {dataset_sizes['train']}")
    print(f"Test set size: {dataset_sizes['test']}")
    print(f"Classes found: {class_names}")
    print(f"Class mapping: {image_datasets['train'].class_to_idx}") # Sẽ là {'cats': 0, 'dogs': 1}

    # --- Initialize Model, Loss Function, Optimizer ---
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    print("\nModel Architecture:")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("\nStarting Training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Lặp qua dữ liệu
            batch_count = 0
            for inputs, labels in dataloaders[phase]:
                batch_count += 1
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

            if dataset_sizes[phase] == 0:
                 print(f"Skipping {phase} phase - dataset size is zero.")
                 continue # Bỏ qua phase nếu không có dữ liệu

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print() # Xuống dòng sau mỗi epoch

    # --- End Training ---
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # --- (Optional) Save the trained model ---
    os.makedirs('saved_models', exist_ok=True)
    model_save_path = 'saved_models/cats_dogs_model_32x32.pth'
    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # --- (Optional) Load the model for inference later ---
    # try:
    #     model_loaded = SimpleCNN(num_classes=len(class_names))
    #     model_loaded.load_state_dict(torch.load(model_save_path))
    #     model_loaded.to(device)
    #     model_loaded.eval()
    #     print("Model loaded successfully for inference.")
    # except Exception as e:
    #     print(f"Error loading model: {e}")