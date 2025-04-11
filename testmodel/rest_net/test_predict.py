import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image # Thư viện Pillow để xử lý ảnh
import os
import json # Để lưu/tải class_names nếu cần

# --- Cấu hình ---
model_path = 'resnet50_catdog_finetuned.pth' # Đường dẫn đến model đã lưu
image_path = 'test_set\cats\cat.4001.jpg'  # <<<=== THAY ĐỔI ĐƯỜNG DẪN NÀY đến ảnh bạn muốn test
num_classes = 2 # Phải khớp với lúc train
# Quan trọng: Xác định đúng thứ tự lớp như lúc train.
# Nếu bạn dùng ImageFolder, nó tự sắp xếp theo alphabet.
# Kiểm tra lại thư mục train_set của bạn hoặc đặt lại thủ công:
class_names = ['cats', 'dogs'] # Hoặc ['dog', 'cat'] tùy theo thứ tự ImageFolder đã đọc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# --- Hàm dự đoán ---
def predict_image(model, image_path, transform, class_names, device):
    """
    Tải ảnh, biến đổi và dự đoán lớp bằng model đã cho.

    Args:
        model: Model PyTorch đã huấn luyện và load trọng số.
        image_path (str): Đường dẫn đến file ảnh.
        transform: Phép biến đổi torchvision cần áp dụng.
        class_names (list): Danh sách tên các lớp theo đúng thứ tự index.
        device: Thiết bị (CPU/GPU) để chạy dự đoán.

    Returns:
        tuple: (predicted_class_name, predicted_probability)
               hoặc (None, None) nếu có lỗi.
    """
    try:
        # Tải ảnh bằng Pillow
        img = Image.open(image_path).convert('RGB') # Đảm bảo ảnh là RGB

        # Áp dụng phép biến đổi
        img_t = transform(img)

        # Thêm chiều batch (từ [C, H, W] thành [1, C, H, W])
        batch_t = torch.unsqueeze(img_t, 0)

        # Chuyển tensor sang đúng device
        batch_t = batch_t.to(device)

        # Đặt model ở chế độ eval
        model.eval()

        # Dự đoán mà không tính gradient
        with torch.no_grad():
            outputs = model(batch_t)
            # outputs là raw scores (logits)

        # Tính xác suất bằng Softmax
        probabilities = torch.softmax(outputs, dim=1)[0] # Lấy xác suất cho ảnh đầu tiên (và duy nhất) trong batch

        # Lấy chỉ số của lớp có xác suất cao nhất
        predicted_idx = torch.argmax(probabilities).item()

        # Lấy tên lớp và xác suất tương ứng
        predicted_class_name = class_names[predicted_idx]
        predicted_probability = probabilities[predicted_idx].item()

        return predicted_class_name, predicted_probability

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại: {image_path}")
        return None, None
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        return None, None

# --- Thực thi chính ---

if __name__ == '__main__':
    # 1. Tải lại kiến trúc model (giống lúc train)
    # Không cần tải lại trọng số pre-trained gốc ở đây
    model_loaded = models.resnet50(weights=None) # Không dùng weights gốc
    num_ftrs = model_loaded.fc.in_features
    model_loaded.fc = nn.Linear(num_ftrs, num_classes) # Sửa lớp cuối

    # 2. Tải trọng số đã fine-tune
    try:
        # map_location=device giúp tải model linh hoạt giữa CPU/GPU
        model_loaded.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Trọng số model đã được tải từ {model_path}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file trọng số model: {model_path}")
        exit()
    except Exception as e:
        print(f"Lỗi khi tải trọng số model: {e}")
        # Lỗi này thường xảy ra nếu kiến trúc model hiện tại không khớp với trọng số đã lưu
        print("Hãy chắc chắn rằng kiến trúc model (đặc biệt là lớp fc cuối) giống hệt lúc bạn lưu file .pth")
        exit()

    # 3. Chuyển model sang device
    model_loaded.to(device)

    # 4. Định nghĩa phép biến đổi (giống validation/test lúc train)
    predict_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Chuẩn hóa ImageNet
    ])

    # 5. Gọi hàm dự đoán
    predicted_class, confidence = predict_image(model_loaded, image_path, predict_transform, class_names, device)

    # 6. In kết quả
    if predicted_class is not None:
        print(f"\nẢnh: {os.path.basename(image_path)}")
        print(f"Dự đoán: {predicted_class.upper()}")
        print(f"Độ tin cậy: {confidence:.4f}")

        # Hiển thị ảnh (tùy chọn)
        try:
            import matplotlib.pyplot as plt
            img_display = Image.open(image_path)
            plt.imshow(img_display)
            plt.title(f"Predicted: {predicted_class.upper()} ({confidence:.2f})")
            plt.axis('off')
            plt.show()
        except ImportError:
            print("\n(Cài đặt matplotlib để hiển thị ảnh: pip install matplotlib)")
        except Exception as e:
             print(f"Không thể hiển thị ảnh: {e}")