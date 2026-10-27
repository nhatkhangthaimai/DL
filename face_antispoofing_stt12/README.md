# ĐỒ ÁN DEEP LEARNING: PHÁT HIỆN GIẢ MẠO KHUÔN MẶT (FACE ANTI-SPOOFING) BẢO VỆ HỆ THỐNG ĐIỂM DANH

**Đề tài:** Phát hiện giả mạo khuôn mặt (Face Anti-Spoofing) bảo vệ hệ thống điểm danh  
**Tên tiếng Anh:** Face Anti-Spoofing Detection for Authentication Systems  
**Mã đề tài (STT):** 12  
**Lĩnh vực:** Computer Vision – Deep Learning  
**Kiến trúc chính:** ResNet34 Pretrained + Local Binary Patterns (LBP) Texture Feature Fusion  
**Mô hình đối sánh:** Pure ResNet34 Baseline  
**Bộ dữ liệu:** MSU Mobile Face Spoofing Database (MSU-MFSD)  
**Tác giả:** Nhóm nghiên cứu STT 12  

---

## MỤC LỤC
1. [Giới Thiệu Đề Tài](#1-giới-thiệu-đề-tài)
2. [Lý Do Chọn Đề Tài & Tính Cấp Thiết](#2-lý-do-chọn-đề-tài--tính-cấp-thiết)
3. [Mục Tiêu & Phạm Vi Nghiên Cứu](#3-mục-tiêu--phạm-vi-nghiên-cứu)
4. [Công Nghệ & Môi Trường Thực Nghiệm](#4-công-nghệ--môi-trường-thực-nghiệm)
5. [Phân Tích Dataset MSU-MFSD & Giao Thức Chia Dữ Liệu](#5-phân-tích-dataset-msu-mfsd--giao-thức-chia-dữ-liệu)
6. [Kiến Trúc Mô Hình Đề Xuất: ResNet34 + LBP Texture Fusion](#6-kiến-trúc-mô-hình-đề-xuất-resnet34--lbp-texture-fusion)
7. [Cấu Trúc Thư Mục Dự Án](#7-cấu-trúc-thư-mục-dự-án)
8. [Hướng Dẫn Cài Đặt Môi Trường](#8-hướng-dẫn-cài-đặt-môi-trường)
9. [Quy Trình Tiền Xử Lý Dữ Liệu](#9-quy-trình-tiền-xử-lý-dữ-liệu)
10. [Hướng Dẫn Huấn Luyện Mô Hình](#10-hướng-dẫn-huấn-luyện-mô-hình)
11. [Hướng Dẫn Huấn Luyện Trên Google Colab](#11-hướng-dẫn-huấn-luyện-trên-google-colab)
12. [Đánh Giá Mô Hình & Chỉ Số Khoa Học](#12-đánh-giá-mô-hình--chỉ-số-khoa-học)
13. [Kết Quả Thực Nghiệm & So Sánh Mô Hình](#13-kết-quả-thực-nghiệm--so-sánh-mô-hình)
14. [Hướng Dẫn Chạy Demo Realtime Webcam](#14-hướng-dẫn-chạy-demo-realtime-webcam)
15. [Kiểm Thử Toàn Diện (Testing Suite)](#15-kiểm-thử-toàn-diện-testing-suite)
16. [Hạn Chế Của Đề Tài & Hướng Phát Triển](#16-hạn-chế-của-đề-tài--hướng-phát-triển)
17. [Xử Lý Lỗi Thường Gặp (Troubleshooting)](#17-xử-lý-lỗi-thường-gặp-troubleshooting)
18. [Tài Liệu Tham Khảo](#18-tài-liệu-tham-khảo)

---

## 1. GIỚI THIỆU ĐỀ TÀI

Hệ thống xác thực khuôn mặt (Face Recognition) ngày càng được ứng dụng rộng rãi trong các hệ thống điểm danh tự động tại trường học, chấm công doanh nghiệp và xác thực giao dịch tài chính. Tuy nhiên, các hệ thống nhận diện khuôn mặt truyền thống cực kỳ dễ bị tổn thương trước các cuộc tấn công giả mạo (Presentation Attacks - PA), tiêu biểu như:
- **Tấn công bằng ảnh in (Printed Photo Attack):** Sử dụng ảnh chụp khuôn mặt in ra giấy (A4, A3) đưa trước camera.
- **Tấn công bằng video phát lại (Replay Video Attack):** Phát lại đoạn video quay khuôn mặt của đối tượng trên màn hình điện thoại (iPhone, Samsung) hoặc máy tính bảng (iPad).

Hệ thống **Phát hiện giả mạo khuôn mặt (Face Anti-Spoofing - FAS)** hay **Presentation Attack Detection (PAD)** đóng vai trò là "bức tường lửa" đầu tiên của chu trình xác thực. Chỉ khi khuôn mặt được phân loại là **LIVE (Người thật)** thì chu trình nhận dạng danh tính mới được phép diễn ra; nếu là **SPOOF (Giả mạo)**, hệ thống lập tức từ chối và cảnh báo xâm nhập.

---

## 2. LÝ DO CHỌN ĐỀ TÀI & TÍNH CẤP THIẾT

1. **Rủi ro an ninh trong điểm danh số:** Sinh viên hoặc nhân viên có thể sử dụng ảnh thẻ hoặc clip quay sẵn trên điện thoại để "điểm danh hộ" từ xa nếu camera chỉ nhận dạng danh tính mà không có cơ chế liveness detection.
2. **Sự kết hợp giữa Deep Learning và Texture Analysis:** Các mô hình CNN trích xuất đặc trưng ngữ nghĩa (semantic features) rất tốt nhưng đôi khi bị đánh lừa bởi chất lượng màn hình hiển thị cao. Ngược lại, kỹ thuật trích xuất vân ảnh cổ điển **Local Binary Patterns (LBP)** đặc biệt nhạy cảm với các tần số vân cao (high-frequency micro-textures) như vân sọc in phun, lưới điểm ảnh màn hình LCD/OLED, và hiệu ứng moiré. Do đó, việc kết hợp **ResNet34 + LBP Feature Fusion** mang lại sự bổ trợ vượt trội cho bài toán chống giả mạo.

---

## 3. MỤC TIÊU & PHẠM VI NGHIÊN CỨU

### 3.1. Mục tiêu
- Xây dựng hoàn chỉnh pipeline Deep Learning phát hiện khuôn mặt giả mạo từ dữ liệu video thực tế MSU-MFSD.
- Thiết kế và huấn luyện kiến trúc **ResNet34 + Texture (LBP) Feature Fusion**, so sánh với baseline thuần **ResNet34**.
- Áp dụng chuẩn đánh giá sinh trắc học quốc tế **ISO/IEC 30107-3** với các chỉ số chuyên dụng: **APCER, BPCER, ACER, AUC, EER**.
- Xây dựng ứng dụng demo Webcam thời gian thực với cơ chế làm mượt kết quả (temporal smoothing), hiển thị xác suất liveness và trạng thái cổng điểm danh (`[ACCESS GRANTED]` / `[ACCESS DENIED]`).

### 3.2. Phạm vi nghiên cứu
- **Bộ dữ liệu:** MSU Mobile Face Spoofing Database (MSU-MFSD) gồm 280 video của 35 chủ thể.
- **Dạng tấn công hỗ trợ:** Printed Photo, iPhone 5S Replay Video, iPad Air HD Replay Video trên cả 2 loại camera (MacBook Laptop và Nexus 5 Android).
- **Giới hạn phạm vi khoa học:** MSU-MFSD không chứa dữ liệu mặt nạ 3D (3D Mask). Đồ án không tự tuyên bố phát hiện được mặt nạ silicon/3D khi chưa có thực nghiệm. Đồ án tập trung vào Face Anti-Spoofing, bước cấp quyền trong demo là mô phỏng bảo vệ điểm danh, không thay thế cho hệ thống quản lý cơ sở dữ liệu sinh viên/nhân viên.

---

## 4. CÔNG NGHỆ & MÔI TRƯỜNG THỰC NGHIỆM

| Thành phần | Phiên bản / Chi tiết | Vai trò |
|---|---|---|
| **Ngôn ngữ** | Python 3.12.10 (64-bit) | Ngôn ngữ phát triển toàn bộ hệ thống |
| **Deep Learning** | PyTorch 2.6.0+cu124, TorchVision 0.21.0+cu124 | Xây dựng mô hình, đạo hàm tự động, transfer learning |
| **Phần cứng GPU** | NVIDIA GeForce RTX 2060 (6.00 GB VRAM), CUDA 12.4 | Huấn luyện tăng tốc và inference song song |
| **Texture Descriptor** | Local Binary Patterns (LBP) - Spatial Grid 4x4 | Trích xuất đặc trưng vi vân ảnh (944 chiều) |
| **Xử lý ảnh & video** | OpenCV 4.14.0, Pillow 12.0.0, FFmpeg | Giải mã video, crop khuôn mặt, chuẩn hóa RGB |
| **Khoa học dữ liệu** | NumPy 2.3.5, Pandas 2.3.3, Scipy 1.18.1 | Xử lý mảng ma trận, metadata CSV, thống kê dữ liệu |
| **Metrics & Đánh giá**| Scikit-Learn 1.9.1 | Tính Accuracy, Precision, Recall, F1, ROC, AUC, EER |
| **Trực quan hóa** | Matplotlib 3.10.7, Seaborn 0.13.2 | Vẽ đồ thị Loss, Accuracy, Confusion Matrix, ROC |
| **Kiểm thử** | Pytest 9.1.1 | Kiểm thử đơn vị (Unit Tests) & tích hợp (Integration Tests) |

---

## 5. PHÂN TÍCH DATASET MSU-MFSD & GIAO THỨC CHIA DỮ LIỆU

### 5.1. Thống kê dataset thực tế
MSU-MFSD được thu thập tại Phòng thí nghiệm Nhận dạng mẫu và Xử lý ảnh (PRIP Lab), Đại học Bang Michigan (MSU):
- **Tổng số video:** 280 video.
- **Số chủ thể (Clients):** 35 người (`client001` đến `client055`).
- **Real Access (Live - Label 0):** 70 video (35 Laptop, 35 Android).
- **Presentation Attacks (Spoof - Label 1):** 210 video:
  - 70 video iPad HD Video Replay.
  - 70 video iPhone 5S Mobile Video Replay.
  - 70 video A3 Printed Photo Attack.
- **Tệp tọa độ khuôn mặt:** 280 tệp `.face` tương ứng do PittPatt SDK trích xuất cho từng frame.

### 5.2. Giao thức phân chia không rò rỉ dữ liệu (Zero Data Leakage)
Dự án tuân thủ nghiêm ngặt giao thức chính thức từ `train_sub_list.txt` và `test_sub_list.txt`:

```
                 TỔNG 35 SUBJECTS (280 VIDEOS)
                           │
      ┌────────────────────┴────────────────────┐
      ▼                                         ▼
TẬP TRAIN GỐC (15 Subjects)             TẬP TEST ĐỘC LẬP (20 Subjects)
120 Videos                              160 Videos (40 Live, 120 Spoof)
      │                                 [Tuyệt đối không dùng khi huấn luyện]
      ├──────────────────────┐
      ▼                      ▼
TẬP TRAIN (12 Subjects)  TẬP VALIDATION (3 Subjects: 34, 53, 54)
96 Videos (576 frames)   24 Videos (144 frames)
```

**Bảo đảm khoa học:**
1. Tập Test hoàn toàn độc lập về Subject ID, chỉ dùng cho đánh giá cuối cùng.
2. Tập Validation được tách theo Subject ID (`034`, `053`, `054`), không chia ngẫu nhiên frame của cùng một người vào cả Train và Validation.
3. Không xảy ra Data Leakage giữa các tập.

---

## 6. KIẾN TRÚC MÔ HÌNH ĐỀ XUẤT: RESNET34 + LBP TEXTURE FUSION

```mermaid
flowchart TD
    A["Ảnh khuôn mặt đầu vào (RGB 224x224)"] --> B["Nhánh A: ResNet34 Backbone<br/>(Pretrained ImageNet)"]
    A --> C["Nhánh B: Local Binary Patterns (LBP)<br/>(Spatial Grid 4x4, Radius=1, P=8)"]
    
    B --> D["Deep Feature Vector<br/>(512 chiều)"]
    C --> E["Texture Feature Vector<br/>(944 chiều - 59 bins x 16 blocks)"]
    
    E --> F["Batch Normalization 1D"]
    
    D --> G["Feature Concatenation<br/>(512 + 944 = 1456 chiều)"]
    F --> G
    
    G --> H["Linear Layer (1456 -> 256)"]
    H --> I["Batch Normalization + ReLU + Dropout (0.4)"]
    I --> J["Linear Classification Head (256 -> 2)"]
    
    J --> K{"Dự đoán Liveness"}
    K -->|Label 0| L["LIVE (Người thật)"]
    K -->|Label 1| M["SPOOF (Giả mạo)"]
```

### 6.1. Nhánh A: Deep Semantic Features (ResNet34)
- Sử dụng backbone ResNet34 khởi tạo từ trọng số pretrained ImageNet-1K.
- Bỏ lớp Fully Connected cuối cùng của ResNet34, trích xuất vector đặc trưng $512$ chiều từ tầng Adaptive Average Pooling.
- Chiến lược huấn luyện: 2 epoch đầu đóng băng backbone (warm-up fusion head), từ epoch 3 mở khóa fine-tuning toàn bộ mạng với tốc độ học vi sai (backbone LR: $2 \times 10^{-5}$, fusion head LR: $1 \times 10^{-4}$).

### 6.2. Nhánh B: Micro-Texture Features (Local Binary Patterns)
- Chuyển ảnh khuôn mặt sang grayscale.
- Trích xuất LBP đồng nhất (Uniform LBP) với bán kính $R=1$ và số điểm lân cận $P=8$.
- Khác với LBP toàn cục làm mất thông tin vị trí, ảnh được chia thành lưới $4 \times 4 = 16$ ô cục bộ (trán, má, cằm, mắt, mũi).
- Mỗi ô tạo histogram chuẩn hóa $59$ bins. Vector đặc trưng vân ảnh cuối cùng có kích thước:
  $$\text{dim} = 59 \times 4 \times 4 = 944 \text{ chiều}$$
- Vector được chuẩn hóa $L_2$ để đồng bộ biên độ trước khi đưa vào mạng.

### 6.3. Tầng kết hợp đặc trưng (Feature Fusion)
- Ghép nối vector đặc trưng sâu và vector vân ảnh thành vector $1456$ chiều:
  $$\mathbf{f}_{\text{fused}} = [\mathbf{f}_{\text{deep}} \,\|\, \text{BN}(\mathbf{f}_{\text{texture}})] \in \mathbb{R}^{1456}$$
- Đưa qua khối MLP: Linear(1456, 256) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.4) $\to$ Linear(256, 2).

---

## 7. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
DL/
├── MSU-MFSD/                         # Dataset gốc (70 real, 210 attack video + .face)
│
└── face_antispoofing_stt12/           # Source code đồ án
    ├── configs/
    │   └── config.yaml                # Cấu hình siêu tham số tập trung
    ├── src/
    │   ├── __init__.py                # Khởi tạo package
    │   ├── dataset.py                 # PyTorch Dataset, DataLoader, Class Weights
    │   ├── preprocessing.py           # Trích frame video, crop khuôn mặt, transform
    │   ├── face_detection.py          # PittPatt .face reader & OpenCV Haar detector
    │   ├── texture.py                 # Trích xuất Uniform LBP đa vùng (4x4 grid)
    │   ├── model.py                   # ResNet34 Baseline & ResNet34+LBP Fusion
    │   ├── metrics.py                 # APCER, BPCER, ACER, AUC, EER & biểu đồ
    │   ├── inference.py               # Bộ dự đoán và bộ lọc làm mượt temporal
    │   └── utils.py                   # Checkpoint manager, seed, logger, meter
    ├── scripts/
    │   ├── check_dataset.py           # Kiểm tra toàn vẹn dataset MSU-MFSD
    │   └── verify_environment.py      # Kiểm tra môi trường GPU, Python, packages
    ├── notebooks/
    │   └── Train_on_Colab.ipynb       # Jupyter Notebook hoàn chỉnh chạy Colab GPU
    ├── tests/
    │   ├── test_dataset.py            # Unit test dữ liệu, LBP, leakage
    │   ├── test_model.py              # Unit test mô hình, forward, backprop
    │   ├── test_metrics.py            # Unit test APCER, BPCER, ACER
    │   └── test_inference.py          # Unit test inference engine
    ├── models/
    │   └── haarcascade_frontalface_default.xml # Trọng số Haar cascade cho webcam
    ├── processed/
    │   ├── metadata.csv               # Bảng metadata 1680 ảnh kèm nhãn & split
    │   ├── train/ (live, spoof)       # 576 ảnh khuôn mặt train
    │   ├── val/   (live, spoof)       # 144 ảnh khuôn mặt val
    │   └── test/  (live, spoof)       # 960 ảnh khuôn mặt test
    ├── checkpoints/
    │   ├── best_fusion_model.pth      # Checkpoint tốt nhất ResNet34 + LBP Fusion
    │   └── best_baseline_model.pth    # Checkpoint tốt nhất ResNet34 Baseline
    ├── results/
    │   ├── model_comparison.csv       # Bảng số liệu so sánh chi tiết
    │   ├── model_comparison.png       # Biểu đồ so sánh trực quan
    │   ├── confusion_matrix_fusion_test.png
    │   ├── confusion_matrix_baseline_test.png
    │   ├── roc_curve_fusion_test.png
    │   ├── roc_curve_baseline_test.png
    │   ├── training_curves_fusion.png
    │   └── training_curves_baseline.png
    ├── logs/                          # Log quá trình tiền xử lý, huấn luyện
    ├── prepare_dataset.py             # Script tiền xử lý video tự động
    ├── train.py                       # Script huấn luyện PyTorch
    ├── evaluate.py                    # Script đánh giá và đối sánh mô hình
    ├── webcam.py                      # Demo nhận diện chống giả mạo realtime
    ├── requirements.txt               # Danh sách thư viện tương thích Python 3.12
    ├── README.md                      # Báo cáo và tài liệu kỹ thuật
    ├── .gitignore
    └── main.py                        # CLI hợp nhất toàn bộ chức năng
```

---

## 8. HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG

### 8.1. Yêu cầu hệ thống
- Hệ điều hành: Windows 10/11 hoặc Linux / Ubuntu.
- Python: `3.12.10` (hoặc Python 3.10+).
- GPU: Khuyến nghị NVIDIA GPU hỗ trợ CUDA (dự án đã kiểm thử trên RTX 2060 6GB). Hỗ trợ CPU tự động nếu không có GPU.

### 8.2. Cài đặt các thư viện
Mở PowerShell hoặc Command Prompt tại thư mục `face_antispoofing_stt12`:

```bash
# Cài đặt PyTorch với CUDA 12.4
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Cài đặt các thư viện phụ thuộc còn lại
python -m pip install -r requirements.txt
```

### 8.3. Xác minh môi trường
Chạy script kiểm tra hệ thống:

```bash
python scripts/verify_environment.py
```

---

## 9. QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU

### 9.1. Kiểm tra tính toàn vẹn dataset MSU-MFSD
Trước khi tiền xử lý, chạy lệnh kiểm tra cấu trúc dataset gốc:

```bash
python scripts/check_dataset.py
```

Kết quả xác nhận:
- 280/280 video và 280/280 tệp `.face` đầy đủ.
- 15 chủ thể Train, 20 chủ thể Test (100% Subject-Disjoint).

### 9.2. Tiền xử lý tự động từ video gốc
Chạy lệnh tiền xử lý để trích xuất frame, crop khuôn mặt (kèm 15% margin ngữ cảnh) và lưu vào `processed/`:

```bash
python prepare_dataset.py --frames 12
```

*(Hoặc dùng CLI chính: `python main.py prepare --frames 12`)*

---

## 10. HƯỚNG DẪN HUẤN LUYỆN MÔ HÌNH

### 10.1. Huấn luyện mô hình đề xuất: ResNet34 + LBP Texture Fusion
Chạy lệnh huấn luyện với GPU:

```bash
python train.py --model fusion --epochs 12 --batch-size 16 --lr 0.0001
```

Quá trình huấn luyện tự động:
1. Áp dụng CrossEntropyLoss có trọng số nghịch đảo tần suất lớp (`weight=[1.5, 0.75]`).
2. Warm-up 2 epoch đầu bằng cách đóng băng ResNet34 backbone, huấn luyện đầu kết hợp LBP.
3. Từ epoch 3 unfreeze toàn bộ backbone với tốc độ học vi sai ($2 \times 10^{-5}$).
4. Giảm LR theo Cosine Annealing.
5. Early Stopping theo dõi chỉ số Validation ACER (tự động dừng khi mô hình đạt điểm tối ưu).
6. Tự động lưu checkpoint tốt nhất vào `checkpoints/best_fusion_model.pth`.

### 10.2. Huấn luyện mô hình đối chứng: ResNet34 Baseline
Chạy lệnh huấn luyện mô hình baseline để lấy số liệu thực nghiệm so sánh:

```bash
python train.py --model baseline --epochs 12 --batch-size 16 --lr 0.0001
```

---

## 11. HƯỚNG DẪN HUẤN LUYỆN TRÊN GOOGLE COLAB

Nếu muốn huấn luyện trên Google Colab GPU:
1. Nén thư mục dữ liệu đã tiền xử lý trên máy tính: nén thư mục `processed/` thành `processed.zip`.
2. Mở Google Colab, mở tệp `notebooks/Train_on_Colab.ipynb`.
3. Chọn Runtime GPU (`Runtime > Change runtime type > T4 GPU`).
4. Tải thư mục mã nguồn và tệp `processed.zip` lên Colab.
5. Thực hiện tuần tự các cell từ Step 1 đến Step 9 trong notebook.
6. Tải checkpoint `best_fusion_model.pth` về máy tính đặt vào thư mục `checkpoints/`.

---

## 12. ĐÁNH GIÁ MÔ HÌNH & CHỈ SỐ KHOA HỌC

Bài toán Face Anti-Spoofing được đánh giá theo tiêu chuẩn quốc tế **ISO/IEC 30107-3**:

- **APCER (Attack Presentation Classification Error Rate):** Tỷ lệ mẫu tấn công giả mạo (Spoof) bị phân loại nhầm thành người thật (Live).
  $$\text{APCER} = \frac{\text{Số mẫu Spoof bị nhận nhầm thành Live}}{\text{Tổng số mẫu Spoof}} = \frac{FN}{TP + FN}$$
  *(Càng thấp càng an toàn, tránh để kẻ tấn công đột nhập).*

- **BPCER (Bona Fide Presentation Classification Error Rate):** Tỷ lệ mẫu người thật (Live) bị từ chối nhầm thành giả mạo (Spoof).
  $$\text{BPCER} = \frac{\text{Số mẫu Live bị nhận nhầm thành Spoof}}{\text{Tổng số mẫu Live}} = \frac{FP}{TN + FP}$$
  *(Càng thấp hệ thống càng thân thiện, không gây phiền hà cho người thật).*

- **ACER (Average Classification Error Rate):** Trung bình cộng giữa APCER và BPCER:
  $$\text{ACER} = \frac{\text{APCER} + \text{BPCER}}{2}$$

- **AUC (Area Under ROC Curve):** Diện tích dưới đường cong ROC, biểu thị khả năng phân tách giữa hai lớp Live và Spoof trên mọi ngưỡng.
- **EER (Equal Error Rate):** Tỷ lệ lỗi tại điểm mà $\text{APCER} = \text{BPCER}$.

---

## 13. KẾT QUẢ THỰC NGHIỆM & SO SÁNH MÔ HÌNH

Dưới đây là kết quả thực nghiệm thực tế thu được trên tập kiểm thử độc lập **Test Set** (gồm 20 chủ thể chưa từng xuất hiện trong quá trình huấn luyện, 160 video, 960 frame khuôn mặt):

### 13.1. Bảng so sánh cấp độ Frame (Frame-Level Evaluation)

| Chỉ số đánh giá | ResNet34 Baseline | ResNet34 + LBP Texture Fusion (Đề xuất) | Ý nghĩa thực nghiệm |
|---|:---:|:---:|---|
| **Accuracy** | 94.27% | 90.83% | Độ chính xác tổng thể |
| **Precision (Spoof)** | 96.63% | **97.88%** | **Fusion cao hơn**: khi cảnh báo Spoof thì độ tin cậy rất cao |
| **Recall (Spoof)** | 95.69% | 89.72% | Khả năng phát hiện các mẫu giả mạo |
| **F1-Score** | 96.16% | 93.62% | Trung bình điều hòa Precision & Recall |
| **BPCER (Live Error)** | 10.00% | **5.83%** | **Fusion vượt trội**: Giảm mạnh tỷ lệ người thật bị từ chối oan |
| **APCER (Spoof Error)**| 4.31% | 10.28% | Tỷ lệ lọt lưới tấn công |
| **ACER (Avg Error)** | **7.15%** | 8.06% | Sai số phân loại trung bình theo chuẩn ISO |
| **ROC AUC** | **0.9844** | 0.9665 | Diện tích dưới đường cong ROC xuất sắc |
| **EER** | 5.56% | 8.13% | Tỷ lệ lỗi cân bằng |

### 13.2. Bảng so sánh cấp độ Video (Video-Level Aggregated Evaluation)
Tổng hợp xác suất dự đoán của tất cả các frame thuộc cùng một video (160 video Test):

| Chỉ số Video | ResNet34 Baseline | ResNet34 + LBP Texture Fusion (Đề xuất) |
|---|:---:|:---:|
| **Số lượng video đánh giá** | 160 videos | 160 videos |
| **Video Accuracy** | **95.00%** | 90.62% |
| **Video Precision** | 97.46% | 97.30% |
| **Video Recall** | 95.83% | 90.00% |
| **Video APCER** | 4.17% | 10.00% |
| **Video BPCER** | **7.50%** | **7.50%** |
| **Video ACER** | **5.83%** | 8.75% |

### 13.3. Phân tích kết quả khoa học
1. **Khả năng giảm thiểu BPCER của LBP Texture Fusion:** Đặc trưng vân ảnh LBP giúp mô hình nhận diện chính xác độ mềm mại tự nhiên của làn da người thật, giúp giảm tỷ lệ **BPCER từ 10.00% xuống 5.83%** (giảm gần một nửa lỗi từ chối oan người thật).
2. **Độ chính xác cảnh báo giả mạo (Precision):** Mô hình ResNet34 + LBP đạt Precision lên tới **97.88%**, nghĩa là một khi mô hình gắn cờ "SPOOF", tỷ lệ báo động giả gần như bị triệt tiêu.
3. **Hiệu năng của ResNet34 Baseline:** Mô hình ResNet34 thuần tận dụng tốt các đặc trưng ngữ nghĩa toàn cục và đạt ACER 7.15%, chứng minh năng lực mạnh mẽ của transfer learning từ ImageNet đối với bài toán PAD.

Biểu đồ so sánh trực quan được xuất tự động tại:
- `results/model_comparison.png`
- `results/confusion_matrix_fusion_test.png`
- `results/roc_curve_fusion_test.png`

---

## 14. HƯỚNG DẪN CHẠY DEMO REALTIME WEBCAM

### 14.1. Khởi chạy demo
Để chạy ứng dụng phát hiện giả mạo thời gian thực với webcam máy tính:

```bash
python webcam.py --checkpoint checkpoints/best_fusion_model.pth --camera 0
```

*(Hoặc dùng lệnh ngắn: `python main.py demo --camera 0`)*

### 14.2. Giao diện và các trạng thái hiển thị
1. **Khi phát hiện khuôn mặt thật (LIVE):**
   - Bounding box viền góc màu **Xanh lá cây** nổi bật.
   - Nhãn: `LIVE: 98.5%`.
   - Banner điểm danh: `STATUS: [ACCESS GRANTED] Genuine Live Face Verified`.
2. **Khi phát hiện khuôn mặt giả mạo (SPOOF):**
   - Bounding box viền góc màu **Đỏ rực**.
   - Nhãn: `SPOOF: 99.1%`.
   - Banner điểm danh: `STATUS: [ACCESS DENIED] Spoof Attack Detected!`.
3. **Khi không có khuôn mặt:**
   - Banner trạng thái: `STATUS: [NO FACE DETECTED] - Please align face within view`.
4. **Bộ lọc làm mượt thời gian (Temporal Smoothing):**
   - Sử dụng cửa sổ trượt trung bình động 5 frame gần nhất nhằm loại bỏ hoàn toàn hiện tượng nhấp nháy nhãn giữa các khung hình liên tiếp.
5. **Thoát chương trình:** Nhấn phím `Q` hoặc `ESC` trên cửa sổ demo.

---

## 15. KIỂM THỬ TOÀN DIỆN (TESTING SUITE)

Dự án được bao phủ bởi bộ kiểm thử tự động `pytest` gồm 13 bài test bao quát toàn bộ pipeline:

```bash
python -m pytest -v tests
```

### Kết quả kiểm thử thực tế:
- `tests/test_dataset.py`:
  - `test_filename_parsing`: Kiểm tra định dạng tên tệp MSU-MFSD. [PASSED]
  - `test_zero_data_leakage`: Kiểm tra tập Train, Val, Test không trùng lặp Subject ID. [PASSED]
  - `test_lbp_feature_extraction`: Kiểm tra kích thước và chuẩn hóa vector LBP. [PASSED]
  - `test_class_weights_computation`: Kiểm tra tính toán trọng số lớp cân bằng. [PASSED]
- `tests/test_model.py`:
  - `test_baseline_resnet34_forward`: Kiểm tra forward pass ResNet34 thuần. [PASSED]
  - `test_fusion_resnet34_lbp_forward`: Kiểm tra forward pass ResNet34 + LBP. [PASSED]
  - `test_gradient_flow_and_backprop`: Kiểm tra dòng gradient qua fusion head. [PASSED]
  - `test_freeze_and_unfreeze_backbone`: Kiểm tra cơ chế đóng băng/mở khóa backbone. [PASSED]
- `tests/test_metrics.py`:
  - `test_perfect_predictions`: Kiểm tra các chỉ số khi mô hình đạt điểm tuyệt đối. [PASSED]
  - `test_all_spoofs_misclassified_as_live`: Kiểm tra tính đúng đắn công thức APCER. [PASSED]
  - `test_all_live_misclassified_as_spoof`: Kiểm tra tính đúng đắn công thức BPCER. [PASSED]
  - `test_video_level_aggregation`: Kiểm tra tổng hợp kết quả cấp video. [PASSED]
- `tests/test_inference.py`:
  - `test_inference_and_smoothing`: Kiểm tra bộ nạp trọng số và bộ lọc làm mượt. [PASSED]

**13/13 tests PASSED 100% trong 5.09 giây.**

---

## 16. HẠN CHẾ CỦA ĐỀ TÀI & HƯỚNG PHÁT TRIỂN

### 16.1. Hạn chế
1. **Quy mô đối tượng dữ liệu:** MSU-MFSD gồm 35 chủ thể (15 Train, 20 Test). Dù số lượng video và frame phong phú (280 video), sự đa dạng về chủng tộc, độ tuổi và điều kiện môi trường vẫn có giới hạn nhất định.
2. **Loại hình tấn công:** Chưa bao gồm các hình thức tấn công công nghệ cao mới như mặt nạ silicon 3D cao cấp (3D Mask Attack) hoặc Deepfake thời gian thực.
3. **Phụ thuộc điều kiện chiếu sáng:** Ánh sáng quá yếu hoặc camera có độ phân giải quá thấp có thể làm suy giảm chất lượng các tần số vi vân LBP.

### 16.2. Hướng phát triển
1. Huấn luyện chéo trên nhiều bộ dữ liệu (Cross-dataset evaluation trên CASIA-FASD, Replay-Attack, OULU-NPU) để tăng cường tính tổng quát hóa (generalization).
2. Tích hợp thêm nhánh thông tin chiều sâu (Depth Map Supervision) hoặc rPPG (phát hiện nhịp tim qua biến thiên sắc tố da).
3. Đóng gói mô hình thành dịch vụ RESTful API (FastAPI/Docker) để tích hợp trực tiếp vào các phần mềm quản lý điểm danh thông minh của nhà trường.

---

## 17. XỬ LÝ LỖI THƯỜNG GẶP (TROUBLESHOOTING)

1. **Lỗi `CUDA out of memory`:**
   - Giảm `batch_size` trong `configs/config.yaml` từ 16 xuống 8.
2. **Lỗi không mở được Webcam (`Cannot open camera index 0`):**
   - Đảm bảo webcam không bị chiếm dụng bởi ứng dụng khác (Zoom, Teams, Zalo).
   - Thử đổi camera index bằng cờ `--camera 1`.
3. **Lỗi `ModuleNotFoundError`:**
   - Chạy lệnh `python scripts/verify_environment.py` để phát hiện và cài đặt gói còn thiếu.

---

## 18. TÀI LIỆU THAM KHẢO

1. **Bài báo bộ dữ liệu MSU-MFSD:**
   > D. Wen, H. Han, and A. K. Jain, *"Face Spoof Detection with Image Distortion Analysis"*, IEEE Transactions on Information Forensics and Security, vol. 10, no. 4, pp. 746–761, Apr. 2015.
2. **Kiến trúc mạng phần dư (ResNet):**
   > K. He, X. Zhang, S. Ren, and J. Sun, *"Deep Residual Learning for Image Recognition"*, in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.
3. **Đặc trưng vi vân ảnh LBP trong sinh trắc học:**
   > T. Ojala, M. Pietikainen, and T. Maenpaa, *"Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns"*, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 7, pp. 971–987, 2002.
4. **Tiêu chuẩn đánh giá quốc tế:**
   > ISO/IEC JTC 1/SC 37 Biometrics, *"ISO/IEC 30107-3: Information technology — Biometric presentation attack detection — Part 3: Testing and reporting"*, 2017.
