# Deep Learning — Kho tài liệu và Bài Lab

Chào mừng đến với kho tài liệu bài lab về **Học Sâu (Deep Learning)**. Đây là bộ tài liệu được biên soạn để phục vụ việc học và nghiên cứu của sinh viên Khoa Công nghệ Thông tin, Đại học Văn Lang, chuyên ngành Trí tuệ Nhân tạo.

Các bài lab đi từ nền móng (NumPy, Pandas, PyTorch tensor) đến các kiến trúc cốt lõi (ANN, CNN, RNN, LSTM, Transformer, GAN) và Reinforcement Learning. Mỗi bài có ba phần: **lý thuyết** giải thích "tại sao" trước "công thức gì", **thực hành** để chạy thử, và **bài tập về nhà** có hint.

## Cấu trúc thư mục

```
Deep-Learning/
├── Pandas-Numpy-Matplotlib.ipynb    # Lab 0: nền móng Python — chạy trước khi vào DL
├── 1. PyTorch/
│   ├── Pytorch_basic.ipynb          # Tensor, autograd, GD thủ công
│   └── Iris.csv                     # data tham khảo
├── 2. ANN/
│   ├── ANN-LAB.ipynb                # ANN trên dữ liệu 2D không tách tuyến tính
│   └── MNIST with ANN.ipynb         # Phân loại chữ số viết tay bằng MLP
├── 3. CNN/
│   └── CNN-LAB.ipynb                # Lý thuyết tích chập + CNN trên MNIST
├── 4. RNN/
│   └── RNN.ipynb                    # Dự báo sóng sin + lý thuyết BPTT
├── 5. LSTM/
│   └── LSTM.ipynb                   # Sentiment toy, so sánh LSTM vs RNN, padding
├── 6. Transformers/
│   └── Transformers.ipynb           # Self-attention từ đầu + classifier với nn.TransformerEncoder
├── 7. GAN/
│   └── GAN.ipynb                    # GAN sinh điểm trên đường tròn + bài tập GAN-MNIST
├── 8. Reinforcement Learning/
│   └── Reinforcement_Learning.ipynb # Q-Learning trên FrozenLake + DQN trên CartPole
└── README.md
```

## Mục tiêu môn học

Sau khi hoàn thành 8 lab, sinh viên có thể:

- Tự tin sử dụng PyTorch (tensor, autograd, GPU, Dataset/DataLoader, training loop chuẩn).
- Hiểu cơ chế hoạt động của ANN, CNN, RNN, LSTM, Transformer, GAN — không chỉ "biết dùng" mà còn "biết vì sao".
- Áp dụng vào các bài thực tế: phân loại ảnh, xử lý chuỗi thời gian, NLP, sinh dữ liệu, học hành vi.
- Nhận biết và tránh các bẫy phổ biến: double-softmax, training loop hỏng, vanishing gradient, mode collapse.

## Yêu cầu

### Kiến thức cơ bản
- Python (biến, hàm, vòng lặp, list/dict).
- NumPy/Pandas cơ bản — Lab 0 sẽ ôn lại và mở rộng.
- Khái niệm Machine Learning cơ bản (loss function, gradient descent, train/test split).

### Môi trường
- **Jupyter Notebook** hoặc **Google Colab** (khuyên dùng Colab nếu không có GPU).
- Python 3.8+.
- Cài đặt thư viện:
  ```bash
  pip install torch torchvision numpy pandas matplotlib scikit-learn
  pip install gymnasium                # cho lab Reinforcement Learning
  pip install transformers datasets    # cho bài tập Transformers (tuỳ chọn)
  ```

### Phong cách làm việc
- Luôn đặt `torch.manual_seed(42)` và `np.random.seed(42)` ở đầu notebook để kết quả reproducible.
- Khi nộp bài tập về nhà, bao gồm:
  1. Code đã chạy thành công với output.
  2. Một markdown cell ngắn giải thích kết quả và quan sát.
  3. Đặt tên file: `[HoTen]_LabX_Homework.ipynb`.

## Lộ trình đề xuất (10 tuần)

| Tuần | Nội dung |
|---|---|
| 1 | Lab 0 — NumPy/Pandas/Matplotlib + Lab 1 — PyTorch basic |
| 2 | Lab 2 — ANN (cả ANN-LAB và MNIST with ANN) |
| 3 | Lab 3 — CNN |
| 4 | Lab 4 — RNN |
| 5 | Lab 5 — LSTM + Kiểm tra giữa kỳ |
| 6 | Lab 6 — Transformers |
| 7 | Lab 7 — GAN |
| 8 | Lab 8 — Reinforcement Learning |
| 9 | Ôn tập, làm dự án nhỏ |
| 10 | Thi cuối kỳ |

## Một số quy tắc khi học

1. **Đừng copy-paste mà không hiểu.** Đọc kỹ phần lý thuyết trước, mới chạy code. Code không hiểu thì gỡ ra từng dòng.
2. **Chạy đi chạy lại code.** Đổi siêu tham số xem điều gì xảy ra. Thử phá vỡ code (đổi shape, đổi loss) để hiểu lỗi xuất hiện ra sao.
3. **Vẽ mọi thứ.** Loss curve, accuracy, decision boundary, attention matrix — vẽ giúp em "nhìn thấy" model đang làm gì.
4. **Đọc error message.** PyTorch có error rất rõ ràng — đọc kỹ là 80% đã giải được.
5. **Hỏi "vì sao?".** Vì sao dùng Adam mà không SGD? Vì sao softmax + CrossEntropy thay vì sigmoid + MSE? Vì sao batch_size 64? Mỗi câu hỏi như vậy là một lớp hiểu sâu hơn.

Chúc các em học tốt và thấy được vẻ đẹp của Deep Learning.
