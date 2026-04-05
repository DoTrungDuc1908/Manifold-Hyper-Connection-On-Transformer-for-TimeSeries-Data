import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    """
    Tạo cơ chế Sliding Window cho chuỗi thời gian.
    Mỗi sample lấy ra sẽ bao gồm một đoạn lịch sử (seq_len) và nhãn dự báo (pred_len).
    """
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        # Số lượng cửa sổ có thể trượt trên toàn bộ tập dữ liệu
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        # Tạo Mark giả (toàn số 0) để tránh lỗi "index out of range"
        # x_mark và y_mark thường có cùng shape với seq_x và seq_y nhưng là thông tin thời gian
        x_mark = np.zeros_like(seq_x) 
        y_mark = np.zeros_like(seq_y)

        return (torch.tensor(seq_x, dtype=torch.float32), 
                torch.tensor(seq_y, dtype=torch.float32),
                torch.tensor(x_mark, dtype=torch.float32),
                torch.tensor(y_mark, dtype=torch.float32))

def get_data_loaders(file_path, seq_len=96, pred_len=96, batch_size=32, split_ratios=(0.7, 0.1, 0.2), target_cols=None):
    """
    Đọc dữ liệu, áp dụng Z-score Normalization, chia tập Train/Val/Test 
    và khởi tạo các PyTorch DataLoader.
    
    Args:
        file_path: Đường dẫn tới file csv dữ liệu (VD: weather.csv).
        seq_len: Độ dài cửa sổ đầu vào cho mô hình Transformer.
        pred_len: Số bước thời gian muốn dự báo ở tương lai.
        batch_size: Kích thước batch để huấn luyện.
        split_ratios: Tỷ lệ phân chia tập dữ liệu tuần tự (mặc định 70% Train, 10% Val, 20% Test).
        target_cols: (Tùy chọn) Danh sách tên cột nếu chỉ dự báo một số biến cụ thể.
    """
    
    # 1. Đọc dữ liệu
    df = pd.read_csv(file_path)
    
    # Loại bỏ cột thời gian (thường là cột dạng string/datetime ở vị trí đầu) vì Transformer sẽ học xu hướng tuần tự qua positional/token embedding.
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
        
    # Lọc biến nếu có thiết lập
    if target_cols is not None:
        df = df[target_cols]
        
    data = df.values

    # 2. Chia tập Train / Val / Test (Không shuffle để giữ tính chất thời gian)
    num_samples = len(data)
    num_train = int(num_samples * split_ratios[0])
    num_val = int(num_samples * split_ratios[1])

    train_data = data[:num_train]
    val_data = data[num_train : num_train + num_val]
    test_data = data[num_train + num_val :]

    # 3. Áp dụng Z-score Normalization
    # Lưu ý: StandardScaler() ĐƯỢC FIT ĐỘC LẬP TRÊN TẬP TRAIN để ngăn mô hình "nhìn trộm" thông tin thống kê của Val/Test.
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    # 4. Khởi tạo Dataset cho từng tập phân chia
    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len)

    # 5. Đóng gói thành DataLoader
    # Chú ý: Chỉ xáo trộn (shuffle=True) ở tập Train, các tập Val và Test cần giữ nguyên tuần tự để thuận tiện lúc vẽ đồ thị đánh giá.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Việc trả về cả bộ scaler rất quan trọng để khôi phục lại giá trị gốc (inverse_transform) khi in kết quả RMSE/MAE thật ở khâu test.
    return train_loader, val_loader, test_loader, scaler