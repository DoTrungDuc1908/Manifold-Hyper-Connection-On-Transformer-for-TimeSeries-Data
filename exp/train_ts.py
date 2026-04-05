import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class Exp_Main:
    """
    Trình quản lý vòng lặp huấn luyện chính cho các mô hình Time Series Transformer.
    Đã được tích hợp hệ thống tracking cho mHC và xử lý đa luồng (4 inputs).
    """
    def __init__(self, args, model, train_loader, val_loader, test_loader, scaler, device, learning_rate=1e-4, tracker=None):
        self.args = args  # Lưu args để dùng cho label_len và pred_len
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.device = device
        self.learning_rate = learning_rate
        self.tracker = tracker  # Tích hợp Tracker đánh giá mHC
        self.args = args
        
        # Hàm mất mát MSE
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def prepare_batch(self, batch):
        """
        Tiền xử lý batch dữ liệu chuẩn.
        Sửa lỗi Data Leakage và đồng bộ chiều dài cho Time Marks.
        """
        if len(batch) == 4:
            batch_x, batch_y, _, _ = batch 
        else:
            batch_x, batch_y = batch

        # Đẩy dữ liệu thực tế lên device
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        B = batch_x.shape[0]

        # ==========================================
        # 1. XÂY DỰNG DECODER INPUT (144 bước)
        # ==========================================
        # Lấy label_len (48) từ CUỐI QUÁ KHỨ (batch_x) làm mồi
        dec_inp_start = batch_x[:, -self.args.label_len:, :]
        
        # Phần dự báo (96) là số 0
        dec_inp_end = torch.zeros([B, self.args.pred_len, batch_x.shape[2]]).float().to(self.device)
        
        # Ghép lại thành chuỗi dài 144
        dec_inp = torch.cat([dec_inp_start, dec_inp_end], dim=1).float().to(self.device)

        # ==========================================
        # 2. TẠO MARK THỜI GIAN KHỚP KÍCH THƯỚC
        # ==========================================
        L_x = batch_x.shape[1]    # Seq_len (96)
        L_dec = dec_inp.shape[1]  # Label_len + Pred_len (144)
        
        # Ép chuẩn 4 chiều cho Vanilla Transformer
        batch_x_mark = torch.zeros((B, L_x, 4)).float().to(self.device)
        batch_y_mark = torch.zeros((B, L_dec, 4)).float().to(self.device) # Đã khớp 144

        return batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark

    def train(self, epochs=10, patience=3):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = []
            epoch_start_time = time.time()

            for batch_idx, batch_data in enumerate(self.train_loader):
                # Sử dụng hàm prepare_batch
                batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark = self.prepare_batch(batch_data)

                self.optimizer.zero_grad()

                # Forward pass với 4 inputs
                # Kích thước đầu ra: [Batch, Pred_Len, Num_Vars]
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Cắt nhãn y chỉ lấy phần pred_len ở cuối để tính loss
                batch_y_true = batch_y[:, -self.args.pred_len:, :]
                
                loss = self.criterion(outputs, batch_y_true)
                train_loss.append(loss.item())

                loss.backward()

                # ---- TÍCH HỢP TRACKER mHC ----
                # Ghi lại Grad Norm và Amax Gain TRƯỚC KHI cập nhật trọng số
                if self.tracker is not None:
                    self.tracker.log_gradient_norm(self.model)
                    self.tracker.log_amax_gain(self.model)
                # ------------------------------

                self.optimizer.step()
                
                # Ghi lại loss vào tracker sau mỗi step
                if self.tracker is not None:
                    self.tracker.log_loss(loss.item())

            avg_train_loss = np.average(train_loss)
            val_loss = self.validate()
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch: {epoch + 1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f}")

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model_checkpoint.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Kích hoạt Early Stopping. Quá trình huấn luyện dừng sớm.")
                    break

        # Tải lại trọng số tốt nhất sau khi train xong
        self.model.load_state_dict(torch.load('best_model_checkpoint.pth'))

    def validate(self):
        self.model.eval()
        val_loss = []
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark = self.prepare_batch(batch_data)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                batch_y_true = batch_y[:, -self.args.pred_len:, :]
                
                loss = self.criterion(outputs, batch_y_true)
                val_loss.append(loss.item())

        return np.average(val_loss)

    def test(self):
        self.model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark = self.prepare_batch(batch_data)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                batch_y_true = batch_y[:, -self.args.pred_len:, :]
                
                # Chuyển tensor về numpy array
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y_true.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # 1. Tính toán MSE và MAE trên miền chuẩn hóa (Z-score)
        mse_normalized = np.mean((preds - trues) ** 2)
        mae_normalized = np.mean(np.abs(preds - trues))
        print(f"Đánh giá (Chuẩn hóa) - Test MSE: {mse_normalized:.5f} | Test MAE: {mae_normalized:.5f}")

        # 2. Tính toán trên miền giá trị thực tế (Inverse Transform)
        shape_preds = preds.shape
        preds_flat = preds.reshape(-1, shape_preds[-1])
        trues_flat = trues.reshape(-1, shape_preds[-1])
        
        preds_real = self.scaler.inverse_transform(preds_flat).reshape(shape_preds)
        trues_real = self.scaler.inverse_transform(trues_flat).reshape(shape_preds)

        mse_real = np.mean((preds_real - trues_real) ** 2)
        mae_real = np.mean(np.abs(preds_real - trues_real))
        print(f"Đánh giá (Thực tế)   - Test MSE: {mse_real:.5f} | Test MAE: {mae_real:.5f}")

        # Trả về kết quả (MI sẽ được tính độc lập bên main.py)
        return mse_normalized, mae_normalized