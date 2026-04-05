import torch
import os

def calculate_perturbation_mi(exp, test_loader, args):
    """
    Đánh giá luồng thông tin (Information Flow) bằng phương pháp Nhiễu loạn (Perturbation).
    Hàm này được gọi sau khi mô hình đã hội tụ và load trọng số tốt nhất.
    """
    print("---------------------------------------------------------")
    print("Đang tính toán Information Flow (Self-MI & Cross-MI)...")
    exp.model.eval()

    self_mi = 0.
    cross_mi = 0.
    cross_mi_mt = 0.
    
    # Ma trận đơn vị để bóc tách đường chéo (Self-MI) và phần ngoài (Cross-MI)
    eye_mask = torch.eye(args.enc_in).to(exp.device)

    with torch.no_grad():
        for i, data_batch in enumerate(test_loader):
            # 1. Chuẩn bị dữ liệu (sử dụng hàm có sẵn của Exp class)
            batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark = exp.prepare_batch(data_batch)
            
            B, L, F = batch_x.shape # Batch Size, Seq_Len, Num_Features (Biến)
            
            # 2. Xây dựng Mask vị trí nhiễu loạn cho từng biến
            feature_indices = torch.arange(F, device=exp.device).view(1, F, 1, 1).expand(B, -1, L, F)
            feature_indices_last = torch.arange(F, device=exp.device).view(1, 1, 1, F).expand(B, F, L, -1)
            mask = (feature_indices == feature_indices_last) # Kích thước: [B, F, L, F]
            
            expanded_shape = [B, F, L, F]
            
            # 3. Tạo 5 kịch bản nhiễu (Perturbation Scenarios)
            rp1 = torch.zeros(expanded_shape, device=exp.device)  # Xóa sạch (Zero)
            rp2 = torch.randn(expanded_shape, device=exp.device)  # Nhiễu trắng hoàn toàn
            rp3 = batch_x.unsqueeze(1).expand(-1, F, -1, -1)      # Gốc (Không nhiễu)
            rp4 = 0.5 * torch.randn(expanded_shape, device=exp.device) + 0.5 * rp3 # 50% nhiễu
            rp5 = 0.1 * torch.randn(expanded_shape, device=exp.device) + 0.9 * rp3 # 10% nhiễu
            
            # Mở rộng (Expand) các đầu vào phụ trợ để khớp với không gian đã nhân bản
            batch_x_mark_expand = batch_x_mark.unsqueeze(1).expand(-1, F, -1, -1).reshape(B * F, L, -1)
            dec_inp_expand = dec_inp.unsqueeze(1).expand(-1, F, -1, -1).reshape(B * F, dec_inp.shape[-2], -1)
            batch_y_mark_expand = batch_y_mark.unsqueeze(1).expand(-1, F, -1, -1).reshape(B * F, batch_y_mark.shape[-2], -1)
            
            # Tensor chứa kết quả dự báo của 5 kịch bản
            new_outputs = torch.empty(5, B * F, args.pred_len, F, device=exp.device)
            
            # 4. Chạy mô hình qua các kịch bản
            for n, rp in enumerate([rp1, rp2, rp3, rp4, rp5]):
                # Bơm dữ liệu nhiễu vào vị trí của từng biến
                x_replaced = torch.where(mask, rp, rp3)
                rp_inputs = x_replaced.view(B * F, L, F)
                
                # Dự báo (Tùy mô hình có trả về attns hay không, ta chỉ lấy output đầu tiên)
                out = exp.model(rp_inputs, batch_x_mark_expand, dec_inp_expand, batch_y_mark_expand)
                if isinstance(out, tuple):
                    out = out[0]
                new_outputs[n] = out

            # 5. Phân tích độ nhạy (Sensitivity Analysis) thông qua Độ lệch chuẩn
            # tot: [B, F, Pred_Len, F]
            tot = torch.std(new_outputs, dim=0).reshape(B, F, args.pred_len, F)
            
            # Nén trục Batch và Pred_Len lại để ra ma trận tương quan [F, F]
            sdv = tot.mean(dim=0).mean(dim=1)
            
            # 6. Bóc tách MI
            self_mi += (sdv * eye_mask).sum() / sdv.shape[0]
            cross_mi += (sdv * (1 - eye_mask)).sum() / (sdv.shape[0] * (sdv.shape[0] - 1))
            cross_mi_mt += (sdv * (1 - eye_mask))
            
    # Tính trung bình toàn bộ tập Test
    iters = len(test_loader)
    cross_mi_mt /= iters
    max_cross_mi = (cross_mi_mt * (1 - eye_mask)).max().item()
    self_mi = (self_mi / iters).item()
    cross_mi = (cross_mi / iters).item()
    
    print(f"Kết quả MI - Self MI: {self_mi:.4f} | Avg Cross MI: {cross_mi:.4f} | Max Cross MI: {max_cross_mi:.4f}")
    print("---------------------------------------------------------")
    
    return self_mi, cross_mi, max_cross_mi