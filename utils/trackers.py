import torch
import torch.nn.functional as F
import numpy as np

class StabilityTracker:
    def __init__(self):
        self.loss_history = []
        self.grad_norm_history = []
        self.amax_gain_history = []
        
    def log_loss(self, loss_val):
        self.loss_history.append(loss_val)
        
    def log_gradient_norm(self, model):
        """Tính tổng L2-norm của toàn bộ gradient trong mô hình"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norm_history.append(total_norm)
        
    def log_amax_gain(self, model):
        """
        Quét qua mô hình, tìm các cấu trúc mHC và tính Amax Gain Magnitude.
        Amax = max(sum(|W_row|), sum(|W_col|))
        """
        max_gain = 0.0
        # Tùy thuộc vào tên biến ma trận trọng số trong class HyperConnections (ví dụ: 'mixing_weights')
        for name, param in model.named_parameters():
            if 'hyper' in name.lower() or 'mixing' in name.lower():
                if len(param.shape) == 2: # Nếu là ma trận vuông kết nối các luồng
                    row_sum = torch.max(torch.sum(torch.abs(param), dim=1)).item()
                    col_sum = torch.max(torch.sum(torch.abs(param), dim=0)).item()
                    gain = max(row_sum, col_sum)
                    if gain > max_gain:
                        max_gain = gain
        
        self.amax_gain_history.append(max_gain)

def compute_cosine_similarity_collapse(hidden_states_list):
    """
    hidden_states_list: List các tensor [Batch, Seq, d_model] từ các layer l, l+1,...
    Tính Cosine Similarity giữa layer đầu vào và layer hiện tại để xem biểu diễn có bị sụp đổ không.
    """
    sim_scores = []
    h_0 = hidden_states_list[0].view(-1, hidden_states_list[0].shape[-1]) # Lấy layer đầu tiên làm gốc
    
    for l in range(1, len(hidden_states_list)):
        h_l = hidden_states_list[l].view(-1, hidden_states_list[l].shape[-1])
        # Tính cosine_similarity dọc theo chiều d_model
        sim = F.cosine_similarity(h_0, h_l, dim=1).mean().item()
        sim_scores.append(sim)
        
    return sim_scores