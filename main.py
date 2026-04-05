import argparse
import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import importlib

# Import các công cụ đánh giá
from utils.trackers import StabilityTracker
from utils.visualizer import plot_training_stability
from utils.metrics_mi import calculate_perturbation_mi

# Cấu trúc Data Loader của bạn
from data_provider.data_loader import get_data_loaders

# Quản lý thực nghiệm
from exp.train_ts import Exp_Main

def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting Pipeline: Auto-run and Evaluate')

    # Data Loader Configs
    parser.add_argument('--data_path', type=str, required=True, help='Đường dẫn tới file csv')
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='Loại bài toán')
    
    # Time Sequence Lenths
    parser.add_argument('--seq_len', type=int, default=96, help='Độ dài lịch sử')
    parser.add_argument('--label_len', type=int, default=48, help='Độ dài label (dùng cho decoder)')
    parser.add_argument('--pred_len', type=int, default=96, help='Độ dài tương lai')
    
    # Model Configs (Dùng chung cho class Model)
    parser.add_argument('--enc_in', type=int, default=7, help='Số lượng biến đầu vào')
    parser.add_argument('--dec_in', type=int, default=7, help='Số lượng biến ở decoder')
    parser.add_argument('--c_out', type=int, default=7, help='Số lượng biến đầu ra')
    parser.add_argument('--d_model', type=int, default=512, help='Chiều không gian nhúng')
    parser.add_argument('--n_heads', type=int, default=8, help='Số lượng Attention Heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Số lượng Encoder Layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Số lượng Decoder Layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Kích thước mạng FeedForward')
    parser.add_argument('--dropout', type=float, default=0.1, help='Tỷ lệ Dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='Loại Time Embedding')
    parser.add_argument('--freq', type=str, default='h', help='Tần suất thời gian (h, m, s, etc.)')
    parser.add_argument('--factor', type=int, default=1, help='Hệ số cho ProbSparse Attention')
    parser.add_argument('--activation', type=str, default='gelu', help='Hàm kích hoạt')
    
    # Các cờ đặc biệt (Flags)
    parser.add_argument('--output_attention', action='store_true', default=False, help='Xuất Attention Map')
    parser.add_argument('--no_skip', action='store_true', default=False, help='Tắt Skip Connection')
    parser.add_argument('--fuse_decoder', action='store_true', default=False, help='Bật Fuse Decoder')
    parser.add_argument('--decoder_type', type=str, default='conv2d', help='Loại Fuse Decoder')
    parser.add_argument('--no_zero_norm', action='store_true', default=False, help='Tắt chuẩn hóa')
    parser.add_argument('--use_norm', type=int, default=1, help='Bật chuẩn hóa (1 hoặc 0)')
    
    # Optimization
    parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Số Epoch')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping')
    
    # Custom test
    parser.add_argument('--model_name', type=str, default='all', help='Tên mô hình cụ thể muốn chạy (để all nếu muốn chạy hết)')
    
    return parser.parse_args()

def get_model_instance(model_name, args):
    """
    Hàm khởi tạo model thông minh dành cho môi trường Kaggle/Linux.
    Tự động quét thư mục models để tìm file khớp với model_name.
    """
    project_root = "/kaggle/working/mHC-manifold-constrained-hyper-connections"
    models_dir = os.path.join(project_root, "models")
    
    if project_root not in sys.path:
        sys.path.append(project_root)

    try:
        available_files = [
            f[:-3] for f in os.listdir(models_dir) 
            if f.endswith(".py") and f != "__init__.py"
        ]
    except FileNotFoundError:
        raise ValueError(f"❌ Không tìm thấy thư mục models tại: {models_dir}")

    target_module_name = None
    for f in available_files:
        if f.lower() == model_name.lower():
            target_module_name = f
            break
            
    if not target_module_name:
        raise ValueError(f"❌ Không tìm thấy file '{model_name}.py' trong thư mục {models_dir}. Các file hiện có: {available_files}")

    try:
        module = importlib.import_module(f'models.{target_module_name}')
        
        if 'mhc' in target_module_name.lower():
            print(f"--- [Auto-Config] Phát hiện kiến trúc mHC: Đã ép cờ no_skip = True ---")
            args.no_skip = True
        else:
            args.no_skip = getattr(args, 'no_skip', False)

        model = module.Model(configs=args)
        print(f"✅ Khởi tạo thành công model: {target_module_name}")
        return model

    except AttributeError:
        raise ValueError(f"❌ File 'models/{target_module_name}.py' không có 'class Model'. Vui lòng kiểm tra lại cấu trúc file.")
    except Exception as e:
        raise ValueError(f"❌ Lỗi phát sinh khi nạp model {target_module_name}: {str(e)}")

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên: {device}")

    # 1. Nạp DataLoader dùng chung
    print(f"Nạp dữ liệu từ: {args.data_path}")
    train_loader, val_loader, test_loader, scaler = get_data_loaders(
        file_path=args.data_path, seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size
    )

    # 2. Danh sách mô hình
    default_models = ['vanilla_transformer', 'mHC_vanilla_transformer', 'iTransformer', 'mHC_iTransformer', 'mHC_patchtst', 'patchtst'] 
    models_to_run = [args.model_name] if args.model_name != 'all' else default_models
    
    # Dictionary lưu trữ
    results = {'mse': {}, 'mae': {}, 'self_mi': {}, 'cross_mi': {}, 'max_cross_mi': {}, 'trackers': {}, 'params': {}}
    save_dir = './eval_results/'
    weights_dir = os.path.join(save_dir, 'saved_weights') # Thư mục lưu riêng weights
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # 3. Vòng lặp tự động Huấn Luyện & Đánh Giá
    for model_name in models_to_run:
        print("\n" + "="*60)
        print(f"🚀 XỬ LÝ MÔ HÌNH: {model_name.upper()}")
        print("="*60)
        
        try:
            model = get_model_instance(model_name, args)
            tracker = StabilityTracker()
            
            # ĐẾM SỐ LƯỢNG THAM SỐ (PARAMETERS)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            results['params'][model_name] = total_params
            print(f"📦 Kích thước mô hình (Trainable Parameters): {total_params:,}")
            
            exp = Exp_Main(
                model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                scaler=scaler, device=device, learning_rate=args.learning_rate, tracker=tracker, args=args
            )

            # Huấn luyện
            print(f"--- Bắt đầu Huấn luyện ({model_name}) ---")
            exp.train(epochs=args.epochs, patience=args.patience)
            
            # LƯU FILE TRỌNG SỐ (WEIGHTS) ĐỘC LẬP SAU KHI TRAIN
            weight_file_path = os.path.join(weights_dir, f"{model_name}_best_weights.pth")
            torch.save(exp.model.state_dict(), weight_file_path)
            print(f"💾 Đã lưu trọng số hội tụ tốt nhất tại: {weight_file_path}")

            # Kiểm thử (Forecasting Performance)
            print(f"--- Bắt đầu Kiểm thử ({model_name}) ---")
            mse, mae = exp.test()
            results['mse'][model_name] = mse
            results['mae'][model_name] = mae

            # Đánh giá Thông tin Tương hỗ
            print(f"--- Đánh giá Information Flow ({model_name}) ---")
            self_mi, cross_mi, max_cross_mi = calculate_perturbation_mi(exp, test_loader, args)
            results['self_mi'][model_name] = self_mi
            results['cross_mi'][model_name] = cross_mi
            results['max_cross_mi'][model_name] = max_cross_mi
            
            results['trackers'][model_name] = tracker

            # Lưu log thô có thêm số Parameter
            with open(os.path.join(save_dir, 'summary_log.txt'), 'a') as f:
                f.write(f"Model: {model_name} | Params: {total_params:,} | MSE: {mse:.4f} | MAE: {mae:.4f} | Self-MI: {self_mi:.4f} | Avg-Cross-MI: {cross_mi:.4f}\n")
                
            del model, exp
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ LỖI trong quá trình chạy mô hình {model_name}: {e}")
            continue

    # 4. TỔNG HỢP VÀ VẼ BIỂU ĐỒ SO SÁNH
    print("\n📊 ĐANG VẼ BIỂU ĐỒ TỔNG HỢP...")
    
    # Chỉ lấy các mô hình đã chạy thành công cả MI để vẽ biểu đồ
    valid_models = [m for m in results['mse'].keys() if m in results['cross_mi']]
    if not valid_models:
        print("Không có mô hình nào chạy thành công để vẽ biểu đồ.")
        return

    x = np.arange(len(valid_models))
    width = 0.35
    
    # Biểu đồ 1: MSE & MAE
    mse_vals = [results['mse'][m] for m in valid_models]
    mae_vals = [results['mae'][m] for m in valid_models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, mse_vals, width, label='MSE', color='salmon')
    ax.bar(x + width/2, mae_vals, width, label='MAE', color='skyblue')
    ax.set_ylabel('Sai số (Thấp hơn là tốt hơn)')
    ax.set_title('So sánh Hiệu suất Dự báo')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_models, rotation=15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'compare_forecasting.png'))
    plt.close()

    # Biểu đồ 2: Information Flow (Mutual Information)
    cross_mi_vals = [results['cross_mi'][m] for m in valid_models]
    max_cross_mi_vals = [results['max_cross_mi'][m] for m in valid_models]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, cross_mi_vals, width, label='Avg Cross-MI', color='lightgreen')
    ax.bar(x + width/2, max_cross_mi_vals, width, label='Max Cross-MI', color='forestgreen')
    ax.set_ylabel('Điểm thông tin (Cao hơn là trao đổi chéo tốt hơn)')
    ax.set_title('Đánh giá Giao tiếp Liên biến (Inter-variate Attention)')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_models, rotation=15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'compare_mutual_information.png'))
    plt.close()

    # Biểu đồ 3: So sánh độ ổn định
    base_model = 'iTransformer'
    mhc_model = 'mHC_iTransformer'
    if base_model in valid_models and mhc_model in valid_models:
        plot_training_stability(
            tracker_baseline=results['trackers'][base_model],
            tracker_mHC=results['trackers'][mhc_model],
            save_dir=save_dir
        )

    print(f"\n✅ HOÀN TẤT! Kết quả và biểu đồ được lưu tại {save_dir}")
    print(f"✅ Trọng số mô hình (.pth) được lưu tại {weights_dir}")

if __name__ == '__main__':
    main()