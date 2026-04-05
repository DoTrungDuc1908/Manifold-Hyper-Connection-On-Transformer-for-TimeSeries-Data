import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_stability(tracker_baseline, tracker_mHC, save_dir):
    """Vẽ 3 biểu đồ: Loss Curve, Grad Norm, và Amax Gain"""
    os.makedirs(save_dir, exist_ok=True)
    steps = np.arange(len(tracker_baseline.loss_history))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss Curve
    axs[0].plot(steps, tracker_baseline.loss_history, label='Baseline (Residual)', alpha=0.7)
    axs[0].plot(steps, tracker_mHC.loss_history, label='mHC', alpha=0.7)
    axs[0].set_title('Training Loss Curve')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    # 2. Gradient Norm
    axs[1].plot(steps, tracker_baseline.grad_norm_history, label='Baseline (Residual)', alpha=0.7)
    axs[1].plot(steps, tracker_mHC.grad_norm_history, label='mHC', alpha=0.7)
    axs[1].set_title('Gradient Norm (L2)')
    axs[1].set_xlabel('Steps')
    axs[1].legend()
    
    # 3. Amax Gain (Chỉ mHC mới có ý nghĩa kiểm soát gain qua ma trận)
    axs[2].plot(steps, tracker_mHC.amax_gain_history, label='mHC Amax Gain', color='green')
    axs[2].axhline(y=1.0, color='r', linestyle='--', label='Ideal Mapping (=1)')
    axs[2].set_title('mHC Forward/Backward Amax Gain')
    axs[2].set_xlabel('Steps')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_stability.png'))
    plt.close()

def plot_representation_collapse(sim_baseline, sim_mHC, save_dir):
    """Vẽ biểu đồ Cosine Similarity qua các layer"""
    layers = np.arange(1, len(sim_baseline) + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(layers, sim_baseline, marker='o', label='Baseline (Residual)')
    plt.plot(layers, sim_mHC, marker='s', label='mHC')
    plt.title('Representation Collapse (Cosine Similarity to Input)')
    plt.xlabel('Layer Depth')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, 'representation_collapse.png'))
    plt.close()