import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Exp_Main:
    """
    Unified experiment runner for Autoformer / iTransformer / PatchTST /
    vanilla Transformer and HC/mHC variants.

    Key assumptions:
        - Dataloader returns:
            batch_x:      [B, seq_len, C]
            batch_y:      [B, label_len + pred_len, C]
            batch_x_mark: [B, seq_len, time_dim]
            batch_y_mark: [B, label_len + pred_len, time_dim]
        - Loss is computed only on the final pred_len steps.
        - Best checkpoint is selected by validation loss.
    """

    def __init__(
        self,
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        device,
        learning_rate=1e-4,
        tracker=None,
    ):
        self.args = args
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.device = device
        self.learning_rate = learning_rate
        self.tracker = tracker

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self.grad_norm_max_history: List[float] = []
        self.activation_norm_history: List[float] = []
        self.epoch_time_history: List[float] = []

        self.best_epoch: Optional[int] = None
        self.best_val_loss: float = float("inf")
        self.residual_diagnostics: Dict[str, Dict[str, float]] = {}
        self.test_metrics: Dict[str, float] = {}

        self._activation_cache: List[float] = []
        if getattr(self.args, "log_activation_norm", False):
            self._register_activation_hooks()

    def _register_activation_hooks(self):
        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if torch.is_tensor(output):
                self._activation_cache.append(float(output.detach().norm(p=2).item()))

        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if cls_name.endswith("layer") and ("encoder" in name.lower() or "decoder" in name.lower()):
                module.register_forward_hook(hook_fn)

    def _compute_grad_norm(self):
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        return total_norm_sq ** 0.5

    def _feature_slice(self):
        return -1 if getattr(self.args, "features", "M") == "MS" else 0

    def _select_pred_target(self, outputs, batch_y):
        f_dim = self._feature_slice()
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        true = batch_y[:, -self.args.pred_len:, f_dim:]
        return outputs, true

    def prepare_batch(self, batch):
        """
        Prepare official-style batch.

        Preferred input:
            batch_x, batch_y, batch_x_mark, batch_y_mark

        Fallback for old dataloaders:
            batch_x, batch_y
            marks are created as zeros.
        """
        if len(batch) == 4:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        elif len(batch) == 2:
            batch_x, batch_y = batch
            B = batch_x.shape[0]
            x_mark_dim = 4 if getattr(self.args, "freq", "h") == "h" else 3
            batch_x_mark = torch.zeros(B, batch_x.shape[1], x_mark_dim)
            batch_y_mark = torch.zeros(B, self.args.label_len + self.args.pred_len, x_mark_dim)
        else:
            raise ValueError(f"Unsupported batch format with {len(batch)} elements.")

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # Official decoder input:
        # known label_len part + zero pred_len part.
        if batch_y.shape[1] >= self.args.label_len + self.args.pred_len:
            dec_known = batch_y[:, :self.args.label_len, :]
        else:
            # Old dataloader fallback: batch_y contains only pred_len.
            dec_known = batch_x[:, -self.args.label_len:, :]

        dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
        dec_inp = torch.cat([dec_known, dec_zeros], dim=1).float().to(self.device)

        # Ensure y_mark length matches decoder input length.
        dec_len = self.args.label_len + self.args.pred_len
        if batch_y_mark.shape[1] != dec_len:
            B = batch_x.shape[0]
            mark_dim = batch_x_mark.shape[-1]
            fixed_mark = torch.zeros(B, dec_len, mark_dim, device=self.device)
            copy_len = min(dec_len, batch_y_mark.shape[1])
            fixed_mark[:, :copy_len, :] = batch_y_mark[:, :copy_len, :]
            batch_y_mark = fixed_mark

        return batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark

    def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs

    def _run_one_epoch(self):
        self.model.train()
        losses = []
        grad_norms = []

        for batch_data in self.train_loader:
            batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark = self.prepare_batch(batch_data)

            self.optimizer.zero_grad()
            outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, true = self._select_pred_target(outputs, batch_y)

            loss = self.criterion(outputs, true)
            loss.backward()

            if getattr(self.args, "grad_clip", 0.0) and self.args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
                grad_norm = float(grad_norm)
            else:
                grad_norm = float(self._compute_grad_norm())

            if self.tracker is not None:
                self.tracker.log_gradient_norm(self.model)
                self.tracker.log_amax_gain(self.model)
                self.tracker.log_loss(float(loss.item()))

            self.optimizer.step()
            losses.append(float(loss.item()))
            grad_norms.append(grad_norm)

        return float(np.mean(losses)), grad_norms

    def train(
        self,
        epochs=10,
        patience=3,
        weight_save_path="best_model_checkpoint.pth",
        min_epochs=0,
        disable_early_stopping=False,
    ):
        self.train_loss_history = []
        self.val_loss_history = []
        self.grad_norm_history = []
        self.grad_norm_max_history = []
        self.activation_norm_history = []
        self.epoch_time_history = []

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            self._activation_cache = []

            avg_train_loss, grad_norms = self._run_one_epoch()
            val_loss = self.validate()
            epoch_time = time.time() - epoch_start

            avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
            max_grad_norm = float(np.max(grad_norms)) if grad_norms else 0.0
            avg_activation_norm = float(np.mean(self._activation_cache)) if self._activation_cache else 0.0

            self.train_loss_history.append(float(avg_train_loss))
            self.val_loss_history.append(float(val_loss))
            self.grad_norm_history.append(avg_grad_norm)
            self.grad_norm_max_history.append(max_grad_norm)
            self.activation_norm_history.append(avg_activation_norm)
            self.epoch_time_history.append(float(epoch_time))

            print(
                f"Epoch {epoch + 1}/{epochs} | Time {epoch_time:.2f}s | "
                f"Train {avg_train_loss:.6f} | Val {val_loss:.6f} | "
                f"GradMean {avg_grad_norm:.4f} | GradMax {max_grad_norm:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_epoch = epoch + 1
                self.best_val_loss = float(best_val_loss)
                if weight_save_path is not None:
                    torch.save(self.model.state_dict(), weight_save_path)
            else:
                patience_counter += 1

            if not disable_early_stopping and epoch + 1 >= min_epochs and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch: {self.best_epoch}")
                break

        if weight_save_path is not None:
            try:
                state_dict = torch.load(weight_save_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded best checkpoint from {weight_save_path}")
            except FileNotFoundError:
                print(f"Warning: best checkpoint not found at {weight_save_path}")

        self.residual_diagnostics = self.collect_residual_diagnostics()
        return self.train_loss_history, self.val_loss_history

    def validate(self):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch_data in self.val_loader:
                batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark = self.prepare_batch(batch_data)
                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, true = self._select_pred_target(outputs, batch_y)

                loss = self.criterion(outputs, true)
                losses.append(float(loss.item()))

        return float(np.mean(losses)) if losses else float("inf")

    def _inverse_transform_array(self, arr):
        """
        Inverse transform for M/S/MS settings.

        If arr last dimension matches scaler.n_features_in_, use scaler directly.
        If arr has one target dimension under MS, use the last scaler column.
        """
        if self.scaler is None:
            return arr

        shape = arr.shape
        last_dim = shape[-1]
        flat = arr.reshape(-1, last_dim)

        n_features = getattr(self.scaler, "n_features_in_", None)
        if n_features is None:
            return arr

        if last_dim == n_features:
            return self.scaler.inverse_transform(flat).reshape(shape)

        if last_dim == 1 and hasattr(self.scaler, "mean_") and hasattr(self.scaler, "scale_"):
            target_idx = -1 if getattr(self.args, "features", "M") == "MS" else 0
            mean = self.scaler.mean_[target_idx]
            scale = self.scaler.scale_[target_idx]
            return (flat * scale + mean).reshape(shape)

        return arr

    def test(self, return_real_metrics=False):
        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for batch_data in self.test_loader:
                batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark = self.prepare_batch(batch_data)
                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, true = self._select_pred_target(outputs, batch_y)

                preds.append(outputs.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mse_norm = float(np.mean((preds - trues) ** 2))
        mae_norm = float(np.mean(np.abs(preds - trues)))

        preds_real = self._inverse_transform_array(preds)
        trues_real = self._inverse_transform_array(trues)
        mse_real = float(np.mean((preds_real - trues_real) ** 2))
        mae_real = float(np.mean(np.abs(preds_real - trues_real)))

        self.test_metrics = {
            "mse_normalized": mse_norm,
            "mae_normalized": mae_norm,
            "mse_real": mse_real,
            "mae_real": mae_real,
        }

        print(f"Test normalized | MSE: {mse_norm:.6f} | MAE: {mae_norm:.6f}")
        print(f"Test real scale | MSE: {mse_real:.6f} | MAE: {mae_real:.6f}")

        if return_real_metrics:
            return mse_real, mae_real

        return mse_norm, mae_norm

    def collect_residual_diagnostics(self):
        diagnostics = {}

        for name, module in self.model.named_modules():
            if not hasattr(module, "residual_mix"):
                continue

            mix = module.residual_mix
            if not hasattr(mix, "logits"):
                continue

            logits = mix.logits.detach()
            if hasattr(mix, "sinkhorn"):
                H = mix.sinkhorn(logits).detach()
            else:
                H = torch.softmax(logits, dim=-1).detach()

            row_sum = H.sum(dim=-1)
            col_sum = H.sum(dim=-2)

            try:
                spectral_norm = torch.linalg.matrix_norm(H, ord=2).item()
            except Exception:
                spectral_norm = float("nan")

            try:
                condition_number = torch.linalg.cond(H).item()
            except Exception:
                condition_number = float("nan")

            entropy = -torch.sum(H * torch.log(H + 1e-8)).item()

            diagnostics[name] = {
                "row_sum_mean": float(row_sum.mean().item()),
                "row_sum_std": float(row_sum.std().item()) if row_sum.numel() > 1 else 0.0,
                "col_sum_mean": float(col_sum.mean().item()),
                "col_sum_std": float(col_sum.std().item()) if col_sum.numel() > 1 else 0.0,
                "spectral_norm": float(spectral_norm),
                "condition_number": float(condition_number),
                "entropy": float(entropy),
            }

        return diagnostics
