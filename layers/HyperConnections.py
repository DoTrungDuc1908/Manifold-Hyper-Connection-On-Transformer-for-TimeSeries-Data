import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperConnection(nn.Module):
    def __init__(self, d_model, num_streams, layer_fn, dynamic=True):
        super().__init__()
        self.d_model = d_model
        self.num_streams = num_streams
        self.dynamic = dynamic
        self.layer_fn = layer_fn
        self.pre_proj = nn.Linear(d_model, 1, bias=False)
        self.post_proj = nn.Linear(d_model, num_streams, bias=False)
        self.res_logits = nn.Parameter(torch.zeros(num_streams, num_streams))
        if dynamic:
            self.dynamic_router = nn.Linear(d_model, num_streams * num_streams)
        self.stream_embeds = nn.Parameter(torch.randn(num_streams, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def init_streams(self, x):
        B, T, D = x.shape
        streams = x.unsqueeze(2) + self.stream_embeds.unsqueeze(0).unsqueeze(0)
        return streams

    def compute_pre_map(self, X):
        scores = self.pre_proj(X).squeeze(-1)
        scores = torch.sigmoid(scores)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.einsum('btn,btnd->btd', scores, X)
        return out

    def compute_post_map(self, y):
        scores = self.post_proj(y)
        scores = torch.sigmoid(scores)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.einsum('btn,btd->btnd', scores, y)
        return out

    def compute_residual_map(self, X):
        H = self.res_logits
        if self.dynamic:
            dynamic_delta = self.dynamic_router(X.mean(dim=2))
            B, T, _ = dynamic_delta.shape
            dynamic_delta = dynamic_delta.view(B, T, self.num_streams, self.num_streams)
            H = H.unsqueeze(0).unsqueeze(0) + dynamic_delta
            H = F.softmax(H, dim=-1)
        else:
            H = F.softmax(H, dim=-1)
        return H

    def apply_residual_mix(self, X, H):
        if H.dim() == 2:
            out = torch.einsum('nm,btmd->btnd', H, X)
        else:
            out = torch.einsum('btnm,btmd->btnd', H, X)
        return out

    def forward(self, x):
        X = self.init_streams(x)
        pre_x = self.compute_pre_map(X)
        y = self.layer_fn(pre_x)
        Y = self.compute_post_map(y)
        H_res = self.compute_residual_map(X)
        X_res = self.apply_residual_mix(X, H_res)
        out = X_res + Y
        out = self.norm(out)
        return out


class SinkhornProjection(nn.Module):
    def __init__(self, num_iters=10, eps=1e-6):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps

    def forward(self, logits):
        P = torch.exp(logits)
        for _ in range(self.num_iters):
            P = P / (P.sum(dim=-1, keepdim=True) + self.eps)
            P = P / (P.sum(dim=-2, keepdim=True) + self.eps)
        return P


class mHC(nn.Module):
    def __init__(self, d_model, num_streams=4, sinkhorn_iters=10):
        super().__init__()
        self.d_model = d_model
        self.num_streams = num_streams
        self.pre_proj = nn.Linear(d_model, num_streams, bias=False)
        self.post_proj = nn.Linear(d_model, num_streams, bias=False)
        self.res_logits = nn.Parameter(torch.zeros(num_streams, num_streams))
        self.sinkhorn = SinkhornProjection(num_iters=sinkhorn_iters)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, layer_fn):
        B, T, S, D = x.shape
        x = self.norm(x)
        stream_summary = x.mean(dim=2)
        H_pre = torch.sigmoid(self.pre_proj(stream_summary))
        H_pre = H_pre.unsqueeze(-1)
        layer_input = (x * H_pre).sum(dim=2)
        layer_output = layer_fn(layer_input)
        H_post = torch.sigmoid(self.post_proj(layer_output))
        H_post = H_post.unsqueeze(-1)
        expanded_output = H_post * layer_output.unsqueeze(2)
        H_res = self.sinkhorn(self.res_logits)
        residual_stream = torch.einsum('ij, b t j d -> b t i d', H_res, x)
        out = residual_stream + expanded_output
        return out
