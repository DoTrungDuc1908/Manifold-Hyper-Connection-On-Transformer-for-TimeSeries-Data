import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.HyperConnections import HyperConnection, mHC

class ConvLayer(nn.Module):

    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu', no_skip=False):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.no_skip = no_skip

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        if self.no_skip:
            x = self.norm1(self.dropout(new_x))
        else:
            x = self.norm1(x + self.dropout(new_x))
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return (self.norm2(x + y), attn)

class Encoder(nn.Module):

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return (x, attns)

class DecoderLayer(nn.Module):

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)

class Decoder(nn.Module):

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x

class HCResidualMix(nn.Module):

    def __init__(self, num_streams):
        super().__init__()
        self.num_streams = num_streams
        self.logits = nn.Parameter(torch.zeros(num_streams, num_streams))

    def forward(self, X):
        H = F.softmax(self.logits, dim=-1)
        out = torch.einsum('ij,btjd->btid', H, X)
        return out

class SinkhornProjection(nn.Module):

    def __init__(self, num_iters=10, eps=1e-06):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps

    def forward(self, logits):
        P = torch.exp(logits)
        for _ in range(self.num_iters):
            P = P / (P.sum(dim=-1, keepdim=True) + self.eps)
            P = P / (P.sum(dim=-2, keepdim=True) + self.eps)
        return P

class mHCResidualMix(nn.Module):

    def __init__(self, num_streams, sinkhorn_iters=10):
        super().__init__()
        self.num_streams = num_streams
        self.logits = nn.Parameter(torch.zeros(num_streams, num_streams))
        self.sinkhorn = SinkhornProjection(num_iters=sinkhorn_iters)

    def forward(self, X):
        H = self.sinkhorn(self.logits)
        out = torch.einsum('ij,btjd->btid', H, X)
        return out

class StreamMixin:

    def init_streams(self, x):
        X = x.unsqueeze(2).repeat(1, 1, self.num_streams, 1)
        X = X + self.stream_embeds.unsqueeze(0).unsqueeze(0)
        return X

    def collapse_streams(self, X):
        weights = F.softmax(self.stream_collapse, dim=0)
        out = torch.einsum('s,btsd->btd', weights, X)
        return out

class EncoderLayer_HC(nn.Module):

    def __init__(self, attention, d_model, d_ff=None, num_streams=4, dropout=0.1, resid_dropout=None, ffn_dropout=None, activation='gelu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        resid_dropout = dropout if resid_dropout is None else resid_dropout
        ffn_dropout = dropout if ffn_dropout is None else ffn_dropout
        self.num_streams = num_streams
        self.attention = attention
        self.pre_map = nn.Linear(d_model, num_streams)
        self.post_map = nn.Linear(d_model, num_streams)
        self.residual_mix = HCResidualMix(num_streams=num_streams)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def aggregate_streams(self, X):
        stream_context = X.mean(dim=2)
        weights = F.softmax(self.pre_map(stream_context), dim=-1)
        z = torch.einsum('bts,btsd->btd', weights, X)
        return z

    def distribute_streams(self, y):
        weights = F.softmax(self.post_map(y), dim=-1)
        Y = torch.einsum('bts,btd->btsd', weights, y)
        return Y

    def forward(self, X, attn_mask=None, tau=None, delta=None):
        z = self.aggregate_streams(X)
        new_z, attn = self.attention(z, z, z, attn_mask=attn_mask, tau=tau, delta=delta)
        z = self.norm1(z + self.resid_dropout(new_z))
        y = self.ffn_dropout(self.activation(self.conv1(z.transpose(-1, 1))))
        y = self.ffn_dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm2(z + y)
        Y = self.distribute_streams(y)
        X_res = self.residual_mix(X)
        X_out = X_res + Y
        return (X_out, attn)

class EncoderLayer_mHC(nn.Module):

    def __init__(self, attention, d_model, d_ff=None, num_streams=4, dropout=0.1, resid_dropout=None, ffn_dropout=None, activation='gelu', sinkhorn_iters=10):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        resid_dropout = dropout if resid_dropout is None else resid_dropout
        ffn_dropout = dropout if ffn_dropout is None else ffn_dropout
        self.num_streams = num_streams
        self.attention = attention
        self.pre_map = nn.Linear(d_model, num_streams)
        self.post_map = nn.Linear(d_model, num_streams)
        self.residual_mix = mHCResidualMix(num_streams=num_streams, sinkhorn_iters=sinkhorn_iters)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def aggregate_streams(self, X):
        stream_context = X.mean(dim=2)
        weights = F.softmax(self.pre_map(stream_context), dim=-1)
        z = torch.einsum('bts,btsd->btd', weights, X)
        return z

    def distribute_streams(self, y):
        weights = F.softmax(self.post_map(y), dim=-1)
        Y = torch.einsum('bts,btd->btsd', weights, y)
        return Y

    def forward(self, X, attn_mask=None, tau=None, delta=None):
        z = self.aggregate_streams(X)
        new_z, attn = self.attention(z, z, z, attn_mask=attn_mask, tau=tau, delta=delta)
        z = self.norm1(z + self.resid_dropout(new_z))
        y = self.ffn_dropout(self.activation(self.conv1(z.transpose(-1, 1))))
        y = self.ffn_dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm2(z + y)
        Y = self.distribute_streams(y)
        X_res = self.residual_mix(X)
        X_out = X_res + Y
        return (X_out, attn)

class Encoder_HC(nn.Module, StreamMixin):

    def __init__(self, layers, d_model, norm_layer=None, num_streams=4):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.num_streams = num_streams
        self.stream_embeds = nn.Parameter(torch.randn(num_streams, d_model) * 0.02)
        self.stream_collapse = nn.Parameter(torch.zeros(num_streams))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        X = self.init_streams(x)
        attns = []
        for layer in self.layers:
            X, attn = layer(X, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        out = self.collapse_streams(X)
        if self.norm is not None:
            out = self.norm(out)
        return (out, attns)

class Encoder_mHC(nn.Module, StreamMixin):

    def __init__(self, layers, d_model, norm_layer=None, num_streams=4):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.num_streams = num_streams
        self.stream_embeds = nn.Parameter(torch.randn(num_streams, d_model) * 0.02)
        self.stream_collapse = nn.Parameter(torch.zeros(num_streams))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        X = self.init_streams(x)
        attns = []
        for layer in self.layers:
            X, attn = layer(X, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        out = self.collapse_streams(X)
        if self.norm is not None:
            out = self.norm(out)
        return (out, attns)

class DecoderLayer_HC(nn.Module):

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, num_streams=4, dropout=0.1, resid_dropout=None, ffn_dropout=None, activation='gelu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        resid_dropout = dropout if resid_dropout is None else resid_dropout
        ffn_dropout = dropout if ffn_dropout is None else ffn_dropout
        self.num_streams = num_streams
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.pre_map = nn.Linear(d_model, num_streams)
        self.post_map = nn.Linear(d_model, num_streams)
        self.residual_mix = HCResidualMix(num_streams=num_streams)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def aggregate_streams(self, X):
        stream_context = X.mean(dim=2)
        weights = F.softmax(self.pre_map(stream_context), dim=-1)
        z = torch.einsum('bts,btsd->btd', weights, X)
        return z

    def distribute_streams(self, y):
        weights = F.softmax(self.post_map(y), dim=-1)
        Y = torch.einsum('bts,btd->btsd', weights, y)
        return Y

    def forward(self, X, enc_out, x_mask=None, cross_mask=None, tau=None, delta=None):
        z = self.aggregate_streams(X)
        new_z, _ = self.self_attention(z, z, z, attn_mask=x_mask, tau=tau, delta=None)
        z = self.norm1(z + self.resid_dropout(new_z))
        new_z, _ = self.cross_attention(z, enc_out, enc_out, attn_mask=cross_mask, tau=tau, delta=delta)
        z = self.norm2(z + self.resid_dropout(new_z))
        y = self.ffn_dropout(self.activation(self.conv1(z.transpose(-1, 1))))
        y = self.ffn_dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm3(z + y)
        Y = self.distribute_streams(y)
        X_res = self.residual_mix(X)
        X_out = X_res + Y
        return X_out

class DecoderLayer_mHC(nn.Module):

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, num_streams=4, dropout=0.1, resid_dropout=None, ffn_dropout=None, activation='gelu', sinkhorn_iters=10):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        resid_dropout = dropout if resid_dropout is None else resid_dropout
        ffn_dropout = dropout if ffn_dropout is None else ffn_dropout
        self.num_streams = num_streams
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.pre_map = nn.Linear(d_model, num_streams)
        self.post_map = nn.Linear(d_model, num_streams)
        self.residual_mix = mHCResidualMix(num_streams=num_streams, sinkhorn_iters=sinkhorn_iters)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def aggregate_streams(self, X):
        stream_context = X.mean(dim=2)
        weights = F.softmax(self.pre_map(stream_context), dim=-1)
        z = torch.einsum('bts,btsd->btd', weights, X)
        return z

    def distribute_streams(self, y):
        weights = F.softmax(self.post_map(y), dim=-1)
        Y = torch.einsum('bts,btd->btsd', weights, y)
        return Y

    def forward(self, X, enc_out, x_mask=None, cross_mask=None, tau=None, delta=None):
        z = self.aggregate_streams(X)
        new_z, _ = self.self_attention(z, z, z, attn_mask=x_mask, tau=tau, delta=None)
        z = self.norm1(z + self.resid_dropout(new_z))
        new_z, _ = self.cross_attention(z, enc_out, enc_out, attn_mask=cross_mask, tau=tau, delta=delta)
        z = self.norm2(z + self.resid_dropout(new_z))
        y = self.ffn_dropout(self.activation(self.conv1(z.transpose(-1, 1))))
        y = self.ffn_dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm3(z + y)
        Y = self.distribute_streams(y)
        X_res = self.residual_mix(X)
        X_out = X_res + Y
        return X_out

class Decoder_HC(nn.Module, StreamMixin):

    def __init__(self, layers, d_model, norm_layer=None, projection=None, num_streams=4):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.num_streams = num_streams
        self.stream_embeds = nn.Parameter(torch.randn(num_streams, d_model) * 0.02)
        self.stream_collapse = nn.Parameter(torch.zeros(num_streams))

    def forward(self, x, enc_out, x_mask=None, cross_mask=None, tau=None, delta=None):
        X = self.init_streams(x)
        for layer in self.layers:
            X = layer(X, enc_out, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        out = self.collapse_streams(X)
        if self.norm is not None:
            out = self.norm(out)
        if self.projection is not None:
            out = self.projection(out)
        return out

class Decoder_mHC(nn.Module, StreamMixin):

    def __init__(self, layers, d_model, norm_layer=None, projection=None, num_streams=4):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.num_streams = num_streams
        self.stream_embeds = nn.Parameter(torch.randn(num_streams, d_model) * 0.02)
        self.stream_collapse = nn.Parameter(torch.zeros(num_streams))

    def forward(self, x, enc_out, x_mask=None, cross_mask=None, tau=None, delta=None):
        X = self.init_streams(x)
        for layer in self.layers:
            X = layer(X, enc_out, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        out = self.collapse_streams(X)
        if self.norm is not None:
            out = self.norm(out)
        if self.projection is not None:
            out = self.projection(out)
        return out
