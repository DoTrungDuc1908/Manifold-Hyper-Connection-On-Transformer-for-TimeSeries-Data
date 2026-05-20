import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import my_Layernorm, series_decomp


class HCResidualMix(nn.Module):
    def __init__(self, num_streams):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_streams, num_streams))

    def forward(self, X):
        H = F.softmax(self.logits, dim=-1)
        return torch.einsum("ij,btjd->btid", H, X)


class AutoformerStreamMixin:
    def init_streams(self, x):
        X = x.unsqueeze(2).repeat(1, 1, self.num_streams, 1)
        X = X + self.stream_embeds.unsqueeze(0).unsqueeze(0)
        return X

    def collapse_streams(self, X):
        weights = F.softmax(self.stream_collapse, dim=0)
        return torch.einsum("s,btsd->btd", weights, X)


class EncoderLayer_HC(nn.Module):
    """
    Autoformer encoder layer with HC stream topology.
    """

    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        moving_avg=25,
        num_streams=4,
        dropout=0.1,
        resid_dropout=None,
        ffn_dropout=None,
        activation="relu",
    ):
        super().__init__()

        d_ff = d_ff or 4 * d_model
        resid_dropout = dropout if resid_dropout is None else resid_dropout
        ffn_dropout = dropout if ffn_dropout is None else ffn_dropout

        self.attention = attention
        self.num_streams = num_streams

        self.pre_map = nn.Linear(d_model, num_streams)
        self.post_map = nn.Linear(d_model, num_streams)
        self.residual_mix = HCResidualMix(num_streams)

        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)

        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)

        self.resid_dropout = nn.Dropout(resid_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def aggregate_streams(self, X):
        stream_context = X.mean(dim=2)
        weights = F.softmax(self.pre_map(stream_context), dim=-1)
        return torch.einsum("bts,btsd->btd", weights, X)

    def distribute_streams(self, y):
        weights = F.softmax(self.post_map(y), dim=-1)
        return torch.einsum("bts,btd->btsd", weights, y)

    def forward(self, X, attn_mask=None):
        z = self.aggregate_streams(X)

        new_z, attn = self.attention(z, z, z, attn_mask=attn_mask)

        z = z + self.resid_dropout(new_z)
        z, _ = self.decomp1(z)

        y = self.ffn_dropout(
            self.activation(self.conv1(z.transpose(-1, 1)))
        )
        y = self.ffn_dropout(self.conv2(y).transpose(-1, 1))

        seasonal, _ = self.decomp2(z + y)

        Y = self.distribute_streams(seasonal)
        X_res = self.residual_mix(X)

        return X_res + Y, attn


class Encoder_HC(nn.Module, AutoformerStreamMixin):
    def __init__(
        self,
        attn_layers,
        d_model,
        conv_layers=None,
        norm_layer=None,
        num_streams=4,
    ):
        super().__init__()

        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.num_streams = num_streams

        self.stream_embeds = nn.Parameter(torch.randn(num_streams, d_model) * 0.02)
        self.stream_collapse = nn.Parameter(torch.zeros(num_streams))

    def forward(self, x, attn_mask=None):
        X = self.init_streams(x)
        attns = []

        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                X, attn = attn_layer(X, attn_mask=attn_mask)

                x_collapsed = self.collapse_streams(X)
                x_collapsed = conv_layer(x_collapsed)
                X = self.init_streams(x_collapsed)

                attns.append(attn)

            X, attn = self.attn_layers[-1](X, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                X, attn = attn_layer(X, attn_mask=attn_mask)
                attns.append(attn)

        x = self.collapse_streams(X)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer_HC(nn.Module):
    """
    Autoformer decoder layer with HC stream topology.
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        c_out,
        d_ff=None,
        moving_avg=25,
        num_streams=4,
        dropout=0.1,
        resid_dropout=None,
        ffn_dropout=None,
        activation="relu",
    ):
        super().__init__()

        d_ff = d_ff or 4 * d_model
        resid_dropout = dropout if resid_dropout is None else resid_dropout
        ffn_dropout = dropout if ffn_dropout is None else ffn_dropout

        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.pre_map = nn.Linear(d_model, num_streams)
        self.post_map = nn.Linear(d_model, num_streams)
        self.residual_mix = HCResidualMix(num_streams)

        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)

        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)

        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )

        self.resid_dropout = nn.Dropout(resid_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def aggregate_streams(self, X):
        stream_context = X.mean(dim=2)
        weights = F.softmax(self.pre_map(stream_context), dim=-1)
        return torch.einsum("bts,btsd->btd", weights, X)

    def distribute_streams(self, y):
        weights = F.softmax(self.post_map(y), dim=-1)
        return torch.einsum("bts,btd->btsd", weights, y)

    def forward(self, X, cross, x_mask=None, cross_mask=None):
        z = self.aggregate_streams(X)

        z = z + self.resid_dropout(
            self.self_attention(z, z, z, attn_mask=x_mask)[0]
        )
        z, trend1 = self.decomp1(z)

        z = z + self.resid_dropout(
            self.cross_attention(z, cross, cross, attn_mask=cross_mask)[0]
        )
        z, trend2 = self.decomp2(z)

        y = self.ffn_dropout(
            self.activation(self.conv1(z.transpose(-1, 1)))
        )
        y = self.ffn_dropout(self.conv2(y).transpose(-1, 1))

        seasonal, trend3 = self.decomp3(z + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(
            residual_trend.permute(0, 2, 1)
        ).transpose(1, 2)

        Y = self.distribute_streams(seasonal)
        X_res = self.residual_mix(X)

        return X_res + Y, residual_trend


class Decoder_HC(nn.Module, AutoformerStreamMixin):
    def __init__(
        self,
        layers,
        d_model,
        norm_layer=None,
        projection=None,
        num_streams=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.num_streams = num_streams

        self.stream_embeds = nn.Parameter(torch.randn(num_streams, d_model) * 0.02)
        self.stream_collapse = nn.Parameter(torch.zeros(num_streams))

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        X = self.init_streams(x)

        for layer in self.layers:
            X, residual_trend = layer(
                X,
                cross,
                x_mask=x_mask,
                cross_mask=cross_mask,
            )
            trend = trend + residual_trend

        x = self.collapse_streams(X)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, trend
