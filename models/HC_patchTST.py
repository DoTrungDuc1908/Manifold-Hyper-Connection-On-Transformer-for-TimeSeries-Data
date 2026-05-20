import torch
from torch import nn

from layers.Transformer_EncDec import Encoder_HC, EncoderLayer_HC
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from layers.RevIN import RevIN


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


class FlattenHead(nn.Module):
    """
    PatchTST head.

    Input:
        x: [B, n_vars, d_model, patch_num]

    Output:
        x: [B, n_vars, target_window]
    """

    def __init__(
        self,
        n_vars,
        nf,
        target_window,
        head_dropout=0.0,
        fuse_decoder=False,
        decoder_type="conv2d"
    ):
        super().__init__()

        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

        self.fuse_decoder = fuse_decoder
        self.decoder_type = decoder_type

        if self.fuse_decoder:
            if self.decoder_type == "conv2d":
                kernel_width = 8
                self.fuse_proj = nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=(4 + self.n_vars, kernel_width),
                    padding="same"
                )
            elif self.decoder_type == "MLP":
                self.flat_proj = nn.Sequential(
                    nn.Linear(n_vars * nf, n_vars * nf),
                    nn.ReLU()
                )
            else:
                raise ValueError(
                    f"Unsupported decoder_type={decoder_type}. "
                    "Use 'conv2d' or 'MLP'."
                )

    def forward(self, x):
        x = self.flatten(x)

        if self.fuse_decoder:
            if self.decoder_type == "conv2d":
                x = x.unsqueeze(1)
                x = self.fuse_proj(x)
                x = x.squeeze(1)
            elif self.decoder_type == "MLP":
                bsz, n_vars, nf = x.shape
                x = x.contiguous().view(bsz, n_vars * nf)
                x = self.flat_proj(x)
                x = x.view(bsz, n_vars, nf)

        x = self.linear(x)
        x = self.dropout(x)

        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()

        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = getattr(configs, "output_attention", False)

        self.no_skip = getattr(configs, "no_skip", False)
        self.fuse_decoder = getattr(configs, "fuse_decoder", False)
        self.decoder_type = getattr(configs, "decoder_type", "conv2d")
        self.no_zero_norm = getattr(configs, "no_zero_norm", False)

        self.num_streams = getattr(configs, "num_streams", 4)
        self.resid_dropout = getattr(configs, "resid_dropout", configs.dropout)
        self.ffn_dropout = getattr(configs, "ffn_dropout", configs.dropout)

        self.patch_len = patch_len
        self.stride = stride
        self.padding = stride

        if not self.no_zero_norm:
            self.revin = RevIN(
                num_features=configs.enc_in,
                affine=True
            )
        else:
            self.revin = None

        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            patch_len,
            stride,
            self.padding,
            configs.dropout
        )

        self.encoder = Encoder_HC(
            [
                EncoderLayer_HC(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    num_streams=self.num_streams,
                    dropout=configs.dropout,
                    resid_dropout=self.resid_dropout,
                    ffn_dropout=self.ffn_dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            d_model=configs.d_model,
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2)
            ),
            num_streams=self.num_streams
        )

        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf = configs.d_model * self.patch_num

        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            self.head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                configs.pred_len,
                head_dropout=configs.dropout,
                fuse_decoder=self.fuse_decoder,
                decoder_type=self.decoder_type
            )

        elif self.task_name in ["imputation", "anomaly_detection"]:
            self.head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                configs.seq_len,
                head_dropout=configs.dropout,
                fuse_decoder=self.fuse_decoder,
                decoder_type=self.decoder_type
            )

        elif self.task_name == "classification":
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in,
                configs.num_class
            )

    def _encode(self, x_enc):
        """
        x_enc:
            [B, L, M]

        return:
            enc_out: [B, M, d_model, patch_num]
            attns
        """

        B, _, _ = x_enc.shape

        # Channel independence: [B, L, M] -> [B, M, L]
        x_enc = x_enc.permute(0, 2, 1)

        # [B, M, L] -> [B*M, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # HC encoder internally:
        # [B*M, patch_num, D]
        # -> [B*M, patch_num, S, D]
        # -> [B*M, patch_num, D]
        enc_out, attns = self.encoder(
            enc_out,
            attn_mask=None
        )

        # [B*M, patch_num, D] -> [B, M, patch_num, D]
        enc_out = enc_out.reshape(
            B,
            n_vars,
            enc_out.shape[-2],
            enc_out.shape[-1]
        )

        # [B, M, patch_num, D] -> [B, M, D, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2).contiguous()

        return enc_out, attns

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.revin is not None:
            x_enc = self.revin(x_enc, mode="norm")

        enc_out, attns = self._encode(x_enc)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        if self.revin is not None:
            dec_out = self.revin(dec_out, mode="denorm")

        if self.output_attention:
            return dec_out, attns

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        if self.revin is not None:
            x_enc = self.revin(x_enc, mode="norm")
            if mask is not None:
                x_enc = x_enc.masked_fill(mask == 0, 0)

        enc_out, _ = self._encode(x_enc)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        if self.revin is not None:
            dec_out = self.revin(dec_out, mode="denorm")

        return dec_out

    def anomaly_detection(self, x_enc):
        if self.revin is not None:
            x_enc = self.revin(x_enc, mode="norm")

        enc_out, _ = self._encode(x_enc)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        if self.revin is not None:
            dec_out = self.revin(dec_out, mode="denorm")

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        if self.revin is not None:
            x_enc = self.revin(x_enc, mode="norm")

        enc_out, _ = self._encode(x_enc)

        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec
            )

            if self.output_attention:
                return dec_out[0][:, -self.pred_len:, :], dec_out[1]

            return dec_out[:, -self.pred_len:, :]

        if self.task_name == "imputation":
            return self.imputation(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                mask
            )

        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)

        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)

        return None
