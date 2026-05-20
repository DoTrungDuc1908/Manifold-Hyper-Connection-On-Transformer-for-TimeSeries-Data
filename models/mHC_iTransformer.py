import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder_mHC, EncoderLayer_mHC
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = getattr(configs, "output_attention", False)

        self.use_norm = getattr(
            configs,
            "use_norm",
            not getattr(configs, "no_zero_norm", False)
        )

        self.c_out = getattr(configs, "c_out", getattr(configs, "enc_in", None))

        self.num_streams = getattr(configs, "num_streams", 4)
        self.sinkhorn_iters = getattr(configs, "sinkhorn_iters", 10)
        self.resid_dropout = getattr(configs, "resid_dropout", configs.dropout)
        self.ffn_dropout = getattr(configs, "ffn_dropout", configs.dropout)

        try:
            self.enc_embedding = DataEmbedding_inverted(
                configs.seq_len,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout
            )
        except TypeError:
            try:
                self.enc_embedding = DataEmbedding_inverted(
                    configs.seq_len,
                    configs.d_model,
                    dropout=configs.dropout
                )
            except TypeError:
                self.enc_embedding = DataEmbedding_inverted(
                    configs.seq_len,
                    configs.d_model
                )

        self.encoder = Encoder_mHC(
            [
                EncoderLayer_mHC(
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
                    activation=configs.activation,
                    sinkhorn_iters=self.sinkhorn_iters
                )
                for _ in range(configs.e_layers)
            ],
            d_model=configs.d_model,
            norm_layer=nn.LayerNorm(configs.d_model),
            num_streams=self.num_streams
        )

        self.projector = nn.Linear(
            configs.d_model,
            configs.pred_len,
            bias=True
        )

    def _normalize(self, x_enc):
        if not self.use_norm:
            return x_enc, None, None

        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means

        stdev = torch.sqrt(
            torch.var(
                x_enc,
                dim=1,
                keepdim=True,
                unbiased=False
            ) + 1e-5
        )

        x_enc = x_enc / stdev

        return x_enc, means, stdev

    def _denormalize(self, dec_out, means, stdev, target_dim):
        if not self.use_norm:
            return dec_out

        stdev = stdev[:, 0, :target_dim].unsqueeze(1)
        means = means[:, 0, :target_dim].unsqueeze(1)

        dec_out = dec_out * stdev
        dec_out = dec_out + means

        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, means, stdev = self._normalize(x_enc)

        _, _, N = x_enc.shape

        target_dim = self.c_out if self.c_out is not None else N
        target_dim = min(target_dim, N)

        # [B, L, N] -> [B, N(+covariates), D]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Correct mHC encoder internally uses:
        # [B, token, D] -> [B, token, S, D] -> [B, token, D]
        enc_out, attns = self.encoder(
            enc_out,
            attn_mask=None
        )

        # [B, N(+covariates), D] -> [B, pred_len, N(+covariates)]
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        dec_out = dec_out[:, :, :target_dim]

        dec_out = self._denormalize(
            dec_out,
            means,
            stdev,
            target_dim
        )

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out, attns = self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec
            )

            dec_out = dec_out[:, -self.pred_len:, :]

            if self.output_attention:
                return dec_out, attns

            return dec_out

        return None
