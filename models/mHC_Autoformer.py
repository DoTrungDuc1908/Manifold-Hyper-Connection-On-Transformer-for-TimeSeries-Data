import torch
import torch.nn as nn

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import series_decomp, my_Layernorm
from layers.Autoformer_EncDec_mHC import (
    Encoder_mHC,
    EncoderLayer_mHC,
    Decoder_mHC,
    DecoderLayer_mHC,
)

try:
    from layers.Embed import DataEmbedding_wo_pos
except ImportError:
    from layers.Embed import DataEmbedding as DataEmbedding_wo_pos


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = getattr(configs, "output_attention", False)

        self.num_streams = getattr(configs, "num_streams", 4)
        self.sinkhorn_iters = getattr(configs, "sinkhorn_iters", 10)
        self.resid_dropout = getattr(configs, "resid_dropout", configs.dropout)
        self.ffn_dropout = getattr(configs, "ffn_dropout", configs.dropout)

        self.decomp = series_decomp(configs.moving_avg)

        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.encoder = Encoder_mHC(
            [
                EncoderLayer_mHC(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    num_streams=self.num_streams,
                    dropout=configs.dropout,
                    resid_dropout=self.resid_dropout,
                    ffn_dropout=self.ffn_dropout,
                    activation=configs.activation,
                    sinkhorn_iters=self.sinkhorn_iters,
                )
                for _ in range(configs.e_layers)
            ],
            d_model=configs.d_model,
            norm_layer=my_Layernorm(configs.d_model),
            num_streams=self.num_streams,
        )

        self.decoder = Decoder_mHC(
            [
                DecoderLayer_mHC(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    d_model=configs.d_model,
                    c_out=configs.c_out,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    num_streams=self.num_streams,
                    dropout=configs.dropout,
                    resid_dropout=self.resid_dropout,
                    ffn_dropout=self.ffn_dropout,
                    activation=configs.activation,
                    sinkhorn_iters=self.sinkhorn_iters,
                )
                for _ in range(configs.d_layers)
            ],
            d_model=configs.d_model,
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
            num_streams=self.num_streams,
        )

    def forecast(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)

        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]],
            device=x_enc.device,
            dtype=x_enc.dtype,
        )

        seasonal_init, trend_init = self.decomp(x_enc)

        trend_init = torch.cat(
            [trend_init[:, -self.label_len :, :], mean],
            dim=1,
        )

        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len :, :], zeros],
            dim=1,
        )

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)

        seasonal_part, trend_part = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )

        dec_out = seasonal_part + trend_part

        return dec_out, attns

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out, attns = self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                enc_self_mask=enc_self_mask,
                dec_self_mask=dec_self_mask,
                dec_enc_mask=dec_enc_mask,
            )

            dec_out = dec_out[:, -self.pred_len :, :]

            if self.output_attention:
                return dec_out, attns

            return dec_out

        return None
