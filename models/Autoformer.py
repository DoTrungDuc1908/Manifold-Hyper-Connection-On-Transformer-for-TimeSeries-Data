import torch
import torch.nn as nn

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
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

        self.encoder = Encoder(
            [
                EncoderLayer(
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
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
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
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
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
