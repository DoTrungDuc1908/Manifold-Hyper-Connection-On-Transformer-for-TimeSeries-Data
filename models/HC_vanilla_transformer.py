import torch
import torch.nn as nn

from layers.Transformer_EncDec import (
    Encoder_HC,
    EncoderLayer_HC,
    Decoder_HC,
    DecoderLayer_HC,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.pred_len = configs.pred_len
        self.output_attention = getattr(configs, "output_attention", False)

        self.num_streams = getattr(configs, "num_streams", 4)
        self.resid_dropout = getattr(configs, "resid_dropout", configs.dropout)
        self.ffn_dropout = getattr(configs, "ffn_dropout", configs.dropout)

        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        self.dec_embedding = DataEmbedding(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
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
            norm_layer=nn.LayerNorm(configs.d_model),
            num_streams=self.num_streams
        )

        self.decoder = Decoder_HC(
            [
                DecoderLayer_HC(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
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
                for _ in range(configs.d_layers)
            ],
            d_model=configs.d_model,
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(
                configs.d_model,
                configs.c_out,
                bias=True
            ),
            num_streams=self.num_streams
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(
            x_enc,
            x_mark_enc
        )

        # Encoder_HC internally:
        # [B, L_enc, D] -> [B, L_enc, S, D] -> [B, L_enc, D]
        enc_out, attns = self.encoder(
            enc_out,
            attn_mask=None
        )

        dec_out = self.dec_embedding(
            x_dec,
            x_mark_dec
        )

        # Decoder_HC internally:
        # [B, L_dec, D] -> [B, L_dec, S, D] -> [B, L_dec, D] -> projection
        dec_out = self.decoder(
            dec_out,
            enc_out,
            x_mask=None,
            cross_mask=None
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
