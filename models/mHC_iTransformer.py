import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import EncoderLayer # Đã bỏ import Encoder gốc
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

# Import thư viện mHC
from hyper_connections.hyper_connections_mhc import HyperConnections

# =========================================================================
# CLASS MỚI: mHC_Encoder (Thay thế cho Encoder gốc của Framework)
# =========================================================================
class mHC_Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None, d_model=512):
        super().__init__()
        self.num_layers = len(attn_layers)
        
        # Hàm mở rộng và thu gọn luồng theo chuẩn mHC
        self.expand_fn, self.reduce_fn = HyperConnections.get_expand_reduce_stream_functions(
            num_streams=self.num_layers
        )
        
        self.layers = nn.ModuleList()
        for i, branch in enumerate(attn_layers):
            # Khởi tạo mHC bọc xung quanh mỗi lớp EncoderLayer
            hc = HyperConnections(
                self.num_layers,
                dim=d_model,
                layer_index=i,
                branch=branch,
                num_fracs=1
            )
            self.layers.append(hc)
            
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # 1. Nhân bản luồng thặng dư (Expand)
        residuals = self.expand_fn(x)
        
        attns = []
        # 2. Cho luồng dữ liệu chảy qua đa tạp (manifold)
        for layer in self.layers:
            # HyperConnections sẽ tự động nhận diện tuple (out, attn_maps) 
            # để xử lý residual đúng vào `out` thông qua tree_flatten
            out = layer(residuals, attn_mask=attn_mask)
            if isinstance(out, tuple):
                residuals, attn = out
                attns.append(attn)
            else:
                residuals = out
                
        # 3. Thu gọn luồng thặng dư (Reduce)
        x = self.reduce_fn(residuals)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x, attns


# =========================================================================
# MAIN MODEL: iTransformer + mHC
# =========================================================================
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    Tích hợp mHC (Manifold-Constrained Hyper-Connections) thay thế Residual Connection
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # ⚠️ Ép buộc no_skip = True để tắt cơ chế Residual gốc của EncoderLayer
        self.no_skip = True 
        print('no_skip forced to True for mHC')
        
        self.fuse_decoder = configs.fuse_decoder
        self.decoder_type = configs.decoder_type
        self.no_zero_norm = configs.no_zero_norm
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # =====================================================
        # SỬ DỤNG mHC_Encoder THAY CHO Encoder GỐC
        # =====================================================
        self.encoder = mHC_Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    no_skip=self.no_skip # Đã bị ép thành True
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            d_model=configs.d_model # Truyền thêm d_model cho mHC config
        )
        # =====================================================
        
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if configs.fuse_decoder:
                print('add a fuse layer of decoder')
                if configs.decoder_type == 'conv2d':
                    kw = 8
                    self.fuse_proj = nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=(4+configs.enc_in,kw),
                        padding='same'
                    )
                elif configs.decoder_type == 'MLP':
                    self.fuse_proj = nn.Sequential(nn.Linear(configs.d_model * (4+configs.enc_in),configs.d_model * (4+configs.enc_in),bias=True),nn.ReLU())
             
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            
        if self.task_name == 'imputation':
            self.projector = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projector = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projector = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
            
    def get_attention(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return attns
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if not self.no_zero_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc = x_enc / stdev
        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Đưa qua mHC Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        if self.fuse_decoder:
            if self.decoder_type == 'conv2d':
                enc_out = enc_out.unsqueeze(1)
                enc_out = self.fuse_proj(enc_out)
                enc_out = enc_out.squeeze(1)
            elif self.decoder_type == 'MLP':
                s1,s2,s3 = enc_out.shape
                enc_out = enc_out.view(s1,s2*s3)
                flat_enc_out = self.fuse_proj(enc_out)
                enc_out = flat_enc_out.reshape(s1,s2,-1)

        dec_out = self.projector(enc_out)           
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization from Non-stationary Transformer
        if not self.no_zero_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projector(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.output_attention:
                dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :], attns
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
    
    def latent_rep(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return enc_out