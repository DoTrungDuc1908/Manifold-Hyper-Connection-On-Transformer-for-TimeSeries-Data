import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import EncoderLayer # Bỏ import Encoder gốc
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

# Import module mHC
from hyper_connections.hyper_connections_mhc import HyperConnections

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x


# =========================================================================
# CLASS MỚI: mHC_Encoder
# =========================================================================
class mHC_Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None, d_model=512):
        super().__init__()
        self.num_layers = len(attn_layers)
        
        # Hàm phân tách và thu gọn luồng thặng dư
        self.expand_fn, self.reduce_fn = HyperConnections.get_expand_reduce_stream_functions(
            num_streams=self.num_layers
        )
        
        self.layers = nn.ModuleList()
        for i, branch in enumerate(attn_layers):
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
        # Nhân bản luồng dữ liệu thành num_layers luồng song song
        residuals = self.expand_fn(x)
        
        attns = []
        for layer in self.layers:
            # mHC sẽ tự động xử lý tree_flatten để áp dụng đúng vào biến tensor thực tế
            out = layer(residuals, attn_mask=attn_mask)
            
            # Trích xuất attention maps nếu có
            if isinstance(out, tuple):
                residuals, attn = out
                attns.append(attn)
            else:
                residuals = out
                
        # Thu gọn luồng dữ liệu đa tạp về lại không gian gốc
        x = self.reduce_fn(residuals)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x, attns


# =========================================================================
# MODEL: iTransformer + mHC
# =========================================================================
class Model(nn.Module):
    """
    Kiến trúc iTransformer nguyên bản với mHC.
    Paper link: https://arxiv.org/abs/2310.06625
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # 1. Chuẩn hóa RevIN
        if self.use_norm:
            self.revin = RevIN(num_features=configs.enc_in, affine=True)
            
        # 2. Embedding Đảo ngược (Inverted Embedding)
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # 3. Mạng Encoder dùng mHC
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
                    
                    # ⚠️ QUAN TRỌNG: Ép tắt Residual Connection gốc
                    no_skip=True 
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            d_model=configs.d_model
        )
        
        # 4. Đầu ra Prediction Head
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projector = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projector = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            x_enc = self.revin(x_enc, mode='norm')

        _, _, N = x_enc.shape # N: Số lượng biến cần dự báo

        # Nhúng Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc) 
        
        # Đi qua Encoder mHC
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Phóng chiếu ra tương lai và loại bỏ covariates
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] 

        if self.use_norm:
            dec_out = self.revin(dec_out, mode='denorm')

        return dec_out, attns

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        if self.use_norm:
            x_enc = self.revin(x_enc, mode='norm')
            
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        
        if self.use_norm:
            dec_out = self.revin(dec_out, mode='denorm')
        return dec_out

    def anomaly_detection(self, x_enc):
        if self.use_norm:
            x_enc = self.revin(x_enc, mode='norm')
            
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        
        if self.use_norm:
            dec_out = self.revin(dec_out, mode='denorm')
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1) 
        output = self.projector(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.output_attention:
                return dec_out[:, -self.pred_len:, :], attns
            else:
                return dec_out[:, -self.pred_len:, :]
                
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
            
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
            
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
            
        return None