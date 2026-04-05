import torch
from torch import nn
import math

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)
    Giải quyết vấn đề trượt phân phối (distribution shift) với các tham số affine học được.
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

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0, fuse_decoder=False, decoder_type='conv2d'): 
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        self.fuse_decoder = fuse_decoder
        self.decoder_type = decoder_type
        if fuse_decoder:
            print('add a fuse layer of decoder')
            if self.decoder_type == 'conv2d':
                    kw = 8
                    self.fuse_proj = nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=(4+self.n_vars,kw),
                        padding='same'
                    )
            elif self.decoder_type == 'MLP':
                self.flat_proj = nn.Sequential(nn.Linear(n_vars*nf,n_vars*nf),nn.ReLU())

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        if self.fuse_decoder:
            if self.decoder_type == 'conv2d':
                x = x.unsqueeze(1)                
                x = self.fuse_proj(x)
                x = x.squeeze(1)
            elif self.decoder_type == 'MLP':
                s0,s1,s2 = x.shape
                x = x.reshape(s0,s1*s2)
                x = self.flat_proj(x)
                x = x.reshape(s0,s1,s2)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    """
    PatchTST chuẩn tích hợp RevIN
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        padding = stride
        
        self.no_skip = configs.no_skip
        self.fuse_decoder = configs.fuse_decoder
        self.decoder_type = configs.decoder_type
        self.no_zero_norm = configs.no_zero_norm

        # Khởi tạo RevIN thay vì tính toán thủ công
        # configs.enc_in đại diện cho số lượng biến (num_vars)
        if not self.no_zero_norm:
            self.revin = RevIN(num_features=configs.enc_in, affine=True)

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    no_skip=self.no_skip
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout, fuse_decoder=configs.fuse_decoder, decoder_type=configs.decoder_type)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout, fuse_decoder=configs.fuse_decoder, decoder_type=configs.decoder_type)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def get_attention(self, x_enc, x_mark_enc):
        if not self.no_zero_norm:
            x_enc = self.revin(x_enc, mode='norm')

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        return attns

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. Normalization (RevIN)
        if not self.no_zero_norm:
            x_enc = self.revin(x_enc, mode='norm')

        # 2. Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # 3. Encoder
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # 4. Decoder (Head)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # 5. De-Normalization (RevIN)
        if not self.no_zero_norm:
            dec_out = self.revin(dec_out, mode='denorm')
            
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Trong imputation, x_enc chứa các điểm bị khuyết (mask == 0).
        # Ta tạm thời gán các điểm khuyết bằng 0 sau khi chuẩn hóa để không làm hỏng giá trị RevIN.
        if not self.no_zero_norm:
            x_enc = self.revin(x_enc, mode='norm')
            x_enc = x_enc.masked_fill(mask == 0, 0)

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if not self.no_zero_norm:
            dec_out = self.revin(dec_out, mode='denorm')
        return dec_out

    def anomaly_detection(self, x_enc):
        if not self.no_zero_norm:
            x_enc = self.revin(x_enc, mode='norm')

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if not self.no_zero_norm:
            dec_out = self.revin(dec_out, mode='denorm')
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        if not self.no_zero_norm:
            x_enc = self.revin(x_enc, mode='norm')

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.output_attention:
                return dec_out[0][:, -self.pred_len:, :], dec_out[1]
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
    
    def latent_rep(self, x_enc, x_mark_enc, mask=None):
        if not self.no_zero_norm:
            x_enc = self.revin(x_enc, mode='norm')

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2]*enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 2, 1)
        
        return enc_out