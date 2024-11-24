import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.PatchTST_layers import *
from typing import Callable, Optional
from torch import Tensor


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # Embedding
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_fft = configs.n_fft
        self.hop_length = configs.hop_length
        self.win_length = configs.win_length
        self.channels = configs.enc_in
        self.value = configs.value
        head_dropout = configs.head_dropout
        dropout = configs.dropout
        attn_dropout = configs.fc_dropout
        attn_dropout1 = configs.fc_dropout1
        dropout1 = configs.dropout1
        # Learnable filter
        self.F = self.n_fft // 2 + 1
        self.T = self.seq_len // self.hop_length + 1
        self.complex_weight = nn.Parameter(torch.randn(self.F, self.T, 2, dtype=torch.float32) * self.value)

        # Encoder
        res_attention = True
        pre_norm = True
        d_k = None
        d_v = None
        store_attn = False
        norm = 'BatchNorm'
        self.encoder = TSTEncoder_lrean(self.F, self.T, configs.n_heads, d_k=d_k, d_v=d_v, d_ff=configs.d_ff, norm=norm,
                                  attn_dropout=attn_dropout1, dropout=dropout1,
                                  pre_norm=pre_norm, activation=configs.activation, res_attention=res_attention, n_layers=configs.o_layers,
                                  store_attn=store_attn)
        self.encoder_F = TSTEncoder(configs.kernel_size, configs.stride, self.channels, self.F,configs.n_heads, d_k=d_k, d_v=d_v, d_ff=configs.d_ff_F, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=configs.activation, res_attention=res_attention, n_layers=configs.e_layers,
                                  store_attn=store_attn)
        self.encoder_T = TSTEncoder(configs.kernel_size, configs.stride_kernel,self.channels, self.T, configs.n_heads, d_k=d_k, d_v=d_v, d_ff=configs.d_ff_T,
                                    norm=norm,
                                    attn_dropout=attn_dropout, dropout=dropout,
                                    pre_norm=pre_norm, activation=configs.activation, res_attention=res_attention,
                                    n_layers=configs.e_layers,
                                    store_attn=store_attn)

        # RevIn
        self.revin_layer = RevIN(self.channels, affine=True, subtract_last=False)
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # REVIN
        z = self.revin_layer(x_enc, 'norm').permute(0, 2, 1)  # [BS, n_vars, seq_len]
        BS = z.shape[0]

        # STFT
        u = torch.reshape(z, (z.shape[0] * z.shape[1], z.shape[2]))  # [BS * n_vars, seq_len]
        output_stft = torch.stft(u, self.n_fft, self.hop_length, self.win_length, window=None, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=True)   # [BS * n_vars, F, T]

        # TF-enhanced
        MFCC1 = self.encoder(output_stft) + output_stft

        z = torch.reshape(MFCC1, (BS, self.channels, MFCC1.shape[-2], MFCC1.shape[-1]))  # [BS , n_vars, F, T]
        # F-transformer
        z_f = z.permute(0,3,1,2)
        z_f = torch.reshape(z_f, (z_f.shape[0] * z_f.shape[1], z_f.shape[2], z_f.shape[3]))  #  [BS * T , nvars, F]
        z_f = self.encoder_F(z_f)
        z_f = torch.reshape(z_f, (BS, self.T, z_f.shape[1], z_f.shape[2]))
        z_f = z_f.permute(0,2,3,1) + z
        # T-transformer
        z_t = z_f.permute(0,2,1,3)
        z_t = torch.reshape(z_t, (z_t.shape[0] * z_t.shape[1], z_t.shape[2], z_t.shape[3])) #  [BS * F ,T , nvars]
        z_t = self.encoder_T(z_t)
        z_t = torch.reshape(z_t, (BS, self.F, z_t.shape[1], z_t.shape[2]))
        z_t = z_t.permute(0,2,1, 3) + z_f
        #
        z = z_t# * z_f

        # iDCT-STFT
        MFCC1 = torch.reshape(z, (z.shape[0] * z.shape[1], z.shape[2], z.shape[3]))

        z1 = torch.istft(MFCC1, self.n_fft, self.hop_length, self.win_length, window=None, center=True, normalized=False, onesided=None, length=None, return_complex=False)
        z = torch.reshape(z1, (BS, self.channels, z1.shape[-1]))

        # Head Linear
        x = self.Linear(z).permute(0,2,1)

        x = self.dropout(x)
        x = self.revin_layer(x, 'denorm')
        return x  # [B, L, D]



# Cell
class TSTEncoder(nn.Module):
    def __init__(self, kernel_size, stride,q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(kernel_size, stride,q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                                         attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output

class TSTEncoderLayer(nn.Module):
    def __init__(self, kernel_size, stride, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.bn = ComplexBatchNorm1d(d_model, track_running_stats=False)

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention_fu(kernel_size, stride, d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm_attn_fu = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward

        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        self.ff_fu = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                   get_activation_fn(activation),
                                   nn.Dropout(dropout),
                                   nn.Linear(d_ff, d_model, bias=bias))
        self.Linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.Linear1_fu = nn.Linear(d_model, d_ff, bias=bias)
        self.activation = get_activation_fn(activation)
        self.activation_fu = get_activation_fn(activation)
        self.Dropout = nn.Dropout(dropout)
        self.Dropout_fu = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.Linear2_fu = nn.Linear(d_ff, d_model,  bias=bias)
        self.trans = nn.Sequential(Transpose(1,2), ComplexBatchNorm1d(d_model, track_running_stats=False), Transpose(1,2))



        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm_ffn_fu = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.bn(src)

        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn

        ## Add & Norm
        src2_real = src2.real
        src2_imag = src2.imag
        src2_real = self.dropout_attn(src2_real)
        src2_imag = self.dropout_attn(src2_imag)
        src2 = torch.complex(src2_real, src2_imag)
        src = src2 + src # Add: residual connection with residual dropout

        if not self.pre_norm:
            src = self.bn(src)


        # Feed-forward sublayer
        if self.pre_norm:
            src = self.bn(src)

        # Position-wise Feed-Forward

        src2_real = self.Linear1(src.real) - self.Linear1_fu(src.imag)
        src2_imag = self.Linear1(src.imag) + self.Linear1_fu(src.real)
        src2 = torch.complex(src2_real, src2_imag)
        src2 = torch.complex(self.activation(src2.real), self.activation_fu(src2.imag))
        src2 = torch.complex(self.Dropout(src2.real), self.Dropout_fu(src2.imag))
        src2 = torch.complex(self.Linear2(src2.real) - self.Linear2_fu(src2.imag), self.Linear2(src2.imag) + self.Linear2_fu(src2.real))

        # Add & Norm
        src2_real = src2.real
        src2_imag = src2.imag
        src2_real = self.dropout_ffn(src2_real)
        src2_imag = self.dropout_ffn(src2_imag)
        src = src + torch.complex(src2_real, src2_imag)

        if not self.pre_norm:
            src = self.bn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention_fu(nn.Module):
    def __init__(self, kernel_size, stride, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model  if d_k is None else d_k
        d_v = d_model  if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k , bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v , bias=qkv_bias)
        self.W_Qimag = nn.Linear(d_model, d_k, bias=qkv_bias)
        self.W_Kimag = nn.Linear(d_model, d_k, bias=qkv_bias)
        self.W_Vimag = nn.Linear(d_model, d_v, bias=qkv_bias)
        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear( d_v, d_model), nn.Dropout(proj_dropout))

        self.avg_pool = nn.AvgPool1d(kernel_size, stride)
        self.avg_pool_fu = nn.AvgPool1d(kernel_size, stride)


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        Qimag = Q.imag
        Kimag = K.imag
        Vimag = V.imag
        Q = Q.real
        K = K.real
        V = V.real


        Q_real_output = self.W_Q(Q) - self.W_Qimag(Qimag)
        Q_imag_output = self.W_Q(Qimag) + self.W_Qimag(Q)
        Q_real_output = Q_real_output
        Q_imag_output = Q_imag_output

        K_real_output1 = self.W_K(K) - self.W_Kimag(Kimag)
        K_imag_output1 = self.W_K(Kimag) + self.W_Kimag(K)
        K_real_output = self.avg_pool(K_real_output1.permute(0, 2, 1)) - self.avg_pool_fu(K_imag_output1.permute(0, 2, 1))
        K_imag_output = self.avg_pool(K_imag_output1.permute(0, 2, 1)) + self.avg_pool_fu(K_real_output1.permute(0, 2, 1))

        V_real_output1 = self.W_V(V) - self.W_Vimag(Vimag)
        V_imag_output1 = self.W_V(Vimag) + self.W_Vimag(V)
        V_real_output = self.avg_pool(V_real_output1.permute(0, 2, 1)).permute(0, 2, 1) - self.avg_pool_fu(V_imag_output1.permute(0, 2, 1)).permute(0, 2, 1)
        V_imag_output = self.avg_pool(V_imag_output1.permute(0, 2, 1)).permute(0, 2, 1) + self.avg_pool_fu(V_real_output1.permute(0, 2, 1)).permute(0, 2, 1)

        q_s = torch.complex(Q_real_output, Q_imag_output)
        k_s = torch.complex(K_real_output, K_imag_output)
        v_s = torch.complex(V_real_output, V_imag_output)

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output1, attn_weights1, attn_scores1 = self.sdp_attn(q_s.real, k_s.real, v_s.real, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output2, attn_weights2, attn_scores2 = self.sdp_attn(q_s.real, k_s.imag, v_s.imag, prev=prev,key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output3, attn_weights3, attn_scores3 = self.sdp_attn(q_s.imag, k_s.real, v_s.imag, prev=prev,key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output4, attn_weights4, attn_scores4 = self.sdp_attn(q_s.imag, k_s.imag, v_s.real, prev=prev,key_padding_mask=key_padding_mask, attn_mask=attn_mask)

            output5, attn_weights5, attn_scores5 = self.sdp_attn(q_s.real, k_s.real, v_s.imag, prev=prev,key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output6, attn_weights6, attn_scores6 = self.sdp_attn(q_s.real, k_s.imag, v_s.real, prev=prev,key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output7, attn_weights7, attn_scores7 = self.sdp_attn(q_s.imag, k_s.real, v_s.real, prev=prev,key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output8, attn_weights8, attn_scores8 = self.sdp_attn(q_s.imag, k_s.imag, v_s.imag, prev=prev,key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output1, attn_weights1, attn_scores1 = self.sdp_attn(q_s.real, k_s.real, v_s.real, prev=prev,
                                                                 key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output2, attn_weights2, attn_scores2 = self.sdp_attn(q_s.real, k_s.imag, v_s.imag, prev=prev,
                                                                 key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output3, attn_weights3, attn_scores3 = self.sdp_attn(q_s.imag, k_s.real, v_s.imag, prev=prev,
                                                                 key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output4, attn_weights4, attn_scores4 = self.sdp_attn(q_s.imag, k_s.imag, v_s.real, prev=prev,
                                                                 key_padding_mask=key_padding_mask, attn_mask=attn_mask)

            output5, attn_weights5, attn_scores5 = self.sdp_attn(q_s.real, k_s.real, v_s.imag, prev=prev,
                                                                 key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output6, attn_weights6, attn_scores6 = self.sdp_attn(q_s.real, k_s.imag, v_s.real, prev=prev,
                                                                 key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output7, attn_weights7, attn_scores7 = self.sdp_attn(q_s.imag, k_s.real, v_s.real, prev=prev,
                                                                 key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            output8, attn_weights8, attn_scores8 = self.sdp_attn(q_s.imag, k_s.imag, v_s.imag, prev=prev,
                                                                 key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        output=torch.complex((output1-output2-output3-output4),(output5+output6+output7-output8))
        attn_weights=torch.complex((attn_weights1-attn_weights2-attn_weights3-attn_weights4),(attn_weights5+attn_weights6+attn_weights7-attn_weights8))
        attn_scores=torch.complex((attn_scores1-attn_scores2-attn_scores3-attn_scores4),(attn_scores5+attn_scores6+attn_scores7-attn_scores8))

        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        outputimag = self.to_out(output.imag)
        outputreal = self.to_out(output.real)
        real_outputreal=outputreal-outputimag
        imag_outputreal=outputreal+outputimag
        output = torch.complex(real_outputreal,imag_outputreal)
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights




class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights




class TSTEncoder_lrean(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer_lrean_fu(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        if self.res_attention:
            for mod in self.layers: output = mod(output)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output

class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features,3))
            self.bias = nn.Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype = torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:,:2],1.4142135623730951)
            nn.init.zeros_(self.weight[:,2])
            nn.init.zeros_(self.bias)


class ComplexBatchNorm1d(_ComplexBatchNorm):
    def forward(self, input):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean_r = input.real.mean(dim=0).type(torch.complex64)
            mean_i = input.imag.mean(dim=0).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, ...]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = input.real.var(dim=0, unbiased=False) + self.eps
            Cii = input.imag.var(dim=0, unbiased=False) + self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=0)
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 0]

            self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 1]

            self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :] * input.real + Rri[None, :] * input.imag).type(torch.complex64) \
                + 1j * (Rii[None, :] * input.imag + Rri[None, :] * input.real).type(torch.complex64)

        if self.affine:
            input = (self.weight[None, :, 0] * input.real + self.weight[None, :, 2] * input.imag + \
                     self.bias[None, :, 0]).type(torch.complex64) \
                    + 1j * (self.weight[None, :, 2] * input.real + self.weight[None, :, 1] * input.imag + \
                            self.bias[None, :, 1]).type(torch.complex64)

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return input


class TSTEncoderLayer_lrean_fu(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.complex_weight = nn.Parameter(torch.randn(q_len, d_model, 2, dtype=torch.float32) * 0.02)
        self.bn2 = ComplexBatchNorm1d(d_model, track_running_stats=False)
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm_attn_fu = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward

        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        self.ff_fu = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                   get_activation_fn(activation),
                                   nn.Dropout(dropout),
                                   nn.Linear(d_ff, d_model, bias=bias))
        self.Linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.Linear1_fu = nn.Linear(d_model, d_ff, bias=bias)
        self.activation = get_activation_fn(activation)
        self.activation_fu = get_activation_fn(activation)
        self.Dropout = nn.Dropout(dropout)
        self.Dropout_fu = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.Linear2_fu = nn.Linear(d_ff, d_model,  bias=bias)


        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm_ffn_fu = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor) -> Tensor:
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.bn2(src)

        weight = torch.view_as_complex(self.complex_weight)
        src2 = src * weight

        ## Add & Norm
        src2_real = src2.real
        src2_imag = src2.imag
        src2_real = self.dropout_attn(src2_real)
        src2_imag = self.dropout_attn(src2_imag)
        src2 = torch.complex(src2_real, src2_imag)
        src = src2 + src  # Add: residual connection with residual dropout

        if not self.pre_norm:
            src = self.bn2(src)

        # Feed-forward sublaye
        if self.pre_norm:
            src = self.bn2(src)

        # Position-wise Feed-Forward
        src2_real = self.Linear1(src.real) - self.Linear1_fu(src.imag)
        src2_imag = self.Linear1(src.imag) + self.Linear1_fu(src.real)
        src2 = torch.complex(src2_real, src2_imag)
        src2 = torch.complex(self.activation(src2.real), self.activation_fu(src2.imag))
        src2 = torch.complex(self.Dropout(src2.real), self.Dropout_fu(src2.imag))
        src2 = torch.complex(self.Linear2(src2.real) - self.Linear2_fu(src2.imag), self.Linear2(src2.imag) + self.Linear2_fu(src2.real))


        # Add & Norm
        src2_real = src2.real
        src2_imag = src2.imag
        src2_real = self.dropout_ffn(src2_real)
        src2_imag = self.dropout_ffn(src2_imag)
        src =  torch.complex(src2_real, src2_imag) + src

        if not self.pre_norm:
            src = self.bn2(src)

        return src

