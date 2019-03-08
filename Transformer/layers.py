# -*- coding=utf-8 -*-

import Transformer.Constants as Constants
from omnibox.tools import *
from omnibox.Torch import *


# &&&&&&&&&&&&&&&&&&&&&&&&&&&
# >> | some sub-layers | <<
# &&&&&&&&&&&&&&&&&&&&&&&&&&&

# scaled dot attention
class ScaledDotAttention(nn.Module):
    def __init__(self, scale_factor, attn_dropout=0.1):
        super(ScaledDotAttention, self).__init__()
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V, mask):
        att = t.bmm(Q, K.transpose(1,2))
        att = att/self.scale_factor
        if mask is not None:
            att = att.masked_fill_(mask, -np.inf)
        att = self.softmax(att)
        att = self.dropout(att)
        return t.bmm(att, V), att


# multi-head attention layer
class MultiHeadAttention(nn.Module):
    """解释一下dim_k和dim_v的问题：\
        在经典用法下，dim_k和dim_v是相同的，本质上是decoder输入的y与encoder输出的k进行点积再归一化\
        得到attention权重，最后把权重作用到encoder输出上(即默认使用K，命名为V，此时dim_k和dim_v是相\
        同的), 但实质上, 两者却可以是不同的，权重可以加诸到进行维度变换后的 V = W * K 上，例如ELMo的\
        做法，使得输出的dim_v长于dim_k。因此在一般情况下，Q, K, V三者第二个维度是一样的。
    """
    def __init__(self, num_head, dim_model, dim_k, dim_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.w_qs = nn.Linear(dim_model, num_head*dim_k)
        self.w_ks = nn.Linear(dim_model, num_head*dim_k)
        self.w_vs = nn.Linear(dim_model, num_head*dim_v)

        init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2. / (dim_model + dim_k)))
        init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2. / (dim_model + dim_k)))
        init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2. / (dim_model + dim_v)))

        self.attention = ScaledDotAttention(scale_factor=np.power(dim_k, 0.5))
        self.layer_norm = nn.LayerNorm(dim_model)

        self.fc = nn.Linear(num_head*dim_v, dim_model)
        init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.dim_k, self.dim_v, self.num_head
        sz_b, len_q, _ = q.size() # sz_b = batch size, len_q= num of words in q, _ = embed_dim
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


# position_wise_feed_forward networks
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_input, d_hidden, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_input, d_hidden, kernel_size=1) # position wise
        self.w_2 = nn.Conv1d(d_hidden, d_input, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_input)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        h = x.transpose(1,2)
        h = self.w_1(h)
        h = F.relu(h)

        h = self.w_2(h)
        h = h.transpose_(1,2)

        h = self.dropout(h)
        output = self.layer_norm(h+residual)

        return output


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# >> | Encoder - Decoder layers | <<
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

class EncoderLayer(nn.Module):
    """ Encoder端包含两层：[多头注意力层，按位置的全连接FF层]"""
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, self_attn_mask=None):
        enc_output, enc_slf_attn = self.self_att(enc_input, enc_input, enc_input,
                                                 mask=self_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, self_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.self_attn(
            dec_input, dec_input, dec_input, mask=self_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn











