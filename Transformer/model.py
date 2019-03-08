# -*- coding=utf-8 -*-
import Transformer.Constants as Constants
from omnibox.tools import *
from omnibox.Torch import *
from Transformer.layers import *


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# >> | some useful functions | <<
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(t.float).unsqueeze(-1)
    # unsqueeze(-1) = tf.expand_dims(tensor,axis=-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_pos_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_pos_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return t.FloatTensor(sinusoid_table)  # 为了防止默认tensor类型改变，主动定义为FloatTensor


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """

    sz_b, len_s = seq.size()
    subsequent_mask = t.triu(
        t.ones((len_s, len_s), device=seq.device, dtype=t.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)  # q 的 第二维度 d_k
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# >> | Encoder-Decoder Modules | <<
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


class Encoder(nn.Module):
    """ Encoder model with self-attn mechanism"""
    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super(Encoder, self).__init__()
        n_position = len_max_seq + 1
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])  # 按照 n_layers 定义的数量用 ModuleList 将 encoder layers 串联起来

    def forward(self, src_seq, src_pos, return_attn=False):
        enc_self_attn_list = []

        # -- prepare masks
        self_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)  # self-attention, seq_k=seq_q
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)  # enc_output = encoding output

        for enc_layer in self.layer_stack:
            enc_output, enc_self_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                self_attn_mask=self_attn_mask)
            if not return_attn: continue
            else:
                enc_self_attn_list += [enc_self_attn]

        if return_attn: return enc_output, enc_self_attn_list
        return enc_output,  # 这逗号也太骚了


class Decoder(nn.Module):
    """ Decoder model with self-attn mechanism"""
    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super(Decoder, self).__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        dec_self_attn_list, dec_enc_attn_list = [], []  # 两个list分别储存两个attention层的注意力权值

        # -- prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        self_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        self_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq)+self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                self_attn_mask=self_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if not return_attns: continue
            else:
                dec_self_attn_list += [dec_self_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns: return dec_output, dec_self_attn_list, dec_enc_attn_list
        return dec_output,


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# >> | Transformer Main Model | <<
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


class Transformer(nn.Module):
    """ Seq2Seq model with purely attention mechanism"""
    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_project_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_project = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_project.weight)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        if tgt_emb_project_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_project.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
                "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]  # 注意：在axis=1上都只取到倒数第二个

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_project(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))














