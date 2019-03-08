# -*- coding:utf-8 -*-
from omnibox.Torch import *
from omnibox.tools import *
from Transformer import Constants


def paired_collate_fn(insts):   # 成对的勘误函数， 就是把勘误函数map到src和tgt的insts数据集上
    src_insts, tgt_insts = list(zip(*insts)) # 星号：传入的参数将会先被“打包”成为一个元组，然后再当做参数传入。
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts) # 返回的形态其实类似于 ((a,b),(c,d))


def collate_fn(insts): # 重要的勘误函数
    ''' Pad the instance to the max seq length in batch 做 padding '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts]) # 对词序列padding batch的长度，使得整个batch的形状变得规整

    batch_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])# 对位置标记序列padding batch的长度，使得整个batch的形状变得规整

    batch_seq = t.LongTensor(batch_seq) # 转变为long tensor
    batch_pos = t.LongTensor(batch_pos) # 位置转变为long tensor可以理解，位置是整数

    return batch_seq, batch_pos


class TranslationDataset(tdata.Dataset):
    def __init__(
                self,
                src_word2idx, tgt_word2idx,  # 输入的参量
                src_insts=None, tgt_insts=None):

        assert src_insts
        assert not tgt_insts or (len(src_insts) == len(tgt_insts))

        src_idx2word = {idx: word for word, idx in src_word2idx.items()} # 制作idx2word字典，实质上只是把word2idx字典倒转过来
        self._src_word2idx = src_word2idx # 前面加下划线表明为不可修改的内参，相当于back-up副本
        self._src_idx2word = src_idx2word # 前面加下划线表明为不可修改的内参，相当于back-up副本
        self._src_insts = src_insts       # 前面加下划线表明为不可修改的内参，相当于back-up副本

        tgt_idx2word = {idx: word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._tgt_insts = tgt_insts

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_insts: # 如果有tgt_insts则返回src和tgt，否则只返回src
            return self._src_insts[idx], self._tgt_insts[idx]
        return self._src_insts[idx]
