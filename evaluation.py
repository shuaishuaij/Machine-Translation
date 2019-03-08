# -*- coding=utf-8 -*-
from Transformer import Constants
from omnibox.tools import *
from omnibox.Torch import *



def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1] # .max(0) 按每列取最大的，返回 tuple(最大值， 最大值索引) |  .max(1) 则为按每行取。按行最大值（预测结果）的位置索引
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD) # .ne()逐个元素对比序列内容和mask标记，记下[true,true,false,false,...], true为不等 因此True的位置为非pad词
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item() # 只选择True的

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing: # 用平滑因子eps进行平滑后再算loss
        eps = 0.1
        n_class = pred.size(1)

        one_hot = t.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        # 做了两件事：1.按照pred的维度制作一个0向量 2.将gold标签放回去（1：按行， index， 值）最终输出：[0,0,1,0,0,1,1,0,...]

        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1) # 注意 log_softmax 里面有取对数有除法，有可能计算出nan

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later  这里其实算是手动实现cross_entropy了
    else: # 直接算loss，注意参数，ignore_index直接去除padding
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss