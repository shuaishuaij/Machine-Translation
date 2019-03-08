# -*- coding=utf-8 -*-
""" Manage beam search info structure.

    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""
import Transformer.Constants as Constants
from omnibox.tools import *
from omnibox.Torch import *

class BeamSearch(object):
    """ Beam Search """
    def __init__(self, size, device=False):  # size here refers to beam size = 5
        self.size = size
        self._done = False

        # The score for each translation on the beam.
        self.scores = t.zeros((size,), dtype=t.float, device=device)

        # Info of **>>   self.scores   <<** :
        # has type : <class 'torch.Tensor'>
        # has shape : torch.Size([5])
        # has len : 5
        # has value : tensor([0., 0., 0., 0., 0.])

        self.all_scores = []

        # The back pointers at each time-step.
        self.prev_ks = []     # previous k scores

        # The output at each time-step
        self.next_ys = [t.full((size,), Constants.PAD, dtype=t.long, device=device)]

        # Info of **>>   self.next_ys   <<** :
        # has type : <class 'list'>
        # has len : 1
        # has value : [tensor([0, 0, 0, 0, 0])]

        self.next_ys[0][0] = Constants.BOS  # 在最开头加上BOS = 2 （形状为一列，因此是tensor[0][0]）

        # Info of **>>   self.next_ys   <<** :
        # has type : <class 'list'>
        # has len : 1
        # has value : [tensor([2, 0, 0, 0, 0])]


    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """Get the backpointers for the current time step."""
        return self.prev_ks[-1]  # 返回最后上一轮n个beam的概率最大的词的绝对位置

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        """Update beam status and check if finished or not."""
        num_words = word_prob.size(1)

        # Info of **>>   word_prob   <<** :
        # has type : <class 'torch.Tensor'>
        # has shape : torch.Size([5, 3149])
        # has len : 5

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)     # 计算下一轮条件概率：因为使用了log，因此
        else:                                                                       # 概率乘法变成了log加法
            beam_lk = word_prob[0]

        # Info of **>>   word_prob[0]   <<** :
        # has type : <class 'torch.Tensor'>
        # has shape : torch.Size([3149])
        # has len : 3149
        # has value : tensor([-283.3875, -192.7573, -241.2736,  ..., -302.3145, -239.4608,
        #         -261.1005])

        # Info of **>>   self.scores.unsqueeze(1).expand_as(word_prob)   <<** :
        # has type : <class 'torch.Tensor'>
        # has shape : torch.Size([5, 3149])
        # has len : 5
        # has value : tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        #         [0., 0., 0.,  ..., 0., 0., 0.],
        #         [0., 0., 0.,  ..., 0., 0., 0.],
        #         [0., 0., 0.,  ..., 0., 0., 0.],
        #         [0., 0., 0.,  ..., 0., 0., 0.]])


        # 第一轮

        # Info of **>>   beam_lk   <<** :
        # has type : <class 'torch.Tensor'>
        # has shape : torch.Size([3149])
        # has len : 3149
        
        # 直接预测概率最大的第一个词的前n_beam项

        # 第二轮开始：
        
        # '''
        # Info of **>>   beam_lk   <<** :
        # has type : <class 'torch.Tensor'>
        # has shape : torch.Size([5, 3149])
        # has len : 5
        # '''

        flat_beam_lk = beam_lk.view(-1) # .view(-1) 进行flatten

        # '''
        # Info of **>>   flat_beam_lk   <<** :
        # has type : <class 'torch.Tensor'>
        # has shape : torch.Size([15745])
        # has len : 15745
        # '''
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort     -关键点1
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort
        
        # '''
        # Info of **>>   best_scores   <<** :
        # has type : <class 'torch.Tensor'>
        # has shape : torch.Size([5])
        # has len : 5
        # has value : tensor([   0.0000,  -89.4840, -104.7260, -118.5945, -128.7336])   
        # '''
        
        # '''
        # Info of **>>   best_scores_id   <<** :
        # has type : <class 'torch.Tensor'>
        # has shape : torch.Size([5])
        # has len : 5
        # has value : tensor([ 427,  250, 3606,    1, 3005])
        # '''
        
        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from , which means knowing the last word
        prev_k = best_scores_id / num_words
        
        # '''
        # Info of **>>   prev_k   <<** :
        # has type : <class 'torch.Tensor'>
        # has shape : torch.Size([5])
        # has len : 5
        # has value : tensor([0, 0, 1, 0, 0])
        # '''
        
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)  # 得到所选单词在词典里的绝对位置id
        
        # '''
        # Info of **>>   self.prev_ks   <<** :
        # has type : <class 'list'>
        # has len : 4
        # has value : [tensor([0, 0, 0, 0, 0]), tensor([0, 1, 0, 1, 0]), tensor([0, 1, 2, 0, 1]), tensor([0, 1, 2, 3, 4])]
        # '''
        
        # '''
        # Info of **>>   self.next_ys   <<** :
        # has type : <class 'list'>
        # has len : 3
        # has value : [tensor([2, 0, 0, 0, 0]), tensor([572, 1406, 2942, 710, 1110]), tensor([1744,457,1817,698,524])]
        # '''

        # End condition is when top-of-beam is EOS.
        # Finally, it checks the stopping condition (if an EOS token is at the end of the most possible sequence)
        if self.next_ys[-1][0].item() == Constants.EOS:  # [0], in top-k, [0] is the most possible one
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return t.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[Constants.BOS] + h for h in hyps]
            dec_seq = t.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))



if __name__ == '__main__':
  pass













