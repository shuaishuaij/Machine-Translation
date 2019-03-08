# -*- coding=utf-8 -*-
from Transformer import Constants
from omnibox.tools import *
from omnibox.Torch import *
from main import *
from dataset import TranslationDataset, paired_collate_fn
from parameters import *


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#      >> | Meta Env | <<
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default=mode, help='prepro/train/test/debug')
parser.add_argument('-use_gpu', action='store_true', help='select gpu or cpu model')


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#    >> | Prepro Settings | <<
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

if mode == 'prepro':
    parser.add_argument('-train_src', default=train_src)
    parser.add_argument('-train_tgt', default=train_tgt)
    parser.add_argument('-valid_src', default=valid_src)
    parser.add_argument('-valid_tgt', default=valid_tgt)
    parser.add_argument('-save_data', default=save_data)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=max_len, help='default: 50')
    parser.add_argument('-min_word_count', type=int, default=min_word_count, help='default: 5')
    parser.add_argument('-keep_case', action='store_true', default=True)
    parser.add_argument('-share_vocab', action='store_true', default=True)
    parser.add_argument('-vocab', default=vocab)
    opt = parser.parse_args()


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# >> | Train Settings | <<
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

if mode == 'train' or mode == 'debug':

    parser.add_argument('-data', default=data, help='path to the data')

    parser.add_argument('-epoch', type=int, default=epochs, help='default: 10')
    parser.add_argument('-batch_size', type=int, default=batch_size, help='default: 64')
    # parser.add_argument('-d_word_vec', type=int, default=d_word_vec, help='default: 512')
    parser.add_argument('-d_model', type=int, default=d_model, help='default: 512')
    parser.add_argument('-d_inner_hid', type=int, default=d_inner_hid, help='default: 2048')
    parser.add_argument('-d_k', type=int, default=d_k, help='default: 64')
    parser.add_argument('-d_v', type=int, default=d_v, help='default: 64')

    parser.add_argument('-n_head', type=int, default=n_head, help='default: 8')
    parser.add_argument('-n_layers', type=int, default=n_layers, help='default: 6')
    parser.add_argument('-n_warmup_steps', type=int, default=n_warmup_steps, help='default: 4000')

    parser.add_argument('-dropout', type=float, default=dropout, help='default: 0.1')
    parser.add_argument('-embs_share_weight', action='store_true', default=embs_share_weight, help='default: True')
    parser.add_argument('-proj_share_weight', action='store_true', default=proj_share_weight, help='default: True')

    parser.add_argument('-log', default=log, help='default: None')
    parser.add_argument('-save_model', default=save_model, help='default: None')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default=save_mode, help='default: best')

    parser.add_argument('-no_cuda', action='store_true', default=no_cuda, help='default: True')
    parser.add_argument('-label_smoothing', action='store_true', default=label_smoothing, help='default: True')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

if mode == 'test':

    parser.add_argument('-model', default=load_model_path,
                        help='Path to model .pt file')
    parser.add_argument('-src', default=src_file_path,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', default=vocab_file_path,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-output', default=os.path.join(output_path,'pred.txt') ,
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=beam_size,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=test_batch_size,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=n_best,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true', default=no_cuda, help='default: True')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda





def main(opt):
    if opt.mode == 'prepro':
        prepro(opt)
    if opt.mode == 'debug':
        opt.epoch = 1
        train(opt)
    if opt.mode == 'train':
        train(opt)
    if opt.mode == 'test':
        test(opt)


if __name__ == '__main__':
    main(opt)






























