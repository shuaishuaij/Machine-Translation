# -*- coding=utf-8 -*-
import Transformer.Constants as Constants
from omnibox.tools import *
from omnibox.Torch import *
from evaluation import *
from dataset import TranslationDataset, paired_collate_fn
from Transformer.model import Transformer
from Transformer.optimizer import ScheduledOptim
from dataset import collate_fn, TranslationDataset
from Transformer.goTranslator import TransformerTranslator



def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file, 'r', encoding='utf-8') as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()  # 先全部小写
            words = sent.split()  # 因为已经分好词，按分词符拆开
            if len(words) > max_sent_len:  # 裁剪超出长度的sentence
                trimmed_sent_count += 1  # 同时记录一下裁剪了多少个句子
            word_inst = words[:max_sent_len]  # 对所有句子做，上述条件判断只是记录数据

            if word_inst:  # 裁剪完后,如果还剩下词，前后加上BOS和EOS，再作为word_insts: [BOS, w1, w2, ..., EOS]
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:  # 否则裁剪完不剩下词了，则当作None
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts  # 返回list, 里面每个项是[BOS, w1, w2, ..., EOS]

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in
            word_insts]  # dict.get() 函数返回指定键的值，如果值不在字典中返回默认值, 这里即UNK。

def build_vocab_idx(word_insts, min_word_count):
    """ Trim vocab by number of occurence """

    full_vocab = set(w for sent in word_insts for w in sent)  # 将word_insts里的所有词集中起来，进行set()去重，得到完整词表
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}  # 制作字典, 前四项放常规的四个标识符 0,1,2,3

    word_count = {w: 0 for w in full_vocab}  # 词频字典：每个从词典里取出的词, 先全部初始化为0

    for sent in word_insts:  # 遍历word_insts, 对于每个词，出现一次则在字典里面对应的词的value+1
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0  # 忽略的词数
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:  # 至少出现min_word_count次的词才被考虑，
                word2idx[word] = len(word2idx)  # 这个词被加入word2idx字典，id从4开始，4，5，6，7，...(前0，1，2，3是标识符)
            else:
                ignored_word_count += 1  # 否则词频要求不达标，则ignore

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
            'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx  # 这样即得到{'<blank>':0, '<unk>':1, '<s>':2, '</s>':3, 'word1':4, 'word2':5, ...}的字典



def prepro(opt):
    """ Functions to pre-processing data"""
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts)) # 若src和tgt数量不相等，则按照少的那个的数量裁剪多的那个
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    valid_src_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    valid_tgt_word_insts = read_instances_from_file(
        opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # Build vocabulary
    if opt.vocab:
        predefined_data = t.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src'] # json文件，同时包含了src和tgt吧，都在'dict'这个key下
        tgt_word2idx = predefined_data['dict']['tgt']
    else:   # 若没有预定义的vocab，则重建vocab
        if opt.share_vocab:  # 情况一，src和tgt的词同时放入一起制作字典
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:                # 情况二，src和tgt的词分开制作字典
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # word to index  注意[train_src_insts, train_tgt_insts]是一组， [valid_src_insts, valid_tgt_insts]是一组
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    # 最后，全部封装进一个dict，并使用torch.save存为pickle文件
    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    t.save(data, opt.save_data)
    print('[Info] Finish.')


def train(opt):
    """ Functions to train the model """

    def train_epoch(model, training_data, optimizer, device, smoothing):
        """ Epoch operation in training phase """

        model.train()

        total_loss = 0
        n_word_total = 0
        n_word_correct = 0

        for batch in tqdm(
                training_data, mininterval=2,
                desc='  - (Training)   ', leave=False):
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            optimizer.zero_grad()  # 清空梯度
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)  # 计算预测值

            # backward
            loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)  # 计算loss
            loss.backward()  # 回传梯度

            # update parameters
            optimizer.step_and_update_lr()  # 更新参数

            # note keeping
            total_loss += loss.item()  # 记录总loss

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

        loss_per_word = total_loss / n_word_total  # 计算平均到每个词的梯度
        accuracy = n_word_correct / n_word_total  # 计算accuracy
        return loss_per_word, accuracy

    def eval_epoch(model, validation_data, device):
        """ Epoch operation in evaluation phase """

        model.eval()

        total_loss = 0
        n_word_total = 0
        n_word_correct = 0

        with t.no_grad():
            for batch in tqdm(
                    validation_data, mininterval=2,
                    desc='  - (Validation) ', leave=False):
                # prepare data
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
                gold = tgt_seq[:, 1:]

                # forward
                pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
                loss, n_correct = cal_performance(pred, gold, smoothing=False)

                # note keeping
                total_loss += loss.item()

                non_pad_mask = gold.ne(Constants.PAD)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy

    def prepare_dataloaders(data, opt):
        # ========= Preparing DataLoader =========#
        train_loader = t.utils.data.DataLoader(
            TranslationDataset(
                src_word2idx=data['dict']['src'],
                tgt_word2idx=data['dict']['tgt'],
                src_insts=data['train']['src'],
                tgt_insts=data['train']['tgt']),
            num_workers=2,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn,
            shuffle=True)

        valid_loader = t.utils.data.DataLoader(
            TranslationDataset(
                src_word2idx=data['dict']['src'],
                tgt_word2idx=data['dict']['tgt'],
                src_insts=data['valid']['src'],
                tgt_insts=data['valid']['tgt']),
            num_workers=2,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn)
        return train_loader, valid_loader


    def train_run(model, training_data, validation_data, optimizer, device, opt):
        log_train_file = None
        log_valid_file = None

        if opt.log:
            log_train_file = opt.log + '.train.log'
            log_valid_file = opt.log + '.valid.log'

            print('[Info] Training performance will be written to file: {} and {}'.format(
                log_train_file, log_valid_file))

            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch,loss,ppl,accuracy\n')
                log_vf.write('epoch,loss,ppl,accuracy\n')

        valid_accus = []
        for epoch_i in range(opt.epoch):  # 正式开始训练的循环
            print('[ Epoch', epoch_i, ']')

            start = time.time()
            train_loss, train_accu = train_epoch(
                model, training_data, optimizer, device, smoothing=opt.label_smoothing)
            print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, \
                  elapse: {elapse:3.3f} min'.format(
                      ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                      elapse=(time.time()-start)/60))

            start = time.time()
            valid_loss, valid_accu = eval_epoch(model, validation_data, device)
            print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, \
                    elapse: {elapse:3.3f} min'.format(
                        ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                        elapse=(time.time()-start)/60))

            valid_accus += [valid_accu]

            model_state_dict = model.state_dict() # Returns a dictionary containing a whole state of the module.
            checkpoint = {
                'model': model_state_dict,
                'settings': opt,
                'epoch': epoch_i}

            if opt.save_model:
                if opt.save_mode == 'all':
                    model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                    t.save(checkpoint, model_name)
                elif opt.save_mode == 'best':
                    model_name = opt.save_model + '.chkpt'
                    if valid_accu >= max(valid_accus):
                        t.save(checkpoint, model_name)
                        print('    - [Info] The checkpoint file has been updated.')

            if log_train_file and log_valid_file:
                with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                    log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                        epoch=epoch_i, loss=train_loss,
                        ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                    log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                        epoch=epoch_i, loss=valid_loss,
                        ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # >> | MAIN TRAINING LOOP | <<
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    #========= Loading Dataset =========#
    data = t.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = t.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_project_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train_run(transformer, training_data, validation_data, optimizer, device ,opt)




def train_GPU(opt):
    """ Functions to train the model """

    def train_epoch(model, training_data, optimizer, device, smoothing):
        """ Epoch operation in training phase """

        model.train()

        total_loss = 0
        n_word_total = 0
        n_word_correct = 0

        for batch in tqdm(
                training_data, mininterval=2,
                desc='  - (Training)   ', leave=False):
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            optimizer.zero_grad()  # 清空梯度
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)  # 计算预测值

            # backward
            loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)  # 计算loss
            loss.backward()  # 回传梯度

            # update parameters
            optimizer.step_and_update_lr()  # 更新参数

            # note keeping
            total_loss += loss.item()  # 记录总loss

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

        loss_per_word = total_loss / n_word_total  # 计算平均到每个词的梯度
        accuracy = n_word_correct / n_word_total  # 计算accuracy
        return loss_per_word, accuracy

    def eval_epoch(model, validation_data, device):
        """ Epoch operation in evaluation phase """

        model.eval()

        total_loss = 0
        n_word_total = 0
        n_word_correct = 0

        with t.no_grad():
            for batch in tqdm(
                    validation_data, mininterval=2,
                    desc='  - (Validation) ', leave=False):
                # prepare data
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
                gold = tgt_seq[:, 1:]

                # forward
                pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
                loss, n_correct = cal_performance(pred, gold, smoothing=False)

                # note keeping
                total_loss += loss.item()

                non_pad_mask = gold.ne(Constants.PAD)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy

    def prepare_dataloaders(data, opt):
        # ========= Preparing DataLoader =========#
        train_loader = t.utils.data.DataLoader(
            TranslationDataset(
                src_word2idx=data['dict']['src'],
                tgt_word2idx=data['dict']['tgt'],
                src_insts=data['train']['src'],
                tgt_insts=data['train']['tgt']),
            num_workers=2,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn,
            shuffle=True)

        valid_loader = t.utils.data.DataLoader(
            TranslationDataset(
                src_word2idx=data['dict']['src'],
                tgt_word2idx=data['dict']['tgt'],
                src_insts=data['valid']['src'],
                tgt_insts=data['valid']['tgt']),
            num_workers=2,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn)
        return train_loader, valid_loader


    def train_run(model, training_data, validation_data, optimizer, device, opt):
        log_train_file = None
        log_valid_file = None

        if opt.log:
            log_train_file = opt.log + '.train.log'
            log_valid_file = opt.log + '.valid.log'

            print('[Info] Training performance will be written to file: {} and {}'.format(
                log_train_file, log_valid_file))

            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch,loss,ppl,accuracy\n')
                log_vf.write('epoch,loss,ppl,accuracy\n')

        valid_accus = []
        for epoch_i in range(opt.epoch):  # 正式开始训练的循环
            print('[ Epoch', epoch_i, ']')

            start = time.time()
            train_loss, train_accu = train_epoch(
                model, training_data, optimizer, device, smoothing=opt.label_smoothing)
            print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, \
                  elapse: {elapse:3.3f} min'.format(
                      ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                      elapse=(time.time()-start)/60))

            start = time.time()
            valid_loss, valid_accu = eval_epoch(model, validation_data, device)
            print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, \
                    elapse: {elapse:3.3f} min'.format(
                        ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                        elapse=(time.time()-start)/60))

            valid_accus += [valid_accu]

            model_state_dict = model.state_dict() # Returns a dictionary containing a whole state of the module.
            checkpoint = {
                'model': model_state_dict,
                'settings': opt,
                'epoch': epoch_i}

            if opt.save_model:
                if opt.save_mode == 'all':
                    model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                    t.save(checkpoint, model_name)
                elif opt.save_mode == 'best':
                    model_name = opt.save_model + '.chkpt'
                    if valid_accu >= max(valid_accus):
                        t.save(checkpoint, model_name)
                        print('    - [Info] The checkpoint file has been updated.')

            if log_train_file and log_valid_file:
                with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                    log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                        epoch=epoch_i, loss=train_loss,
                        ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                    log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                        epoch=epoch_i, loss=valid_loss,
                        ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # >> | MAIN TRAINING LOOP | <<
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    #========= Loading Dataset =========#
    data = t.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = t.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_project_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train_run(transformer, training_data, validation_data, optimizer, device ,opt)






    

def test(opt):
    """ Functions to test the model and implement machine translation"""
    # Prepare DataLoader
    preprocess_data = t.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    test_src_word_insts = read_instances_from_file(
        opt.src,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case)
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])

    test_loader = t.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=test_src_insts),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=collate_fn)

    translator = TransformerTranslator(opt)

    with open(opt.output, 'w', encoding='utf-8') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp, all_scores = translator.translate_batch(*batch)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
                    f.write(pred_line + '\n')
    print('[Info] Finished.')


