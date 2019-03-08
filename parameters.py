# Meta Env
mode = 'test'
no_cuda = True

if mode == 'prepro':
    # Pre-Processing
    train_src = ''
    train_tgt = ''
    valid_src = ''
    valid_tgt = ''
    save_data = ''
    vocab = None
    min_word_count = 5
    max_len = 50

if mode == 'train' or mode == 'debug':
    # Train
    data = './data/WMT_multi30k.atok.low.pt'
    epochs = 10
    batch_size = 64
    d_word_vec = None
    d_model = 512
    d_inner_hid = 2048
    d_k = 64
    d_v = 64
    n_head = 8
    n_layers = 6
    dropout = 0.1
    n_warmup_steps = 4000
    label_smoothing = True
    save_mode = 'best'
    save_model = 'trained-20190306'
    log = './log.txt'
    embs_share_weight = False
    proj_share_weight = True

if mode == 'test':
    # Test
    load_model_path = './trained-20190306.chkpt'  # Path to model .pt file
    src_file_path = './data/WMT_multi30k/test.en.atok'    # Source sequence to decode (one line per sequence)
    vocab_file_path = './data/WMT_multi30k.atok.low.pt'  # Source sequence to decode (one line per sequence)
    output_path = './'      # Path to output the predictions (each line will be the decoded sequence)
    beam_size = 5         # Beam size
    test_batch_size = 30  # Batch size
    n_best = 1            # If verbose is set, will output the n_best decoded sentences











