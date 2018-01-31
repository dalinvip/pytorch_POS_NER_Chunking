# @Author : bamtercelboo
# @Datetime : 2018/1/30 16:05
# @File : hyperparams.py
# @Last Modify Time : 2018/1/30 16:05
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  hyperparams.py
    FUNCTION :
        The File is used to config hyper parameters for train, model etc.
"""

import torch
import random
torch.manual_seed(121)
random.seed(121)
# random seed
seed_num = 233


class Hyperparams():
    def __init__(self):
        # Datasets
        self.Conll2000 = True
        self.train_path = "./Data/conll2000/train.txt"
        # self.train_path = "./Data/conll2000/train_test.txt"
        self.dev_path = None
        self.test_path = "./Data/conll2000/test.txt"
        # self.test_path = "./Data/conll2000/test_test.txt"

        self.Conll2003 = False
        # self.train_path = "./Data/conll2003/eng.train"
        # self.dev_path = "./Data/conll2003/eng.testa"
        # self.test_path = "./Data/conll2003/eng.testb"

        self.shuffle = True
        self.epochs_shuffle = True

        # model
        self.model_PNC = True
        self.embed_dim = 100
        self.dropout = 0.5
        self.dropout_embed = 0.3
        self.clip_max_norm = 5

        # select optim algorhtim for train
        self.Adam = True
        self.learning_rate = 0.001
        self.learning_rate_decay = 1   # value is 1 means not change lr
        # L2 weight_decay
        self.weight_decay = 1e-8  # default value is zero in Adam SGD
        # self.weight_decay = 0   # default value is zero in Adam SGD
        self.epochs = 1000
        self.train_batch_size = 16
        self.dev_batch_size = None  # "None meaning not use batch for dev"
        self.test_batch_size = None  # "None meaning not use batch for test"
        self.log_interval = 1
        self.dev_interval = 100
        self.test_interval = 100
        self.save_dir = "snapshot"
        # whether to delete the model after test acc so that to save space
        self.rm_model = True

        # min freq to include during built the vocab, default is 1
        self.min_freq = 1

        # word_Embedding
        self.word_Embedding = True
        self.word_Embedding_Path = "./Pretrain_Embedding/enwiki.emb.source_RT2k_OOV.txt"
        # self.word_Embedding_Path = "/home/lzl/mszhang/suda_file0120/sentence_classification_richfeat/enwiki.emb.source_feat_SST1.txt"
        # self.word_Embedding_Path = "/home/lzl/mszhang/suda_file_0113/file/context/sentence_classification/enwiki.emb.source_CR.txt"
        # self.word_Embedding_Path = "/home/lzl/mszhang/suda_file_0113/file/context/enwiki.emb.source_CR.txt"

        # GPU
        self.use_cuda = False
        self.gpu_device = -1  # -1 meaning no use cuda
        self.num_threads = 1



