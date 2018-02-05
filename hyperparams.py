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
        # Dataset
        self.Conll2000 = False
        self.Chunking = False
        # self.train_path = "./Data/Conll2000_Chunking/train.txt"
        # self.dev_path = None
        # self.test_path = "./Data/Conll2000_Chunking/test.txt"

        self.POS = False
        # self.train_path = "./Data/Conll2000_POS/train.txt"
        # self.dev_path = None
        # self.test_path = "./Data/Conll2000_POS/test.txt"

        self.Conll2003 = True
        self.NER = True
        # self.train_path = "./Data/conll2003/eng.train"
        # self.dev_path = "./Data/conll2003/eng.testa"
        # self.test_path = "./Data/conll2003/eng.testb"

        self.train_path = "./Data/Conll2003_NER/train.txt"
        self.dev_path = "./Data/Conll2003_NER/valid.txt"
        self.test_path = "./Data/Conll2003_NER/test.txt"



        #
        # self.train_path = "./Data/conll2003/eng.train"
        #
        # self.dev_path = "./Data/conll2003/eng.testa"


        #
        # self.test_path = "./Data/conll2003/eng.testb"

        self.shuffle = True
        self.epochs_shuffle = True

        # model
        self.model_PNC = True
        self.embed_dim = 100
        self.dropout = 0.5
        self.dropout_embed = 0.3
        self.clip_max_norm = 10

        # select optim algorhtim for train
        self.Adam = True
        self.learning_rate = 0.01
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
        self.word_Embedding = False
        self.ininital_from_Pretrained = True
        # self.word_Embedding_Path = "/home/lzl/mszhang/suda_file0120/file/file0120/richfeat/enwiki.emb.feature"
        # self.word_Embedding_Path = "./Pretrain_Embedding/enwiki.emb.source_Conll2000.txt"
        # self.word_Embedding_Path = "./Pretrain_Embedding/enwiki.emb.source_feat_Conll2000_1_NoZero.txt"
        # self.word_Embedding_Path = "./Pretrain_Embedding/enwiki.emb.source_Conll2003_OOV.txt"
        self.word_Embedding_Path = "./Pretrain_Embedding/richfeat.enwiki.emb.feature.small.0120.txt"
        # self.word_Embedding_Path = "./Pretrain_Embedding/parallel.enwiki.emb.feature.small"
        # self.word_Embedding_Path = "/home/lzl/mszhang/suda_file0120/file/file0120/context/pos_chunking_ner/enwiki.emb.source_Conll2003.txt"
        # self.word_Embedding_Path = "/home/lzl/mszhang/suda_file_0113/file/context/sentence_classification/enwiki.emb.source_CR.txt"
        # self.word_Embedding_Path = "/home/lzl/mszhang/suda_file_0113/file/context/enwiki.emb.source_CR.txt"

        # GPU
        self.use_cuda = False
        self.gpu_device = -1  # -1 meaning no use cuda
        self.num_threads = 1



