# @Author : bamtercelboo
# @Datetime : 2018/1/31 9:24
# @File : model_PNC.py
# @Last Modify Time : 2018/1/31 9:24
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  model_PNC.py
    FUNCTION : Part-of-Speech Tagging(POS), Named Entity Recognition(NER) and Chunking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable as Variable
import random
import torch.nn.init as init
import numpy as np
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)


class PNC(nn.Module):

    def __init__(self, args):
        super(PNC, self).__init__()
        self.args = args
        self.cat_size = 2

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        paddingId = args.paddingId

        self.embed = nn.Embedding(V, D, padding_idx=paddingId)

        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.embed.weight.requires_grad = False

        self.dropout_embed = nn.Dropout(args.dropout_embed)
        self.dropout = nn.Dropout(args.dropout)

        self.linear = nn.Linear(in_features=D * 5, out_features=C, bias=True)
        init.xavier_uniform(self.linear.weight)
        self.linear.bias.data.uniform_(-np.sqrt(6 / (D + 1)), np.sqrt(6 / (D + 1)))

    def cat_embedding(self, x):
        # print(x)
        batch = x.size(0)
        word_size = x.size(1)
        cated_embed = torch.zeros(batch, word_size, self.args.embed_dim * 5)
        for id_batch in range(batch):
            batch_word_list = np.array(x[id_batch].data).tolist()
            batch_word_list.insert(0, [0] * self.args.embed_dim)
            batch_word_list.insert(1, [0] * self.args.embed_dim)
            batch_word_list.insert(word_size, [0] * self.args.embed_dim)
            batch_word_list.insert(word_size + 1, [0] * self.args.embed_dim)
            batch_word_embed = torch.from_numpy(np.array(batch_word_list)).type(torch.FloatTensor)
            cat_list = []
            for id_word in range(word_size):
                cat_list.append(torch.cat(batch_word_embed[id_word:(id_word + 5)]).unsqueeze(0))
            sentence_cated_embed = torch.cat(cat_list)
            cated_embed[id_batch] = sentence_cated_embed
        if self.args.use_cuda is True:
            cated_embed = Variable(cated_embed).cuda()
        else:
            cated_embed = Variable(cated_embed)
        return cated_embed

    def forward(self, batch_features):
        word = batch_features.word_features
        x = self.embed(word)  # (N,W,D)
        cated_embed = self.cat_embedding(x)
        logit = self.linear(cated_embed)
        return logit

