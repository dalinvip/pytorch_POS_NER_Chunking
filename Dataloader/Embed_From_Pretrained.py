# @Author : bamtercelboo
# @Datetime : 2018/2/3 14:03
# @File : Embed_From_Pretrained.py
# @Last Modify Time : 2018/2/3 14:03
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Embed_From_Pretrained.py
    FUNCTION : None
"""

import os
import sys
import torch
import torch.nn.init as init
import numpy as np
import random
import torch.nn as nn
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)


def Pretrain_Embed(file, alphabet, unk, padding):
    f = open(file, encoding='utf-8')
    allLines = f.readlines()
    # allLines = f.readlines()[1:]
    indexs = set()
    info = allLines[0].strip().split(' ')
    embDim = len(info) - 1
    emb = nn.Embedding(alphabet.vocab_size, embDim)

    init.uniform(emb.weight, a=-np.sqrt(3 / embDim), b=np.sqrt(3 / embDim))
    oov_emb = torch.zeros(1, embDim).type(torch.FloatTensor)

    now_line = 0
    for line in allLines:
        now_line += 1
        sys.stdout.write("\rHandling with the {} line.".format(now_line))
        info = line.split(' ')
        wordID = alphabet.loadWord2idAndId2Word(info[0])
        if wordID >= 0:
            indexs.add(wordID)
            for idx in range(embDim):
                val = float(info[idx + 1])
                emb.weight.data[wordID][idx] = val
                oov_emb[0][idx] += val
    f.close()
    print("\nHandle Finished.")
    count = len(indexs) + 1
    for idx in range(embDim):
        oov_emb[0][idx] /= count
    unkID = alphabet.loadWord2idAndId2Word(unk)
    paddingID = alphabet.loadWord2idAndId2Word(padding)
    for idx in range(embDim):
        emb.weight.data[paddingID][idx] = 0
    if unkID != -1:
        for idx in range(embDim):
            emb.weight.data[unkID][idx] = oov_emb[0][idx]
    print("Load Embedding file: ", file, ", size: ", embDim)
    oov = 0
    for idx in range(alphabet.vocab_size):
        if idx not in indexs:
            oov += 1
    print("oov: ", oov, " total: ", alphabet.vocab_size, "oov ratio: ", oov / alphabet.vocab_size)
    print("oov ", unk, "use avg value initialize")
    return emb, embDim