# @Author : bamtercelboo
# @Datetime : 2018/1/31 10:01
# @File : train.py
# @Last Modify Time : 2018/1/31 10:01
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  train.py
    FUNCTION : None
"""

import os
import sys
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import shutil
import random
from eval import Eval, EvalPRF
from eval_bio import entity_evalPRF_exact
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)


def train(train_iter, test_iter, model, args):
    if args.use_cuda:
        model.cuda()

    optimizer = None
    if args.Adam is True:
        print("Adam Training......")
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)

    file = open("./Test_Result.txt", encoding="UTF-8", mode="a", buffering=1)
    best_acc = Best_Result()

    steps = 0
    model_count = 0
    model.train()
    max_dev_acc = -1
    train_eval = Eval()
    dev_eval = Eval()
    test_eval = Eval()
    for epoch in range(1, args.epochs+1):
        print("\n## The {} Epoch，All {} Epochs ！##".format(epoch, args.epochs))
        print("now lr is {}".format(optimizer.param_groups[0].get("lr")))
        random.shuffle(train_iter)
        model.train()
        for batch_count, batch_features in enumerate(train_iter):
            model.zero_grad()
            logit = model(batch_features)
            # print(logit.size())
            cal_train_acc(batch_features, train_eval, logit, args)
            loss = F.cross_entropy(logit.view(logit.size(0) * logit.size(1), logit.size(2)), batch_features.label_features)
            # print(loss)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                sys.stdout.write("\rbatch_count = [{}] , loss is {:.6f} , (correct/ total_num) = acc ({} / {}) = "
                                 "{:.6f}%".format(batch_count + 1, loss.data[0], train_eval.correct_num,
                                                  train_eval.gold_num, train_eval.acc() * 100))
        if steps is not 0:
            # print("\n{} epoch dev F-score".format(epoch))
            # print("\n")
            test_eval.clear()
            eval(test_iter, model, test_eval, file, best_acc, epoch, args)


def eval(data_iter, model, eval_instance, file, best_acc, epoch, args):
    # eval time
    for batch_features in data_iter:
        logit = model(batch_features)
        cal_train_acc(batch_features, eval_instance, logit, args)

    if eval_instance.acc() > best_acc.best_acc:
        best_acc.best_acc = eval_instance.acc()
        best_acc.best_epoch = epoch

    print("\neval: precision = {:.6f}%".format(eval_instance.acc() * 100))
    print("The Current Best Result: {:.6f}, Locate on {} Epoch.".format(best_acc.best_acc * 100, best_acc.best_epoch))
    file.write("The {} Epoch, All {} Epoch.\n".format(epoch, args.epochs))
    file.write("eval: precision = {:.6f}%\n".format(eval_instance.acc() * 100))
    file.write("The Current Best F-score: {:.6f}, Locate on {} Epoch.\n\n".format(best_acc.best_acc, best_acc.best_epoch))
    # print("\neval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%\n".format(p * 100, r * 100, f * 100))


def cal_train_acc(batch_features, train_eval, model_out, args):
    assert model_out.dim() == 3
    train_eval.clear()
    for id_batch in range(model_out.size(0)):
        inst = batch_features.inst[id_batch]
        for id_word in range(inst.words_size):
            maxId = getMaxindex(model_out[id_batch][id_word], model_out.size(2), args)
            if maxId == inst.label_index[id_word]:
                train_eval.correct_num += 1
        train_eval.gold_num += inst.words_size


def getMaxindex(model_out, label_size, args):
    max = model_out.data[0]
    maxIndex = 0
    for idx in range(1, label_size):
        if model_out.data[idx] > max:
            max = model_out.data[idx]
            maxIndex = idx
    return maxIndex


class Best_Result:
    def __init__(self):
        self.best_acc = -1
        self.best_epoch = 1


