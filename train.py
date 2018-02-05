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
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
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

    # lambda1 = lambda epoch: epoch // 5
    # lambda2 = lambda epoch: args.learning_rate_decay * epoch
    # print("lambda1 {} lambda2 {} ".format(lambda1, lambda2))
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=args.learning_rate_decay)

    file = open("./Test_Result.txt", encoding="UTF-8", mode="a", buffering=1)
    best_fscore = Best_Result()

    steps = 0
    model_count = 0
    model.train()
    max_dev_acc = -1
    train_eval = Eval()
    dev_eval = Eval()
    test_eval = Eval()
    # loss_F = nn.NLLLoss()
    # softmax = nn.Softmax(dim=1)
    for epoch in range(1, args.epochs+1):
        scheduler.step()
        print("\n## The {} Epoch，All {} Epochs ！##".format(epoch, args.epochs))
        print("now lr is {}".format(optimizer.param_groups[0].get("lr")))
        random.shuffle(train_iter)
        # random.shuffle(test_iter)
        model.train()
        for batch_count, batch_features in enumerate(train_iter):
            # model.zero_grad()
            optimizer.zero_grad()
            logit = model(batch_features)
            # print(logit.size())
            # print(batch_features.label_features.size())
            loss = F.cross_entropy(logit.view(logit.size(0) * logit.size(1), logit.size(2)), batch_features.label_features)
            # loss = loss_F(softmax(logit.view(logit.size(0) * logit.size(1), logit.size(2))),
            #               batch_features.label_features)
            # print(loss)
            loss.backward()
            # if args.clip_max_norm is not None:
            #     utils.clip_grad_norm(model.parameters(), max_norm=args.clip_max_norm)
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                sys.stdout.write("\rbatch_count = [{}] , loss is {:.6f}".format(batch_count + 1, loss.data[0]))
        if steps is not 0:
            eval(test_iter, model, test_eval, file, best_fscore, epoch, args)


def eval(data_iter, model, eval_instance, file, best_fscore, epoch, args):
    # eval time
    eval_instance.clear_PRF()
    eval_PRF = EvalPRF()
    gold_labels = []
    predict_labels = []
    for batch_features in data_iter:
        logit = model(batch_features)
        loss = F.cross_entropy(logit.view(logit.size(0) * logit.size(1), logit.size(2)),
                               batch_features.label_features, size_average=False)
        for id_batch in range(batch_features.batch_length):
            inst = batch_features.inst[id_batch]
            # eval_PRF = EvalPRF()
            predict_label = []
            for id_word in range(inst.words_size):
                maxId = getMaxindex(logit[id_batch][id_word], logit.size(2), args)
                # if maxId == args.create_alphabet.label_unkId:
                #     continue
                predict_label.append(args.create_alphabet.label_alphabet.from_id(maxId))
            gold_labels.append(inst.labels)
            predict_labels.append(predict_label)
            # print("ewe", len(inst.labels))
            # print("rrr", len(predict_label))
            eval_PRF.evalPRF(predict_labels=predict_label, gold_labels=inst.labels, eval=eval_instance)
            # p, r, f = eval_instance.getFscore()
        # p, r, f = entity_evalPRF_exact(gold_labels=gold_labels, predict_labels=predict_labels)

    # p = p * 100
    # f = f * 100
    # r = r * 100
    # calculate the F-Score
    p, r, f = eval_instance.getFscore()
    if f > best_fscore.best_fscore:
        best_fscore.best_fscore = f
        best_fscore.best_epoch = epoch
    print("\neval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%".format(p, r, f))
    print("The Current Best F-score: {:.6f}, Locate on {} Epoch.".format(best_fscore.best_fscore,
                                                                             best_fscore.best_epoch))
    file.write("The {} Epoch, All {} Epoch.\n".format(epoch, args.epochs))
    file.write("eval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%\n".format(p, r, f))
    file.write("The Current Best F-score: {:.6f}, Locate on {} Epoch.\n\n".format(best_fscore.best_fscore, best_fscore.best_epoch))
    # print("\neval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%\n".format(p * 100, r * 100, f * 100))


def cal_train_acc(batch_features, train_eval, model_out, args):
    assert model_out.dim() == 3
    # train_eval.clear_PRF()
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
        self.best_fscore = -1
        self.best_epoch = 1


