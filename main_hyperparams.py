# @Author : bamtercelboo
# @Datetime : 2018/1/30 19:50
# @File : main_hyperparams.py.py
# @Last Modify Time : 2018/1/30 19:50
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  main_hyperparams.py.py
    FUNCTION : main
"""

import os
import sys
import argparse
import datetime
import torch
from Dataloader.Alphabet import *
from Dataloader.Batch_Iterator import *
from Dataloader import DataConll2000_Loader
from Dataloader import DataConll2003_Loader
import random
import hyperparams as hy
# solve default encoding problem
from imp import reload
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

# init hyperparams instance
hyperparams = hy.Hyperparams()

parser = argparse.ArgumentParser(description="Text Classification for sentence level.")
# Data path
parser.add_argument('-train_path', type=str, default=hyperparams.train_path, help='train data path')
parser.add_argument('-dev_path', type=str, default=hyperparams.dev_path, help='dev data path')
parser.add_argument('-test_path', type=str, default=hyperparams.test_path, help='test data path')
# shuffle data
parser.add_argument('-shuffle', action='store_true', default=hyperparams.shuffle, help='shuffle the data when load data' )
parser.add_argument('-epochs_shuffle', action='store_true', default=hyperparams.epochs_shuffle, help='shuffle the data every epoch' )
# Datasets
parser.add_argument('-Conll2000', action='store_true', default=hyperparams.Conll2000, help='Conll2000 Dataset')
parser.add_argument('-Conll2003', action='store_true', default=hyperparams.Conll2003, help='=Conll2003 Dataset')
# model params
parser.add_argument("-SumPooling", action='store_true', default=hyperparams.SumPooling, help="SumPooling model")
parser.add_argument('-embed_dim', type=int, default=hyperparams.embed_dim, help='embedding dim')
parser.add_argument('-dropout', type=float, default=hyperparams.dropout, help='dropout')
parser.add_argument('-dropout_embed', type=float, default=hyperparams.dropout_embed, help='dropout')
parser.add_argument('-clip_max_norm', type=float, default=hyperparams.clip_max_norm, help='clip_norm params in train')
# Train
parser.add_argument("-Adam", action="store_true", default=hyperparams.Adam, help="elf.Adam = optimizer for train")
parser.add_argument('-lr', type=float, default=hyperparams.learning_rate, help='initial learning rate [default: 0.001]')
parser.add_argument('-learning_rate_decay', type=float, default=hyperparams.learning_rate_decay, help='learn rate decay')
parser.add_argument('-weight_decay', type=float, default=hyperparams.weight_decay, help='weight_decay')
parser.add_argument('-epochs', type=int, default=hyperparams.epochs, help="The number of iterations for train")
parser.add_argument('-batch_size', type=int, default=hyperparams.train_batch_size, help="The number of batch_size for train")
parser.add_argument('-dev_batch_size', type=int, default=hyperparams.dev_batch_size, help='batch size for dev [default: None]')
parser.add_argument('-test_batch_size', type=int, default=hyperparams.test_batch_size, help='batch size for test [default: None]')
parser.add_argument('-log_interval',  type=int, default=hyperparams.log_interval,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-dev_interval', type=int, default=hyperparams.dev_interval, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-test_interval', type=int, default=hyperparams.test_interval, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save_dir', type=str, default=hyperparams.save_dir, help='save model')
parser.add_argument('-rm_model', action="store_true", default=hyperparams.rm_model, help='remove model after test')
# build vocab
parser.add_argument("-min_freq", type=int, default=hyperparams.min_freq, help="build vocab for cut off")
# word_Embedding
parser.add_argument("-word_Embedding", action="store_true", default=hyperparams.word_Embedding, help="whether to use pretrained word embedding")
parser.add_argument("-word_Embedding_Path", type=str, default=hyperparams.word_Embedding_Path, help="Pretrained Embedding Path")
# GPU
parser.add_argument('-use_cuda', action='store_true', default=hyperparams.use_cuda, help='use gpu')
parser.add_argument("-gpu_device", type=int, default=hyperparams.gpu_device, help="gpu device number")
parser.add_argument("-num_threads", type=int, default=hyperparams.num_threads, help="threads number")
# option
args = parser.parse_args()


# load data / create alphabet / create iterator
def load_Conll2000(args):
    print("Loading Conll2000 Data......")
    # read file
    data_loader = DataConll2000_Loader.DataLoader()
    train_data, test_data = data_loader.dataLoader(path=[args.train_path, args.test_path], shuffle=args.shuffle)

    # create the alphabet
    create_alphabet = CreateAlphabet(min_freq=args.min_freq)
    create_alphabet.build_vocab(train_data=train_data, test_data=test_data)

    # create iterator
    create_iter = Iterators()
    train_iter, test_iter = create_iter.createIterator(batch_size=[args.batch_size, len(test_data)],
                                                       data=[train_data, test_data], operator=create_alphabet,
                                                       args=args)
    return train_iter, test_iter, create_alphabet


def main():
    # save file
    mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.mulu = mulu
    args.save_dir = os.path.join(args.save_dir, mulu)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # get iter
    train_iter, test_iter, create_alphabet = load_Conll2000(args)
    print("a")



if __name__ == "__main__":
    main()


