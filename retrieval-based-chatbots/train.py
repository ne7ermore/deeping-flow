import argparse
import os

import tensorflow as tf
import numpy as np

from model import Model
from corpus import middle_load
from data_loader import DataLoader

parser = argparse.ArgumentParser(
    description='A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots')

parser.add_argument('--logdir', type=str, default='logdir')
parser.add_argument('--data', type=str, default='./data/corpus')

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=607)
parser.add_argument('--lr', type=float, default=.001)

parser.add_argument('--dropout', type=float, default=.5)
parser.add_argument('--emb_dim', type=int, default=200)
parser.add_argument('--first_rnn_hsz', type=int, default=200)
parser.add_argument('--fillters', type=int, default=8)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--match_vec_dim', type=int, default=50)
parser.add_argument('--second_rnn_hsz', type=int, default=50)


args = parser.parse_args()

data = middle_load(args.data)
args.max_cont_len = data["max_cont_len"]
args.max_utte_len = data["max_utte_len"]
args.dict_size = data['dict']['dict_size']
args.kernel_size = (args.kernel_size, args.kernel_size)

print("=" * 30 + "arguments" + "=" * 30)
for k, v in args.__dict__.items():
    if k in ("epochs", "seed", "data"):
        pass
    print("{}: {}".format(k, v))
print("=" * 60)

training_data = DataLoader(
    data['train']['utterances'],
    data['train']['responses'],
    data['train']['labels'],
    data['max_cont_len'],
    data['max_utte_len'],
    bsz=args.batch_size)

validation_data = DataLoader(
    data['test']['utterances'],
    data['test']['responses'],
    data['test']['labels'],
    data['max_cont_len'],
    data['max_utte_len'],
    bsz=args.batch_size,
    shuffle=False)

tf.set_random_seed(args.seed)
model = Model(args)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

sv = tf.train.Supervisor(logdir=args.logdir,
                         saver=saver,
                         global_step=model.global_step,
                         summary_op=None)

summary_writer = sv.summary_writer

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

lr = args.lr

with sv.managed_session(config=config) as sess:
    for epoch in range(1, args.epochs + 1):
        if sv.should_stop():
            break

        for batch in training_data:
            step, merged = model.train_step(batch, sess, args.dropout)
            summary_writer.add_summary(merged, step)
            if step % 100 == 0:
                summary_writer.flush()

        corrects = 0.
        for batch in validation_data:
            cor = model.eval_step(batch, sess)
            corrects += cor

        eval_acc = corrects / validation_data.sents_size
        print("epoch {} end | acc - {}".format(epoch, eval_acc))
        print('-' * 90)
