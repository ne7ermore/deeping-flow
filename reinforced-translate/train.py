import argparse

parser = argparse.ArgumentParser(
    description='A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION')

parser.add_argument('--pretrain_dir', type=str, default='./pretrain_dir')
parser.add_argument('--train_dir', type=str, default='./train_dir')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--pretrain_steps', type=int, default=25000)
parser.add_argument('--train_steps', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--data', type=str, default='./data/corpus')
parser.add_argument('--ml_lr', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--emb_dim', type=int, default=100)
parser.add_argument('--rnn_hsz', type=int, default=100)
parser.add_argument('--clip_norm', type=float, default=5.)

args = parser.parse_args()

# ##############################################################################
# Load data
################################################################################
import time

from corpus import middle_load
from data_loader import DataLoader

data = middle_load(args.data)
args.max_len = data["max_len"]
args.src_vs = data['dict']['src_size']
args.tgt_vs = data['dict']['tgt_size']

training_data = DataLoader(
    data['train']['data'],
    data['train']['label'],
    data['max_len'],
    batch_size=args.batch_size)

validation_data = DataLoader(
    data['valid']['data'],
    data['valid']['label'],
    data['max_len'],
    batch_size=args.batch_size)

args.src_id2w = data['dict']['src_id2w']
args.tgt_id2w = data['dict']['tgt_id2w']

# ##############################################################################
# Training
# ##############################################################################
import tensorflow as tf
from tqdm import tqdm
import numpy as np

import os

from model import Supervisor, Model, Reinforced

tf.set_random_seed(args.seed)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = Model(args, args.batch_size)
ml = Supervisor(model, args)
rl = Reinforced(model, args)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

sv = tf.train.Supervisor(logdir=args.train_dir,
                         saver=saver,
                         global_step=model.global_step,
                         summary_op=None)
summary_writer = sv.summary_writer

with sv.managed_session(config=config) as sess:
    for epoch in range(1, 11):
        for batch in tqdm(training_data, mininterval=1, desc='Pre-train Processing', leave=False):
            if sv.should_stop():
                break
            feed_dict = {
                model.src: batch.data,
                model.tgt: batch.label,
                model.dropout: args.dropout,
            }
            _, step, merged = sess.run(
                [ml.train_op, model.global_step, ml.merged], feed_dict)
            summary_writer.add_summary(merged, step)
            if step % 100 == 0:
                summary_writer.flush()

    for epoch in range(1, args.epochs + 1):
        for batch in tqdm(training_data, mininterval=1, desc='Reinforced-train Processing', leave=False):
            if sv.should_stop():
                break
            feed_dict = {
                model.src: batch.data,
                model.tgt: batch.label,
                model.dropout: args.dropout,
            }
            _, step, merged = sess.run(
                [rl.train_op, model.global_step, rl.merged], feed_dict)
            summary_writer.add_summary(merged, step)
            if step % 100 == 0:
                summary_writer.flush()

        print("=" * 30 + "epoch_{} result".format(epoch) + "=" * 30)
        batch = next(validation_data)
        feed_dict = {
            model.src: batch.data,
            model.tgt: batch.label,
            model.dropout: 1.,
        }
        words = sess.run([rl.words], feed_dict)
        [print(" ".join([args.tgt_id2w[_id] for _id in ids]))
         for ids in words[0]]
