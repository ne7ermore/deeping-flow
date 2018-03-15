import argparse

parser = argparse.ArgumentParser(
    description='A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION')

parser.add_argument('--logdir', type=str, default='logdir_{}')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--data', type=str, default='./data/corpus')
parser.add_argument('--ml_lr', type=float, default=0.001)
parser.add_argument('--mix_lr', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--enc_hsz', type=int, default=64)
parser.add_argument('--gamma', type=float, default=0.9984)
parser.add_argument('--clip_norm', type=float, default=5.)

args = parser.parse_args()

# ##############################################################################
# Load data
################################################################################
import time

from corpus import middle_load
from data_loader import DataLoader

data = middle_load(args.data)
args.d_max_len = data["max_w_len"]
args.l_max_len = data["max_l_len"]
args.src_vs = data['dict']['src_size']
args.tgt_vs = data['dict']['tgt_size']
args.dec_hsz = args.enc_hsz * 2
args.logdir = args.logdir.format(time.time())

training_data = DataLoader(
    data['train']['data'],
    data['train']['label'],
    data['max_w_len'],
    data['max_l_len'],
    data['dict']['tgt_size'],
    batch_size=args.batch_size)

validation_data = DataLoader(
    data['valid']['data'],
    data['valid']['label'],
    data['max_w_len'],
    data['max_l_len'],
    data['dict']['tgt_size'],
    batch_size=args.batch_size,
    shuffle=False)

id2word = data['dict']['id2word']

# ##############################################################################
# Training
# ##############################################################################
import tensorflow as tf
from tqdm import tqdm
import numpy as np

import os

from model import Summarizor

tf.set_random_seed(args.seed)

train_log = os.path.join(args.logdir, "train")
if not os.path.exists(train_log):
    os.makedirs(train_log)

model = Summarizor(args, args.batch_size)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1000000000)

sv = tf.train.Supervisor(logdir=train_log,
                         saver=saver,
                         global_step=model.global_step,
                         summary_op=None)

summary_writer = sv.summary_writer

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

try:
    with sv.managed_session(config=config) as sess:
        for batch in training_data:
            merged, step = model.ml_train_step(batch, sess)
            summary_writer.add_summary(merged, step)
            if step % 100 == 0:
                summary_writer.flush()

        for epoch in range(1, args.epochs + 1):
            if sv.should_stop():
                break
            # for batch in tqdm(training_data, mininterval=1, desc='Train Processing', leave=False):
            for batch in training_data:
                merged, step = model.mix_train_step(batch, sess)
                summary_writer.add_summary(merged, step)
                if step % 100 == 0:
                    summary_writer.flush()

            loss = 0.
            for batch in validation_data:
                _loss = model.eval_step(batch, sess)
                loss += _loss

            loss /= validation_data.data_size
            print("epoch - {} end | loss - {}".format(epoch, loss))
            print("=" * 40 + "Summarizor" + "=" * 40)
            words = model.generate(next(validation_data), sess)
            [print(" ".join([id2word[_id] for _id in ws])) for ws in words]
            print('-' * 90)

except KeyboardInterrupt:
    sv.stop()
