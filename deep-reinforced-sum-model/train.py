import argparse

parser = argparse.ArgumentParser(description='A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION')

parser.add_argument('--device', type=str, default='/gpu:0')
parser.add_argument('--logdir', type=str, default='logdir')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--u_scope', type=float, default=.1)
parser.add_argument('--n_scope', type=float, default=.1)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--data', type=str, default='./data/corpus')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--enc_hsz', type=int, default=128)
parser.add_argument('--l_2', type=float, default=.0)
parser.add_argument('--gamma', type=float, default=0.9984)

args = parser.parse_args()

# ##############################################################################
# Load data
################################################################################
from corpus import middle_load
from data_loader import DataLoader

data = middle_load(args.data)
args.d_max_len = data["max_w_len"]
args.l_max_len = data["max_l_len"]
args.vocab_size = data['dict']['src_size']
args.dec_hsz = args.enc_hsz*2

training_data = DataLoader(
             data['train']['data'],
             data['train']['label'],
             data['max_w_len'],
             data['max_l_len'],
             data['dict']['src_size'],
             batch_size=args.batch_size)

validation_data = DataLoader(
              data['valid']['data'],
              data['valid']['label'],
              data['max_w_len'],
              data['max_l_len'],
              data['dict']['src_size'],
              batch_size=args.batch_size,
              shuffle=False)

# ##############################################################################
# Training
# ##############################################################################
import tensorflow as tf
from tensorflow.python import debug as tf_debug
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
saver = tf.train.Saver()

sv = tf.train.Supervisor(logdir=train_log,
                         saver=saver,
                         global_step=model.global_step,
                         summary_op=None)

summary_writer = sv.summary_writer

try:
    with sv.managed_session() as sess:
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        for epoch in range(1, args.epochs+1):
            if sv.should_stop(): break
            for batch in tqdm(training_data, mininterval=1, desc='Train Processing', leave=False):
                merged, step = model.train_step(batch, sess)
                summary_writer.add_summary(merged, step)

                if step % 100 == 0:
                    summary_writer.flush()

except KeyboardInterrupt:
    sv.stop()


