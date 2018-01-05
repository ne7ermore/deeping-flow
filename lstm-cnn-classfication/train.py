import argparse

parser = argparse.ArgumentParser(description='LSTM CNN Classification')

parser.add_argument('--device', type=str, default='/gpu:0')
parser.add_argument('--logdir', type=str, default='logdir')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--u_scope', type=float, default=.1)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--data', type=str, default='./data/corpus')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--hidden_sizes', type=str, default='64,128,128')
parser.add_argument('--l_2', type=float, default=.0)
parser.add_argument('--filter_sizes', type=str, default='2,3,4')
parser.add_argument('--num_filters', type=int, default=128)

args = parser.parse_args()

# ##############################################################################
# Load data
################################################################################
from corpus import middle_load
from data_loader import DataLoader

data = middle_load(args.data)
args.max_len = data["max_len"]
args.vocab_size = data['dict']['vocab_size']
args.label_size = data['dict']['label_size']
args.hidden_sizes = list(map(int, args.hidden_sizes.split(",")))
args.filter_sizes = list(map(int, args.filter_sizes.split(",")))

training_data = DataLoader(
             data['train']['src'],
             data['train']['label'],
             args.max_len,
             args.label_size,
             batch_size=args.batch_size)

validation_data = DataLoader(
              data['valid']['src'],
              data['valid']['label'],
              args.max_len,
              args.label_size,
              batch_size=args.batch_size,
              shuffle=False)

# ##############################################################################
# Training
# ##############################################################################
import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
import numpy as np

from model import Model

tf.set_random_seed(args.seed)

train_log = os.path.join(args.logdir, "train")
if not os.path.exists(train_log): 
    os.makedirs(train_log)

model = Model(args)

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

        print('-' * 90)       
        for epoch in range(1, args.epochs+1):
            if sv.should_stop(): break
            for batch in tqdm(training_data, mininterval=1, desc='Train Processing', leave=False):
                merged, step = model.train_step(batch, sess)
                summary_writer.add_summary(merged, step)

                if step % 1000 == 0:
                    summary_writer.flush()

            corrects = losses = 0
            for batch in validation_data:
                loss, cor = model.eval_step(batch, sess)
                losses += loss
                corrects += cor
               
            eval_loss = losses/validation_data.stop_step
            eval_acc = corrects/validation_data.sents_size
            print("epoch - {} | loss - {:.5f} | acc - {}".format(epoch, eval_loss, eval_acc))
            print('-' * 90)                

except KeyboardInterrupt:
    sv.stop()

