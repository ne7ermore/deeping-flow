import argparse
import datetime

parser = argparse.ArgumentParser(description='SeqMatchSeq')

parser.add_argument('--device', type=str, default='/gpu:0',
                    help='train device')
parser.add_argument('--epochs', type=int, default=32,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size for training')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

parser.add_argument('--save', type=str, default='./dssm.model',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='./data/dssm_middle',
                    help='location of the data corpus')
parser.add_argument('--not-use-w2v', action='store_true',
                    help='no word2vec')
parser.add_argument('--w2v-file', type=str, default='./data/pre-train.w2v',
                    help='pre trained word2vec')

parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout)')
parser.add_argument('--attn-dim', type=int, default=0,
                    help='preprocess dim')
parser.add_argument('--emb-dim', type=int, default=128,
                    help='number of embedding dimension')
parser.add_argument('--l_2', type=float, default=.0,
                    help='l_2 regularization')
parser.add_argument('--eps', type=float, default=1e-6)
parser.add_argument('--filter-sizes', type=str, default='1,2,3',
                    help='filter sizes')
parser.add_argument('--num-filters', type=int, default=128,
                    help='number of filters')

args = parser.parse_args()

args.use_w2v = not args.not_use_w2v
args.filter_sizes = list(map(int, args.filter_sizes.split(",")))
import tensorflow as tf

# ##############################################################################
# set config
# ##############################################################################
tf.set_random_seed(args.seed)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# ##############################################################################
# Load data
# ##############################################################################
from data_loader import DataLoader
from utils import middle_load, load_pre_w2c

data = middle_load(args.data)
if args.use_w2v:
    args.w2v = load_pre_w2c(args.w2v_file, data['dict']['src'])


args.max_len = data["max_lenth_src"]
if args.attn_dim == 0: args.attn_dim = args.emb_dim

args.word_vocab = None

args.vocab_size = data['dict']['src_size']

training_data = DataLoader(
             data['train']['src'],
             data['train']['tgt'],
             data['train']['label'],
             args.max_len,
             batch_size=args.batch_size)

validation_data = DataLoader(
              data['valid']['src'],
              data['valid']['tgt'],
              data['valid']['label'],
              args.max_len,
              batch_size=args.batch_size,
              shuffle=False)

# ##############################################################################
# Training
# ##############################################################################
from model import Model
from tqdm import tqdm
import numpy as np

with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        model = Model(args)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(args.lr)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        sess.run(tf.global_variables_initializer())

    def train_step(q, d, label):
        feed_dict = {
            model.q_in: q,
            model.a_in: d,
            model.label: label,
            model.dropout: args.dropout
        }

        _, step, loss, acc = sess.run([train_op, global_step, model.loss, model.accuracy], feed_dict)
        return loss, acc

    def eval_step(q, d, label):
        feed_dict = {
            model.q_in: q,
            model.a_in: d,
            model.label: label,
            model.dropout: 1.
        }
        step, loss, acc = sess.run([global_step, model.loss, model.acc], feed_dict)
        return loss, accuracy

    print('-' * 90)
    for epoch in range(1, args.epochs+1):
        losses = accs = 0.
        for q, d, label in tqdm(training_data, mininterval=1,
                    desc='Train Processing', leave=False):
            loss, acc = train_step(q, d, label)
            losses += loss
            accs += acc
            current_step = tf.train.global_step(sess, global_step)
        print("epoch - {} | loss - {}".format(epoch, loss/training_data.stop_step, accs/training_data.stop_step))
        print('-' * 90)

        losses = accs = 0.
        for q, d, label in tqdm(validation_data, mininterval=1,
                    desc='Train Processing', leave=False):
            _loss, _acc = eval_step(q, d, label)
            loss += _loss
            accs += acc
        print("eval: loss - {} | acc - {}".format(loss / validation_data.stop_step, accs / validation_data.stop_step))
        print('-' * 90)
