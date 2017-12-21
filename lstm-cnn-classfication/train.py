import argparse

parser = argparse.ArgumentParser(description='LSTM CNN Classification')

parser.add_argument('--device', type=str, default='/gpu:0')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=1111)
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
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from model import Model

with tf.device(args.device), tf.Graph().as_default():
     tf.set_random_seed(args.seed)
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

     def train_step(data, label):
          feed_dict = {
               model.input: data,
               model.label: label,
               model.dropout: args.dropout
          }

          _, step, loss, cor = sess.run([train_op, global_step, model.loss, model.corrects], feed_dict)
          return loss, cor

     def eval_step(data, label):
          feed_dict = {
               model.input: data,
               model.label: label,
               model.dropout: 1.
          }
          step, loss, cor = sess.run([global_step, model.loss, model.corrects], feed_dict)
          return loss, cor     

     train_losses, train_accs, eval_losses, eval_accs = [], [], [], []
     print('-' * 90)
     for epoch in range(1, args.epochs+1):
          losses = corrects = 0.
          for data, label in tqdm(training_data, mininterval=1, desc='Train Processing', leave=False):
               loss, cor = train_step(data, label)
               losses += loss
               corrects += cor
               current_step = tf.train.global_step(sess, global_step)
               
          train_loss = losses/training_data.stop_step
          train_acc = corrects/training_data.sents_size
          train_losses.append(train_loss)
          train_accs.append(train_acc)
          print("start of epoch - {} | loss - {} | acc - {}".format(epoch, train_loss, train_acc))
          print('-' * 90)

          losses = corrects = 0.
          for data, label in validation_data:
               loss, cor = eval_step(data, label)
               losses += loss
               corrects += cor
               
          eval_loss = losses/validation_data.stop_step
          eval_acc = corrects/validation_data.sents_size
          eval_losses.append(eval_loss)
          eval_accs.append(eval_acc)
          print("end of epoch - {} | loss - {} | acc - {}".format(epoch, eval_loss, eval_acc))
          print('-' * 90)

     print(train_losses)
     print(train_accs)
     print(eval_losses)
     print(eval_accs)
