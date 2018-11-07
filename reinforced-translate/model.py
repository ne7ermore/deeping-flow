import math

import tensorflow as tf
import numpy as np

from tqdm import tqdm

import editdistance

from const import BOS, PAD, EOS


def pad_mask(seq, index, shape, dtype=tf.float32):
    indexs = tf.constant(index, shape=shape, dtype=tf.int32)
    return tf.cast(tf.not_equal(seq, indexs), dtype=dtype)


class Levenshtein:
    def __init__(self, src_id2w, tgt_id2w):
        self.src_id2w = src_id2w
        self.tgt_id2w = tgt_id2w

    def compute_levenshtein(self, src_ix, tgt_ix):
        src = [[self.tgt_id2w[_id] for _id in ids] for ids in src_ix]
        tgt = [[self.tgt_id2w[_id] for _id in ids] for ids in tgt_ix]

        distances = (editdistance.eval(s, t) for s, t in zip(src, tgt))

        distances = np.array(list(distances), dtype='float32')
        return distances


class Model(object):
    def __init__(self, args, batch_size):
        self.args = args
        self.batch_size = batch_size
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('init_variables'):
            self.src = tf.placeholder(
                tf.int32, [batch_size, args.max_len], name="src")
            self.tgt = tf.placeholder(
                tf.int32, [batch_size, args.max_len], name="tgt")
            self.dropout = tf.placeholder(tf.float32, name="dropout")

        with tf.variable_scope('init_layers'):
            self.src_emb = tf.keras.layers.Embedding(args.src_vs, args.emb_dim)
            self.tgt_emb = tf.keras.layers.Embedding(args.tgt_vs, args.emb_dim)
            self.enc_cell = tf.nn.rnn_cell.GRUCell(args.rnn_hsz)
            self.dec_hidden = tf.keras.layers.Dense(args.rnn_hsz)
            self.dec_cell = tf.nn.rnn_cell.GRUCell(args.rnn_hsz)
            self.logits = tf.keras.layers.Dense(args.tgt_vs)

    def encode(self):
        emb = self.src_emb(self.src)
        _, hidden = tf.nn.dynamic_rnn(
            self.enc_cell, emb, dtype=tf.float32, scope="encode")

        if self.dropout != 1.:
            hidden = tf.nn.dropout(hidden, self.dropout)

        return self.dec_hidden(hidden)

    def teacher_forcing(self):
        with tf.variable_scope('teacher_forcing'):
            word = tf.constant(BOS, shape=[self.batch_size, 1], dtype=tf.int32)
            dec_state, emb, outputs = self.encode(), self.tgt_emb(word), []

            for step in range(self.args.max_len):
                if self.dropout != 1.:
                    dec_state = tf.nn.dropout(dec_state, self.dropout)

                _, dec_state = tf.nn.dynamic_rnn(
                    cell=self.dec_cell,
                    inputs=emb,
                    initial_state=dec_state,
                    dtype=tf.float32,
                    scope="decode")

                props = self.logits(dec_state)
                outputs.append(props[:, None])

                emb = self.tgt_emb(self.tgt[:, step][:, None])

            return tf.nn.log_softmax(tf.concat(outputs, 1))

    def sample(self, prev_index, max_props=True):
        with tf.variable_scope('sample'):
            word = tf.constant(BOS, shape=[self.batch_size, 1], dtype=tf.int32)
            outputs, words, emb = [], [], self.tgt_emb(word)

            dec_state = self.encode()

            for step in range(self.args.max_len):
                if self.dropout != 1.:
                    dec_state = tf.nn.dropout(dec_state, self.dropout)

                _, dec_state = tf.nn.dynamic_rnn(
                    cell=self.dec_cell,
                    inputs=emb,
                    initial_state=dec_state,
                    dtype=tf.float32)

                props = self.logits(dec_state)

                if max_props:
                    word = tf.expand_dims(
                        tf.cast(tf.argmax(props, -1), tf.int32), -1)
                else:
                    word = tf.cast(tf.multinomial(props, 1), tf.int32)
                    s_prop = tf.expand_dims(tf.gather_nd(
                        props, tf.concat((prev_index, word), -1)), -1)
                    outputs.append(s_prop)

                words.append(word)

                emb = self.tgt_emb(word)

            if max_props:
                return tf.concat(words, 1)
            else:
                return tf.concat(words, 1), tf.concat(outputs, 1)


class Supervisor:
    def __init__(self, model, args):
        self.args = args

        self.prev = tf.expand_dims(tf.constant(
            np.arange(args.batch_size * args.max_len, dtype=np.int32)), -1)

        with tf.variable_scope('supervisord_loss'):
            optimizer = tf.train.AdamOptimizer(
                args.ml_lr, beta1=0.9, beta2=0.98, epsilon=1e-8)

            props = model.teacher_forcing()
            props = tf.reshape(props, [-1, args.tgt_vs])

            tgt = tf.reshape(model.tgt, [-1, 1])

            _index = tf.concat((self.prev, tgt), -1)

            tgt_props = tf.reshape(tf.gather_nd(props, _index), [
                                   args.batch_size, args.max_len])
            mask = pad_mask(model.tgt, PAD, [args.batch_size, args.max_len])

            self.loss = -tf.reduce_sum(tgt_props * mask) / tf.reduce_sum(mask)
            self.train_op = optimizer.minimize(
                self.loss, global_step=model.global_step)

            tf.summary.scalar('loss', self.loss)

            self.merged = tf.summary.merge_all()


class Reinforced:
    def __init__(self, model, args):
        self.args = args

        self.levenshtein = Levenshtein(args.src_id2w, args.tgt_id2w)
        self.prev = tf.expand_dims(tf.constant(
            np.arange(args.batch_size, dtype=np.int32)), -1)

        with tf.variable_scope('reinforced_loss'):
            optimizer = tf.train.AdamOptimizer(
                args.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)

            b_words = model.sample(self.prev)
            s_words, s_props = model.sample(self.prev, False)

            rewards = self.compute_levenshtein(model.tgt, s_words)
            baseline = self.compute_levenshtein(model.tgt, b_words)
            advantage = rewards - baseline

            mask = pad_mask(model.tgt, EOS, [args.batch_size, args.max_len])

            self.loss = -tf.reduce_sum(s_props *
                                       mask * advantage[:, None]) / tf.reduce_sum(mask)
            self.train_op = optimizer.minimize(
                self.loss, global_step=model.global_step)

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar("reward", tf.reduce_mean(rewards))
            tf.summary.scalar("baseline", tf.reduce_mean(baseline))
            tf.summary.scalar("advantage", tf.reduce_mean(advantage))

            self.merged = tf.summary.merge_all()
            self.words = s_words

    def compute_levenshtein(self, words_ix, trans_ix):
        out = tf.py_func(self.levenshtein.compute_levenshtein,
                         [words_ix, trans_ix], tf.float32)
        out.set_shape([None])

        return tf.stop_gradient(out)
