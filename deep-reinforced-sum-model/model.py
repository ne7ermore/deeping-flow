import math

import tensorflow as tf
import numpy as np

from const import BOS, PAD
from rouge import rouge_l
from attention import intra_temp_atten, intra_decoder_atten


def pad_mask(seq, index, shape, dtype=tf.float32):
    indexs = tf.constant(index, shape=shape, dtype=tf.int64)
    return tf.cast(tf.not_equal(seq, indexs), dtype=dtype)


def _gather_index(props, tgt, prev):
    tgt = tf.reshape(tgt, [-1, 1])
    index = tf.concat((prev, tgt), -1)

    return tf.gather_nd(props, index)


def embedding(name_op, vocab_size, emb_dim, uniform_init, zero_pad=True):
    emb_matrix = tf.get_variable(name_op,
                                 dtype=tf.float32,
                                 shape=[vocab_size, emb_dim],
                                 initializer=uniform_init)
    if zero_pad:
        emb_matrix = tf.concat((tf.zeros(shape=[1, emb_dim]),
                                emb_matrix[1:, :]), 0)

    return emb_matrix


class Summarizor(object):
    def __init__(self, args, batch_size):
        self.args = args
        self.batch_size = batch_size
        self.global_step = tf.train.get_or_create_global_step()
        self.prev = tf.expand_dims(tf.constant(
            np.arange(args.batch_size * args.l_max_len, dtype=np.int64)), -1)

        std = 1. / math.sqrt(args.tgt_vs)
        self.uniform_init = tf.random_uniform_initializer(
            -std, std, seed=args.seed)

        self.init_placeholders()
        self.init_graph()

    def init_placeholders(self):
        args = self.args

        self.doc = tf.placeholder(
            tf.int64, [self.batch_size, args.d_max_len], name="doc")

        self.tgt = tf.placeholder(
            tf.int64, [self.batch_size, args.l_max_len], name="tgt")

        self.dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def init_graph(self):
        args = self.args

        with tf.variable_scope("embbeding"):
            self.src_emb = tf.keras.layers.Embedding(
                args.src_vs, args.emb_dim)
            self.tgt_emb_matrix = embedding(
                "tgt_emb_matrix", args.tgt_vs, args.emb_dim, self.uniform_init)
        self.w_enc = tf.get_variable("w_enc_atten", [args.dec_hsz,
                                                     args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init)
        self.w_dec = tf.get_variable("w_dec_atten", [args.dec_hsz,
                                                     args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init)
        self.fw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(args.enc_hsz)
        self.bw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(args.enc_hsz)
        self.dec_hidden = tf.keras.layers.Dense(args.dec_hsz)
        self.dec_cell = tf.nn.rnn_cell.BasicLSTMCell(args.dec_hsz)
        self.w_proj = tf.get_variable("w_proj", [args.emb_dim,
                                                 args.dec_hsz * 3], dtype=tf.float32, initializer=self.uniform_init)
        self.w_out = tf.nn.tanh(
            tf.transpose(tf.matmul(self.tgt_emb_matrix, self.w_proj), perm=[1, 0]))
        self.b_out = tf.Variable(
            tf.constant(0.1, shape=[args.tgt_vs]), name='b_out')
        self.w_base_proj = tf.get_variable("w_base", [args.emb_dim,
                                                      args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init)

    def encode(self):
        args = self.args

        src_emb = self.src_emb(self.doc)
        fw_hidden = self.fw_rnn_cell.zero_state(
            self.batch_size, dtype=tf.float32)
        bw_hidden = self.bw_rnn_cell.zero_state(
            self.batch_size, dtype=tf.float32)

        encodes, (fw_states, bw_states) = tf.nn.bidirectional_dynamic_rnn(
            self.fw_rnn_cell, self.bw_rnn_cell, src_emb,
            initial_state_fw=fw_hidden,
            initial_state_bw=bw_hidden,
            dtype=tf.float32)

        encode_hidden = tf.concat((fw_states.h, bw_states.h), -1)

        if self.dropout != 1.:
            encode_hidden = tf.nn.dropout(encode_hidden, self.dropout)
        encode_hidden = self.dec_hidden(encode_hidden)

        enc_docs = tf.concat(encodes, -1)

        if self.dropout != 1.:
            enc_docs = tf.nn.dropout(enc_docs, self.dropout)
            encode_hidden = tf.nn.dropout(encode_hidden, self.dropout)

        encode_states = tf.contrib.rnn.LSTMStateTuple(h=encode_hidden,
                                                      c=self.dec_cell.zero_state(self.batch_size, dtype=tf.float32).c)
        return encode_states, enc_docs

    def teacher_forcing(self):
        """
        Return:
            props, size - [bsz*time*feats]
            words, size - [bsz*time]
        """
        dec_state, enc_docs = self.encode()

        tgt_emb = tf.nn.embedding_lookup(self.tgt_emb_matrix, tf.constant(
            BOS, shape=[self.batch_size, 1], dtype=tf.int64))

        outputs, args, pre_enc_hiddens = [], self.args, []

        for step in range(args.l_max_len):
            if self.dropout != 1.:
                tgt_emb = tf.nn.dropout(tgt_emb, self.dropout)

            dec_out, dec_state = tf.nn.dynamic_rnn(
                cell=self.dec_cell,
                inputs=tgt_emb,
                initial_state=dec_state,
                dtype=tf.float32)

            dec_hidden = dec_state.h

            if self.dropout != 1.:
                dec_out = tf.nn.dropout(dec_out, self.dropout)
                dec_hidden = tf.nn.dropout(dec_hidden, self.dropout)

            enc_c_t = intra_temp_atten(
                self.w_enc, enc_docs, dec_hidden, args.d_max_len, pre_enc_hiddens)

            if step == 0:
                # We set dec_c to a vector of zeros since the generated sequence is empty on the first decoding step
                dec_c_t = tf.constant(
                    0., shape=[self.batch_size, args.dec_hsz])
            else:
                dec_c_t = intra_decoder_atten(
                    self.w_dec, dec_hidden, dec_out)

            out = tf.concat((dec_hidden, enc_c_t, dec_c_t), -1)

            props = tf.expand_dims(tf.nn.xw_plus_b(
                out, self.w_out, self.b_out), 1)  # bsz*1*f

            tgt_emb = tf.nn.embedding_lookup(
                self.tgt_emb_matrix, tf.expand_dims(self.tgt[:, step], 1))

            outputs.append(props)

        return tf.nn.log_softmax(tf.concat(outputs, 1))

    def sample(self, max_props=True):
        dec_state, enc_docs = self.encode()

        tgt_emb = tf.nn.embedding_lookup(self.tgt_emb_matrix, tf.constant(
            BOS, shape=[self.batch_size, 1], dtype=tf.int64))

        outputs, words = [], []
        args, pre_enc_hiddens = self.args, []

        for step in range(args.l_max_len):
            if self.dropout != 1.:
                tgt_emb = tf.nn.dropout(tgt_emb, self.dropout)

            dec_out, dec_state = tf.nn.dynamic_rnn(
                cell=self.dec_cell,
                inputs=tgt_emb,
                initial_state=dec_state,
                dtype=tf.float32)

            dec_hidden = dec_state.h

            if args.dropout != 1.:
                dec_out = tf.nn.dropout(dec_out, args.dropout)
                dec_hidden = tf.nn.dropout(dec_hidden, args.dropout)

            enc_c_t = intra_temp_atten(
                self.w_enc, enc_docs, dec_hidden, args.d_max_len, pre_enc_hiddens)

            if step == 0:
                # We set dec_c to a vector of zeros since the generated sequence is empty on the first decoding step
                dec_c_t = tf.constant(
                    0., shape=[self.batch_size, args.dec_hsz])
            else:
                dec_c_t = intra_decoder_atten(
                    self.w_dec, dec_hidden, dec_out)

            out = tf.concat((dec_hidden, enc_c_t, dec_c_t), -1)

            props = tf.nn.log_softmax(
                tf.nn.xw_plus_b(out, self.w_out, self.b_out))  # bsz*feats

            if max_props:
                word = tf.expand_dims(tf.argmax(props, -1), -1)
            else:
                word = tf.multinomial(tf.exp(props), 1)
                outputs.append(props[:, None, :])

            words.append(word)
            tgt_emb = tf.nn.embedding_lookup(self.tgt_emb_matrix, word)

        if max_props:
            return tf.concat(words, 1)
        else:
            return tf.concat(words, 1), tf.concat(outputs, 1)


class Supervisor:
    def __init__(self, model, args):
        self.args = args

        with tf.variable_scope('supervisor_loss'):
            optimizer = tf.train.AdamOptimizer(
                args.ml_lr, beta1=0.9, beta2=0.98, epsilon=1e-8)

            loss = self.compute_loss(model)

            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (
                        tf.clip_by_norm(grad, args.clip_norm), var)
            self.train_op = optimizer.apply_gradients(
                gradients, global_step=model.global_step)

            tf.summary.scalar('loss', loss)
            self.merged = tf.summary.merge_all()

    def compute_loss(self, model):
        args = self.args

        props = model.teacher_forcing()
        props = tf.reshape(props, [-1, args.tgt_vs])
        tgt_props = tf.reshape(_gather_index(props, model.tgt, model.prev), [
                               args.batch_size, args.l_max_len])
        mask = pad_mask(model.tgt, PAD, [args.batch_size, args.l_max_len])

        loss = -tf.reduce_sum(tgt_props * mask) / tf.reduce_sum(mask)

        return loss


class Reinforced:
    def __init__(self, model, args):
        self.args = args

        with tf.variable_scope('reinforced_loss'):
            optimizer = tf.train.AdamOptimizer(
                args.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)

            loss, reward, baseline, advantage = self.compute_loss(model)

            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (
                        tf.clip_by_norm(grad, args.clip_norm), var)
            self.train_op = optimizer.apply_gradients(
                gradients, global_step=model.global_step)

            tf.summary.scalar('loss', loss)
            tf.summary.scalar("reward", tf.reduce_mean(reward))
            tf.summary.scalar("baseline", tf.reduce_mean(baseline))
            tf.summary.scalar("advantage", tf.reduce_mean(advantage))
            self.merged = tf.summary.merge_all()

    def compute_loss(self, model):
        args = self.args

        b_words = model.sample()
        s_words, props = model.sample(False)

        s_props = tf.reshape(_gather_index(tf.reshape(
            props, [-1, args.tgt_vs]), s_words, model.prev), [args.batch_size, args.l_max_len])

        baseline = tf.py_func(rouge_l, [b_words, model.tgt], tf.float32)
        reward = tf.py_func(rouge_l, [s_words, model.tgt], tf.float32)
        advantage = reward - baseline

        mask = pad_mask(model.tgt, EOS, [args.batch_size, args.l_max_len])

        loss = -tf.reduce_sum(s_props *
                              mask * advantage[:, None]) / tf.reduce_sum(mask)
        if args.entropy_reg != 0.:
            entropy = -tf.reduce_sum(tf.nn.softmax(props) * props, axis=-1)
            loss -= args.entropy_reg * \
                tf.reduce_sum(entropy * mask) / tf.reduce_sum(mask)

        return loss, reward, baseline, advantage


class MixTrain:
    def __init__(self, model, args):
        self.args = args

        with tf.variable_scope('mix_loss'):
            optimizer = tf.train.AdamOptimizer(
                args.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)

            loss, reward, baseline, advantage = self.compute_loss(model)

            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (
                        tf.clip_by_norm(grad, args.clip_norm), var)
            self.train_op = optimizer.apply_gradients(
                gradients, global_step=model.global_step)

            tf.summary.scalar('loss', loss)
            tf.summary.scalar("reward", tf.reduce_mean(reward))
            tf.summary.scalar("baseline", tf.reduce_mean(baseline))
            tf.summary.scalar("advantage", tf.reduce_mean(advantage))

            self.merged = tf.summary.merge_all()

    def compute_loss(self, model):
        args = self.args

        mask = pad_mask(model.tgt, EOS, [args.batch_size, args.l_max_len])

        props = model.teacher_forcing()
        props = tf.reshape(props, [-1, args.tgt_vs])
        tgt_props = tf.reshape(_gather_index(props, model.tgt, model.prev), [
            args.batch_size, args.l_max_len])
        ml_loss = -tf.reduce_sum(tgt_props * mask) / tf.reduce_sum(mask)

        b_words = model.sample()
        s_words, props = model.sample(False)
        s_props = tf.reshape(_gather_index(tf.reshape(
            props, [-1, args.tgt_vs]), s_words, model.prev), [args.batch_size, args.l_max_len])

        baseline = tf.py_func(rouge_l, [b_words, model.tgt], tf.float32)
        reward = tf.py_func(rouge_l, [s_words, model.tgt], tf.float32)
        advantage = reward - baseline

        rl_loss = -tf.reduce_sum(s_props *
                                 mask * advantage[:, None]) / tf.reduce_sum(mask)
        if args.entropy_reg != 0.:
            entropy = -tf.reduce_sum(tf.nn.softmax(props) * props, axis=-1)
            rl_loss -= args.entropy_reg * \
                tf.reduce_sum(entropy * mask) / tf.reduce_sum(mask)

        loss = args.gamma * rl_loss + (1. - args.gamma) * ml_loss
        self.words = s_words

        return loss, reward, baseline, advantage
