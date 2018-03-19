import math

import tensorflow as tf
import numpy as np

from const import BOS
from rouge import rouge_l
from attention import intra_temp_atten, intra_decoder_atten


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
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        std = 1. / math.sqrt(args.tgt_vs)
        self.norm_init = tf.truncated_normal_initializer(
            stddev=std, seed=args.seed)

        self.uniform_init = tf.random_uniform_initializer(
            -std, std, seed=args.seed)

        self.ml_optimizer = tf.train.AdamOptimizer(
            self.args.ml_lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        self.optimizer = tf.train.AdamOptimizer(
            self.args.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)

        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="vars")

        self.init_placeholders()
        self.init_vars()
        self.build()

    def init_placeholders(self):
        args = self.args

        self.doc = tf.placeholder(
            tf.int64, [self.batch_size, args.d_max_len], name="doc")

        self.label = tf.placeholder(tf.float32, [
            self.batch_size, args.l_max_len, args.tgt_vs], name="label")

        self.label_ids = tf.placeholder(
            tf.int64, [self.batch_size, args.l_max_len], name="label")

        self.dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def init_vars(self):
        args = self.args

        with tf.variable_scope('init_variables'):
            self.src_emb_matrix = embedding(
                "src_emb_matrix", args.src_vs, args.emb_dim, self.uniform_init)
            self.tgt_emb_matrix = embedding(
                "tgt_emb_matrix", args.tgt_vs, args.emb_dim, self.uniform_init)

            self.w_enc = tf.get_variable("w_enc_atten", [args.dec_hsz,
                                                         args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init)

            self.w_dec = tf.get_variable("w_dec_atten", [args.dec_hsz,
                                                         args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init)

            self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(args.dec_hsz)

            self.w_proj = tf.get_variable("w_proj", [args.emb_dim,
                                                     args.dec_hsz * 3], dtype=tf.float32, initializer=self.uniform_init)

            self.w_out = tf.nn.tanh(
                tf.transpose(tf.matmul(self.tgt_emb_matrix, self.w_proj), perm=[1, 0]))

            self.b_out = tf.Variable(
                tf.constant(0.1, shape=[args.tgt_vs]), name='b_out')

            self.w_base_proj = tf.get_variable("w_base", [args.emb_dim,
                                                          args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init)

            self.w_base_out = tf.nn.tanh(
                tf.transpose(tf.matmul(self.tgt_emb_matrix, self.w_base_proj), perm=[1, 0]))

            self.b_base_out = tf.Variable(tf.constant(
                0.1, shape=[args.tgt_vs]), name='b_base')

    def build(self):
        args = self.args

        with tf.variable_scope("embbeding"):
            self.src_emb = tf.nn.embedding_lookup(
                self.src_emb_matrix, self.doc)

        with tf.variable_scope('encode'):
            self._encode()

        with tf.variable_scope('decode'):
            ml_props = self._teacher_forcing()
            rl_props, s_words = self._sample(False)
            b_words = self._sample(True)

        with tf.variable_scope('loss'):
            ml_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=ml_props, labels=self.label)
            ml_loss = tf.reduce_mean(ml_loss)

            rl_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=rl_props, labels=self.label)
            rl_loss = tf.reduce_mean(rl_loss)

            self.mix_loss = args.gamma * rl_loss + (1. - args.gamma) * ml_loss

            p_reward, bl_reward = self._reward(s_words, b_words)
            reward = p_reward - bl_reward
            self.gradients = self.optimizer.compute_gradients(
                self.mix_loss)
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (tf.clip_by_norm(
                        grad * reward, args.clip_norm), var)

            for grad, var in self.gradients:
                tf.summary.histogram(var.name, var)
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradients', grad)

            tf.summary.scalar("p_reward", p_reward)
            tf.summary.scalar("bl_reward", bl_reward)
            tf.summary.scalar("reward", reward)
            tf.summary.scalar('ml_loss', ml_loss)
            tf.summary.scalar('rl_loss', rl_loss)
            tf.summary.scalar('mix_loss', self.mix_loss)

        self.train_op = self.optimizer.apply_gradients(
            self.gradients, global_step=self.global_step)

        self.words = s_words
        self.merged = tf.summary.merge_all()

    def ml_train_step(self, batch, sess):
        feed_dict = {
            self.doc: batch.data,
            self.label: batch.label,
            self.label_ids: batch.label_ids,
            self.dropout: self.args.dropout
        }

        _, step, merged = sess.run([
            self.train_op, self.global_step, self.merged], feed_dict)

        return merged, step

    def mix_train_step(self, batch, sess):
        feed_dict = {
            self.doc: batch.data,
            self.label: batch.label,
            self.label_ids: batch.label_ids,
            self.dropout: self.args.dropout
        }

        _, step, merged = sess.run([
            self.train_op, self.global_step, self.merged], feed_dict)

        return merged, step

    def eval_step(self, batch, sess):
        feed_dict = {
            self.doc: batch.data,
            self.label: batch.label,
            self.label_ids: batch.label_ids,
            self.dropout: 1.
        }

        _, loss = sess.run([self.global_step, self.mix_loss], feed_dict)

        return loss

    def generate(self, batch, sess):
        feed_dict = {
            self.doc: batch.data,
            self.label: batch.label,
            self.label_ids: batch.label_ids,
            self.dropout: 1.
        }

        _, words = sess.run([self.global_step, self.words], feed_dict)

        return words

    def _encode(self):
        args = self.args

        fw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(args.enc_hsz)
        bw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(args.enc_hsz)

        fw_hidden = fw_rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
        bw_hidden = bw_rnn_cell.zero_state(self.batch_size, dtype=tf.float32)

        encodes, (fw_states, bw_states) = tf.nn.bidirectional_dynamic_rnn(
            fw_rnn_cell, bw_rnn_cell, self.src_emb,
            initial_state_fw=fw_hidden,
            initial_state_bw=bw_hidden,
            dtype=tf.float32)

        self.encode_hidden = tf.concat((fw_states.h, bw_states.h), -1)
        self.enc_docs = tf.concat(encodes, -1)

        if args.dropout != 1.:
            self.enc_docs = tf.nn.dropout(self.enc_docs, args.dropout)
            self.encode_hidden = tf.nn.dropout(
                self.encode_hidden, args.dropout)

        self.encode_states = tf.contrib.rnn.LSTMStateTuple(h=self.encode_hidden,
                                                           c=self.rnn_cell.zero_state(self.batch_size, dtype=tf.float32).c)

    def _teacher_forcing(self):
        """
        Return:
            props, size - [bsz*time*feats]
            words, size - [bsz*time]
        """
        word = tf.constant(BOS, shape=[self.batch_size, 1], dtype=tf.int64)
        tgt_emb = tf.nn.embedding_lookup(self.tgt_emb_matrix, word)

        outputs, args, pre_enc_hiddens = [], self.args, []

        dec_state = self.encode_states

        for step in range(args.l_max_len):
            dec_out, dec_state = tf.nn.dynamic_rnn(
                cell=self.rnn_cell,
                inputs=tgt_emb,
                initial_state=dec_state,
                dtype=tf.float32)

            if args.dropout != 1.:
                dec_out = tf.nn.dropout(dec_out, args.dropout)

            dec_hidden = dec_state.h

            enc_c_t = intra_temp_atten(self.w_enc,
                                       self.enc_docs, dec_hidden, args.d_max_len, pre_enc_hiddens)

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
            word = tf.argmax(props, -1)  # bsz*1

            tgt_emb = tf.nn.embedding_lookup(
                self.tgt_emb_matrix, tf.expand_dims(self.label_ids[:, step], 1))

            outputs.append(props)

        return tf.nn.log_softmax(tf.concat(outputs, 1))

    def _sample(self, max_props=True):
        word = tf.constant(BOS, shape=[self.batch_size, 1], dtype=tf.int64)
        tgt_emb = tf.nn.embedding_lookup(self.tgt_emb_matrix, word)

        outputs, words = [], []
        args, pre_enc_hiddens = self.args, []

        dec_state = self.encode_states

        for step in range(args.l_max_len):
            dec_out, dec_state = tf.nn.dynamic_rnn(
                cell=self.rnn_cell,
                inputs=tgt_emb,
                initial_state=dec_state,
                dtype=tf.float32)

            if args.dropout != 1.:
                dec_out = tf.nn.dropout(dec_out, args.dropout)

            dec_hidden = dec_state.h

            enc_c_t = intra_temp_atten(self.w_enc,
                                       self.enc_docs, dec_hidden, args.d_max_len, pre_enc_hiddens)

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

            words.append(word)
            tgt_emb = tf.nn.embedding_lookup(self.tgt_emb_matrix, word)

            outputs.append(tf.expand_dims(props, 1))

        if max_props:
            return tf.concat(words, 1)
        else:
            return tf.concat(outputs, 1), tf.concat(words, 1)

    def _reward(self, s_words, b_words):
        bl_reward = tf.py_func(rouge_l, [b_words, self.label_ids], tf.float32)
        p_reward = tf.py_func(rouge_l, [s_words, self.label_ids], tf.float32)

        return tf.reduce_mean(p_reward), tf.reduce_mean(bl_reward)
