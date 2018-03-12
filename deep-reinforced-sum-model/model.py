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

        self.init_placeholders()
        self.init_vars()
        self.build()

        ml_optimizer = tf.train.AdamOptimizer(
                self.args.ml_lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        ml_gvs = ml_optimizer.compute_gradients(self.ml_loss)
        ml_capped_gvs = [(tf.clip_by_value(grad,
                -args.clip_norm, args.clip_norm), var) for grad, var in ml_gvs if grad is not None]
        self.ml_train_op = ml_optimizer.apply_gradients(ml_capped_gvs, global_step=self.global_step)

        mix_optimizer = tf.train.AdamOptimizer(
                self.args.mix_lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        mix_gvs = mix_optimizer.compute_gradients(self.mix_loss)
        mix_capped_gvs = [(tf.clip_by_value(grad,
                -args.clip_norm, args.clip_norm), var) for grad, var in mix_gvs if grad is not None]
        self.mix_train_op = mix_optimizer.apply_gradients(mix_capped_gvs, global_step=self.global_step)

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
            self.src_emb_matrix = embedding("src_emb_matrix", args.src_vs, args.emb_dim, self.uniform_init)
            self.tgt_emb_matrix = embedding("tgt_emb_matrix", args.tgt_vs, args.emb_dim, self.uniform_init)

            self.w_enc = tf.get_variable("w_enc_atten", [args.dec_hsz,
                args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init)

            self.w_dec = tf.get_variable("w_dec_atten", [args.dec_hsz,
                args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init)

            self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(args.dec_hsz)

            self.w_proj = tf.get_variable("w_proj", [args.emb_dim,
                args.dec_hsz*3], dtype=tf.float32, initializer=self.uniform_init)

            self.w_out = tf.nn.tanh(
                tf.transpose(tf.matmul(self.tgt_emb_matrix, self.w_proj), perm=[1, 0]))

            self.b_out = tf.Variable(
                tf.constant(0.1, shape=[args.tgt_vs]), name='b_out')

            self.w_base_proj = tf.get_variable("w_base", [args.emb_dim,
                args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init)

            self.w_base_out = tf.nn.tanh(
                tf.transpose(tf.matmul(self.tgt_emb_matrix, self.w_base_proj), perm=[1, 0]))

            self.b_base_out = tf.Variable(tf.constant(0.1, shape=[args.tgt_vs]), name='b_base')

    def build(self):
        args = self.args

        with tf.variable_scope("embbeding"):
            self.src_emb = tf.nn.embedding_lookup(self.src_emb_matrix, self.doc)

        with tf.variable_scope('encode'):
            self._encode()

        with tf.variable_scope('decode'):
            ml_props, ml_words, ml_word_props = self._teacher_forcing()
            base_words = self._baseline(ml_words)

        with tf.variable_scope('loss'):
            ml_loss = tf.nn.softmax_cross_entropy_with_logits(
                                logits=ml_props, labels=self.label)
            rl_loss = self._reinforced(ml_words[:, 1:], base_words, ml_word_props)

            self.ml_loss = tf.reduce_mean(ml_loss)
            self.mix_loss = args.gamma * rl_loss + (1. - args.gamma) * self.ml_loss

            tf.summary.scalar('ml_loss', tf.reduce_mean(ml_loss))
            tf.summary.scalar('rl_loss', tf.reduce_mean(rl_loss))
            tf.summary.scalar('mix_loss', self.mix_loss)

        self.words = ml_words[:, 1:]
        self.merged = tf.summary.merge_all()

    def ml_train_step(self, batch, sess):
        feed_dict = {
            self.doc: batch.data,
            self.label: batch.label,
            self.label_ids: batch.label_ids,
            self.dropout: self.args.dropout
        }

        _, step, merged = sess.run([
                self.ml_train_op, self.global_step, self.merged], feed_dict)

        return merged, step

    def mix_train_step(self, batch, sess):
        feed_dict = {
            self.doc: batch.data,
            self.label: batch.label,
            self.label_ids: batch.label_ids,
            self.dropout: self.args.dropout
        }

        _, step, merged = sess.run([
                self.mix_train_op, self.global_step, self.merged], feed_dict)

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
            self.encode_hidden = tf.nn.dropout(self.encode_hidden, args.dropout)

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

        outputs, words, word_props, args, pre_enc_hiddens = [], [word], [], self.args, []

        dec_state = self.encode_states

        for step in range(args.l_max_len):
            dec_out, dec_state = tf.nn.dynamic_rnn(
                                    cell=self.rnn_cell,
                                    inputs=tgt_emb,
                                    initial_state=dec_state,
                                    dtype=tf.float32)
            dec_hidden = dec_state.h

            if args.dropout != 1.:
                dec_out = tf.nn.dropout(dec_out, args.dropout)
                dec_hidden = tf.nn.dropout(dec_hidden, args.dropout)

            enc_c_t = intra_temp_atten(self.w_enc,
                    self.enc_docs, dec_hidden, args.d_max_len, pre_enc_hiddens)

            if step == 0:
                # We set dec_c to a vector of zeros since the generated sequence is empty on the first decoding step
                dec_c_t = tf.constant(0., shape=[self.batch_size, args.dec_hsz])
            else:
                dec_c_t = intra_decoder_atten(
                        self.w_dec, dec_hidden, dec_out)

            out = tf.concat((dec_hidden, enc_c_t, dec_c_t), -1)

            props = tf.expand_dims(tf.nn.xw_plus_b(out, self.w_out, self.b_out), 1) # bsz*1*f
            word = tf.argmax(props, -1) # bsz*1
            word_prop = tf.reduce_max(props, -1) # bsz*1

            tgt_emb = tf.nn.embedding_lookup(
                self.tgt_emb_matrix, tf.expand_dims(self.label_ids[:, step], 1))

            outputs.append(props)
            words.append(word)
            word_props.append(word_prop)

        return tf.nn.log_softmax(tf.concat(outputs, 1)), tf.concat(words, -1), tf.concat(word_props, -1)

    def _baseline(self, words):
        args = self.args

        tgt_emb = tf.nn.embedding_lookup(self.tgt_emb_matrix, words[:, :-1])

        dec_out, _ = tf.nn.dynamic_rnn(
                        cell=self.rnn_cell,
                        inputs=tgt_emb,
                        initial_state=self.encode_states,
                        dtype=tf.float32)

        if args.dropout != 1.:
            dec_out = tf.nn.dropout(dec_out, args.dropout)

        dec_out = tf.reshape(dec_out, [-1, args.dec_hsz])
        props = tf.nn.xw_plus_b(dec_out, self.w_base_out, self.b_base_out)
        words = tf.reshape(tf.argmax(props, -1), [args.batch_size, args.l_max_len])

        return words

    def _reinforced(self, ml_words, base_words, ml_word_props):
        def mask_score(words, scores):
            pad = tf.zeros([args.batch_size, args.l_max_len], dtype=tf.int64)
            mask = tf.cast(tf.greater(words, pad), dtype=tf.float32)

            return tf.multiply(scores, mask), mask

        args = self.args

        bl_reward = tf.py_func(rouge_l, [base_words, self.label_ids], tf.float32)
        p_reward = tf.py_func(rouge_l, [ml_words, self.label_ids], tf.float32)

        masked_score, mask = mask_score(ml_words, bl_reward-p_reward)
        rl_loss = tf.reduce_sum(ml_word_props*masked_score) / tf.reduce_sum(mask)

        return rl_loss
