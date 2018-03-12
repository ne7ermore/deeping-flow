import tensorflow as tf
import numpy as np

import random

class Model(object):
    def __init__(self, args, pre_w2v=None):
        self.args = args
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        #euclidean
        self.l2_loss = tf.constant(0.)

        self.uniform_init = tf.random_uniform_initializer(
                    -args.u_scope, args.u_scope, seed=args.seed)

        self.init_placeholders()
        self.init_graph()

        optimizer = tf.train.AdamOptimizer(
                args.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def init_placeholders(self):
        args = self.args

        self.q_in = tf.placeholder(tf.int64, [None, args.max_len], name="question")
        self.a_in = tf.placeholder(tf.int64, [None, args.max_len], name="answer")
        self.label = tf.placeholder(tf.float32, [None, 2], name="label")
        self.dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def init_graph(self):
        args = self.args

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if args.use_w2v is not None and args.use_w2v:
                word_emb = tf.Variable(tf.constant(args.w2v, dtype=tf.float32))
            else:
                word_emb = tf.Variable(tf.random_uniform([
                        args.vocab_size, args.emb_dim], -1.0, 1.0))

            q_ebd = tf.nn.embedding_lookup(word_emb, self.q_in)
            a_ebd = tf.nn.embedding_lookup(word_emb, self.a_in)

            self.q_ebd = tf.reshape(q_ebd, shape=[-1, args.emb_dim*args.max_len])
            self.a_ebd = tf.reshape(a_ebd, shape=[-1, args.emb_dim*args.max_len])

        # Preprocessing layer
        with tf.name_scope("preprocess_layer"):
            w_u = tf.get_variable(
                "w_u",
                shape=[args.emb_dim*args.max_len, args.attn_dim*args.max_len],
                initializer=tf.contrib.layers.xavier_initializer())
            b_u = tf.Variable(tf.constant(0.1,
                    shape=[args.attn_dim*args.max_len]), name="b_u")

            w_i = tf.get_variable(
                "w_i",
                shape=[args.emb_dim*args.max_len, args.attn_dim*args.max_len],
                initializer=tf.contrib.layers.xavier_initializer())
            b_i = tf.Variable(tf.constant(0.1,
                    shape=[args.attn_dim*args.max_len]), name="b_i")

            def make_ppc(x):
                sigmoid = tf.nn.sigmoid(tf.nn.xw_plus_b(x, w_i, b_i))
                tanh = tf.nn.tanh(tf.nn.xw_plus_b(x, w_u, b_u))
                return sigmoid * tanh

            self.q_hat = make_ppc(self.q_ebd)
            self.a_hat = make_ppc(self.a_ebd)

        # Self attention layer
        with tf.name_scope("self_atten"):
            w_g = tf.get_variable(
                "w_g",
                shape=[args.attn_dim*args.max_len, args.attn_dim*args.max_len],
                initializer=self.uniform_init)
            b_g = tf.Variable(tf.constant(0.1,
                shape=[args.attn_dim*args.max_len]), name="b_g")

            q_t = tf.transpose(tf.nn.xw_plus_b(self.q_hat, w_g, b_g))
            G = tf.nn.softmax(tf.matmul(q_t, self.a_hat))
            self.h = tf.matmul(self.q_hat, G) # bsz*[len*attn_dim]

        # Comparison: Subtraction + Multiplication + nn
        with tf.name_scope("comparison"):
            w_c = tf.get_variable(
                "w_c",
                shape=[args.attn_dim*args.max_len*2, args.attn_dim*args.max_len],
                initializer=self.uniform_init)
            b_c = tf.Variable(tf.constant(0.1,
                shape=[args.attn_dim*args.max_len]), name="b_c")

            sub = tf.square(self.a_hat - self.h)
            mult = self.a_hat * self.h
            temp = tf.concat([sub, mult], axis=-1)

            comp = tf.nn.relu(tf.nn.xw_plus_b(temp, w_c, b_c))
            self.comp = tf.reshape(comp, shape=[-1, args.max_len, args.attn_dim]) # bsz*len*attn_dim

        # add channel
        self.comp_pre = tf.expand_dims(self.comp, -1)

        # aggregation
        cnn_encodes = []
        for index, filter_size in enumerate(args.filter_sizes):
            filter_shape = [filter_size, args.attn_dim, 1, args.num_filters]

            with tf.name_scope("cnn_{}".format(index)):
                cnn_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
                cnn_b = tf.Variable(tf.constant(0.1, shape=[args.num_filters]))

                conv = tf.nn.conv2d(self.comp_pre,
                                    cnn_w,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                encode = tf.nn.relu(tf.nn.bias_add(conv, cnn_b))
                encode = tf.nn.max_pool(encode,
                                        ksize=[1, args.max_len-filter_size+1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID')
                cnn_encodes.append(encode)

        filters_dim = len(args.filter_sizes) * args.num_filters

        self.cnn = tf.reshape(tf.concat(cnn_encodes, -1), [-1, filters_dim])
        self.cnn = tf.nn.dropout(self.cnn, self.dropout)

        with tf.name_scope("projection"):
            w_p = tf.get_variable(
                "w_p",
                shape=[filters_dim, 2],
                initializer=self.uniform_init)
            b_p = tf.Variable(tf.constant(0.1, shape=[2]), name="b_p")

            self.l2_loss += tf.nn.l2_loss(w_p)
            self.l2_loss += tf.nn.l2_loss(b_p)

            scores = tf.nn.xw_plus_b(self.cnn, w_p, b_p)
            self.predict = tf.argmax(scores, 1)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                            logits=scores, labels=self.label)
            self.loss = tf.reduce_mean(losses) + args.l_2 * self.l2_loss
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("corrects"):
            corrects = tf.equal(self.predict, tf.argmax(self.label, 1))
            self.corrects = tf.reduce_sum(tf.cast(corrects, "float"))
            tf.summary.scalar('corrects', self.corrects)

        self.merged = tf.summary.merge_all()

    def train_step(self, src, tgt, label, sess):
        feed_dict = {
            self.q_in: src,
            self.a_in: tgt,
            self.label: label,
            self.dropout: self.args.dropout
        }

        _, step, loss, corrects, merged = sess.run([self.train_op,
                self.global_step, self.loss, self.corrects, self.merged], feed_dict)

        return merged, step

    def eval_step(self, src, tgt, label, sess):
        feed_dict = {
            self.q_in: src,
            self.a_in: tgt,
            self.label: label,
            self.dropout: 1.
        }
        _, loss, corrects = sess.run([self.global_step, self.loss, self.corrects], feed_dict)
        return loss, corrects


