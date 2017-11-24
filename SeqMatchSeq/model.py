import tensorflow as tf
import numpy as np

import random

class Model(object):
    def __init__(self, args, pre_w2v=None):
        self._args = args

        # placeholder: Q, A, label, dropout
        self.q_in = tf.placeholder(tf.int64, [None, args.max_len], name="question")
        self.a_in = tf.placeholder(tf.int64, [None, args.max_len], name="answer")
        self.label = tf.placeholder(tf.float32, [None, 2], name="label")
        self.dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Euclidean
        l2_loss = tf.constant(0.)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if args.use_w2v is not None:
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

            l2_loss += tf.nn.l2_loss(w_u)
            l2_loss += tf.nn.l2_loss(b_u)
            l2_loss += tf.nn.l2_loss(w_i)
            l2_loss += tf.nn.l2_loss(b_i)

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
                initializer=tf.contrib.layers.xavier_initializer())
            b_g = tf.Variable(tf.constant(0.1,
                shape=[args.attn_dim*args.max_len]), name="b_g")

            q_t = tf.transpose(tf.nn.xw_plus_b(self.q_hat, w_g, b_g))
            G = tf.nn.softmax(tf.matmul(q_t, self.a_hat))
            self.h = tf.matmul(self.q_hat, G) # bsz*[len*attn_dim]

            l2_loss += tf.nn.l2_loss(w_g)
            l2_loss += tf.nn.l2_loss(b_g)

        # Comparison: Subtraction + Multiplication + nn
        with tf.name_scope("comparison"):
            w_c = tf.get_variable(
                "w_c",
                shape=[args.attn_dim*args.max_len*2, args.attn_dim*args.max_len],
                initializer=tf.contrib.layers.xavier_initializer())
            b_c = tf.Variable(tf.constant(0.1,
                shape=[args.attn_dim*args.max_len]), name="b_c")

            l2_loss += tf.nn.l2_loss(w_c)
            l2_loss += tf.nn.l2_loss(b_c)

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

                l2_loss += tf.nn.l2_loss(cnn_w)
                l2_loss += tf.nn.l2_loss(cnn_b)

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
                initializer=tf.contrib.layers.xavier_initializer())
            b_p = tf.Variable(tf.constant(0.1, shape=[2]), name="b_p")

            l2_loss += tf.nn.l2_loss(w_p)
            l2_loss += tf.nn.l2_loss(b_p)

            self.project = tf.nn.relu(tf.nn.xw_plus_b(self.cnn, w_p, b_p))
            self.project = tf.nn.dropout(self.project, self.dropout)

            self.predict = tf.argmax(self.project, 1)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                            logits=self.project, labels=self.label)
            self.loss = tf.reduce_mean(losses) + args.l_2 * l2_loss

        with tf.name_scope("acc"):
            corrects = tf.equal(self.predict, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(corrects, "float"))
