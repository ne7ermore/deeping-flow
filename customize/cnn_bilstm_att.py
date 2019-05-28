import tensorflow as tf
import numpy as np


class Model(object):
    def __init__(self, args):
        self.args = args
        self.global_step = tf.train.get_or_create_global_step()
        self.uniform_init = tf.random_uniform_initializer(
            -.1, .1, seed=args.seed)

        self.init_placeholders()
        self.init_graph()

    def init_placeholders(self):
        args = self.args

        self.doc = tf.placeholder(
            tf.int64, [None, args.max_d_len], name="doc")
        self.label = tf.placeholder(tf.int64, [None], name="label")
        self.dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def init_graph(self):
        args = self.args
        bsz = tf.shape(self.doc)[0]

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            if args.use_w2v:
                word_emb = tf.Variable(tf.constant(args.w2v, dtype=tf.float32))
            else:
                word_emb = tf.Variable(tf.random_uniform([
                    args.vocab_size, args.emb_dim], -.1, .1))

            word_emb = tf.concat(
                (tf.zeros(shape=[1, args.emb_dim]), word_emb[1:, :]), 0)

            d_emb = tf.nn.embedding_lookup(word_emb, self.doc)

            if self.dropout != 1.:
                d_emb = tf.nn.dropout(d_emb, self.dropout)

        with tf.name_scope("bilstm"):
            rnn_layers = [tf.nn.rnn_cell.LSTMCell(
                size) for size in args.rnn_sizes]
            fw_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
            bw_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

            fw_hidden = fw_rnn_cells.zero_state(bsz, dtype=tf.float32)
            bw_hidden = bw_rnn_cells.zero_state(bsz, dtype=tf.float32)

            encodes, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_rnn_cells,
                bw_rnn_cells,
                d_emb,
                initial_state_fw=fw_hidden,
                initial_state_bw=bw_hidden,
                dtype=tf.float32
            )

            lstm_encode = tf.concat(encodes, -1)
            if self.dropout != 1.:
                lstm_encode = tf.nn.dropout(lstm_encode, self.dropout)

        last_hsz = args.rnn_sizes[-1] * 2

        with tf.name_scope("attention"):
            atta = tf.keras.layers.Dense(last_hsz)

            u_s = tf.get_variable('u_s',
                                  shape=[last_hsz, last_hsz],
                                  initializer=self.uniform_init)

            encs = tf.transpose(lstm_encode, perm=[1, 0, 2])

            _encs = tf.reshape(encs, [-1, last_hsz])  # [t*b, f]
            u_is = tf.nn.tanh(atta(_encs))

            u_exps = tf.exp(tf.matmul(u_is, u_s))
            u_exps = tf.reshape(
                u_exps, [args.max_d_len, bsz, last_hsz])

            u_exp_sum = tf.reduce_sum(u_exps, axis=0, keepdims=True)
            u_exp_sum = tf.tile(u_exp_sum, [args.max_d_len, 1, 1])

            a_is = tf.div(u_exps, u_exp_sum)
            att_encode = tf.transpose(tf.multiply(encs, a_is), perm=[1, 0, 2])

            if self.dropout != 1.:
                att_encode = tf.nn.dropout(att_encode, self.dropout)

        att_encode = tf.expand_dims(att_encode, -1)
        cnn_encodes = []
        for i, filter_size in enumerate(args.doc_filter_sizes):
            filter_shape = [filter_size, last_hsz, 1, args.num_filters]
            with tf.name_scope('cnn_{}'.format(i)):
                w = tf.Variable(tf.random_uniform(
                    filter_shape, -.1, .1), name="W")
                b = tf.Variable(tf.constant(
                    0., shape=[args.num_filters]), name="b")
                cnn_encode = tf.nn.conv2d(
                    att_encode, w,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='cnn')

                cnn_encode = tf.nn.relu(
                    tf.nn.bias_add(cnn_encode, b), name='relu')
                pool = tf.nn.max_pool(
                    cnn_encode,
                    ksize=[1, args.max_d_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                cnn_encodes.append(pool)

        num_filters_total = args.num_filters * len(args.doc_filter_sizes)
        cnn_encode = tf.concat(cnn_encodes, -1)
        cnn_encode = tf.reshape(cnn_encode, shape=[-1, num_filters_total])
        if self.dropout != 1.:
            cnn_encode = tf.nn.dropout(cnn_encode, self.dropout)

        # prediction layer
        with tf.name_scope('predict'):
            linear = tf.keras.layers.Dense(args.label_size)
            scores = linear(cnn_encode)
            self.predictions = tf.argmax(scores, 1, name='predictions')
            self.softmax = tf.nn.softmax(scores, -1, name="softmax")

            one_hot = tf.one_hot(self.label, args.label_size, dtype=tf.int64)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=scores, labels=one_hot)
            loss = tf.reduce_mean(loss)

            self.train_op = tf.train.AdamOptimizer(
                args.lr, beta1=0.9, beta2=0.98, epsilon=1e-8).minimize(loss, global_step=self.global_step)

            corrects = tf.equal(self.predictions, self.label)
            self.corrects = tf.reduce_sum(
                tf.cast(corrects, "float"), name="corrects")

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('corrects', self.corrects)
            self.merged = tf.summary.merge_all()

    def train_step(self, batch, sess):
        feed_dict = {
            self.doc: batch.docs,
            self.label: batch.label,
            self.dropout: self.args.dropout
        }

        _, step, merged = sess.run(
            [self.train_op, self.global_step, self.merged], feed_dict)

        return step, merged

    def eval_step(self, batch, sess):
        feed_dict = {
            self.doc: batch.docs,
            self.label: batch.label,
            self.dropout: 1.
        }
        step, corrects, merged = sess.run(
            [self.global_step, self.corrects, self.merged], feed_dict)
        return corrects, merged, step

    def test_step(self, batch, sess):
        feed_dict = {
            self.doc: batch.docs,
            self.label: batch.label,
            self.dropout: 1.
        }
        pred_labels, pred_props = sess.run(
            [self.predictions, self.softmax], feed_dict)
        return pred_labels, pred_props
