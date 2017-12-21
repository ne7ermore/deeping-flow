import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, args):

        self.input = tf.placeholder(tf.int64, [args.batch_size, args.max_len], name="sample")
        self.label = tf.placeholder(tf.int64, [args.batch_size, args.label_size], name="label")
        self.dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #euclidean
        l2_loss = tf.constant(0.)        

        #embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            word_emb = tf.Variable(tf.random_uniform([
                    args.vocab_size, args.emb_dim], -1.0, 1.0))

            emb_encode = tf.nn.embedding_lookup(word_emb, self.input)
        
        #lstm layer
        with tf.name_scope("lstm"):
            rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in args.hidden_sizes]

            rnn_cells = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
            rnn_cells = tf.nn.rnn_cell.DropoutWrapper(rnn_cells,
                                        output_keep_prob=self.dropout)

            hidden = rnn_cells.zero_state(args.batch_size, dtype=tf.float32)
            
            lstm_encode, _ = tf.nn.dynamic_rnn(
                                    cell=rnn_cells,
                                    inputs=emb_encode,
                                    initial_state=hidden,
                                    dtype=tf.float32)        
        last_hsz = args.hidden_sizes[-1]
        lstm_encode = tf.expand_dims(lstm_encode, -1)

        #cnn layer
        cnn_encodes = []
        for i, filter_size in enumerate(args.filter_sizes):
            filter_shape = [filter_size, last_hsz, 1, args.num_filters]
            with tf.name_scope('cnn_{}'.format(i)):
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")     
                b = tf.Variable(tf.constant(0., shape=[args.num_filters]), name="b")
                cnn_encode = tf.nn.conv2d(
                    lstm_encode, w,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='cnn')

                cnn_encode = tf.nn.relu(tf.nn.bias_add(cnn_encode, b), name='relu')
                pool = tf.nn.max_pool(
                    cnn_encode,
                    ksize=[1, args.max_len-filter_size+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                cnn_encodes.append(pool)

        num_filters_total = args.num_filters*len(args.filter_sizes)
        cnn_encode = tf.concat(cnn_encodes, -1)
        cnn_encode = tf.reshape(cnn_encode, shape=[-1, num_filters_total])
        cnn_encode = tf.nn.dropout(cnn_encode, self.dropout)

        # prediction layer
        with tf.name_scope('predict'):
            w = tf.get_variable('w',
                shape=[num_filters_total, args.label_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[args.label_size]), name='b')
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)
            scores = tf.nn.xw_plus_b(cnn_encode, w, b, name='scores')
            self.predictions = tf.argmax(scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                                logits=scores, labels=self.label)
            self.loss = tf.reduce_mean(losses) + args.l_2 * l2_loss

        with tf.name_scope('corrects'):
            corrects = tf.equal(
                    self.predictions, tf.argmax(self.label, 1))
            self.corrects = tf.reduce_sum(
                    tf.cast(corrects, "float"), name="corrects")