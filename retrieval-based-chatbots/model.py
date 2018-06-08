import tensorflow as tf
import numpy as np


class Model(object):
    def __init__(self, args):
        self.args = args
        self.global_step = tf.train.get_or_create_global_step()

        with tf.name_scope('init_variables'):
            self.utterances = tf.placeholder(
                tf.int64, [args.batch_size, args.max_cont_len, args.max_utte_len], name="utterances")
            self.responses = tf.placeholder(
                tf.int64, [args.batch_size, args.max_utte_len], name="responses")
            self.labels = tf.placeholder(
                tf.int64, [args.batch_size], name="labels")
            self.dropout = tf.placeholder(tf.float32, name="dropout")

        with tf.name_scope('init_layers'):
            self.emb = tf.keras.layers.Embedding(
                args.dict_size, args.emb_dim)
            self.first_gru = tf.nn.rnn_cell.GRUCell(args.first_rnn_hsz)
            self.transform_A = tf.keras.layers.Dense(
                args.first_rnn_hsz, use_bias=False)
            self.cnn = tf.keras.layers.Conv2D(
                args.fillters,
                args.kernel_size,
                activation=tf.nn.relu)
            self.max_pool = tf.keras.layers.MaxPool2D(
                pool_size=args.kernel_size,
                strides=args.kernel_size,
            )
            self.match_vec = tf.keras.layers.Dense(
                args.match_vec_dim, activation=tf.nn.relu)
            self.second_gru = tf.nn.rnn_cell.GRUCell(args.second_rnn_hsz)
            self.pred = tf.keras.layers.Dense(2, activation=tf.nn.log_softmax)

        self.init_graph()

    def init_graph(self):
        args = self.args

        with tf.name_scope("utterance_response_matching"):
            resps_emb = self.emb(self.responses)
            resps_gru, _ = tf.nn.dynamic_rnn(
                cell=self.first_gru,
                inputs=resps_emb,
                dtype=tf.float32,
                scope="first_gru")
            if self.dropout != 1.:
                resps_gru = tf.nn.dropout(resps_gru, self.dropout)

            resps_emb_t = tf.transpose(resps_emb, perm=[0, 2, 1])
            resps_gru_t = tf.transpose(resps_gru, perm=[0, 2, 1])

            uttes = tf.unstack(self.utterances, axis=1)

            match_vecs = []
            for utte in uttes:
                utte_emb = self.emb(utte)
                mat_1 = tf.matmul(utte_emb, resps_emb_t)

                utte_rnn, _ = tf.nn.dynamic_rnn(
                    cell=self.first_gru,
                    inputs=utte_emb,
                    dtype=tf.float32,
                    scope="first_gru")
                if self.dropout != 1.:
                    utte_rnn = tf.nn.dropout(utte_rnn, self.dropout)

                mat_2 = tf.matmul(self.transform_A(utte_rnn), resps_gru_t)

                M = tf.stack([mat_1, mat_2], axis=3)
                conv_layer = self.cnn(M)
                pool_layer = self.max_pool(conv_layer)
                match_vec = self.match_vec(
                    tf.contrib.layers.flatten(pool_layer))
                match_vecs.append(match_vec)

        with tf.name_scope("matching_accumulation"):
            match_vecs = tf.stack(match_vecs, axis=1)
            if self.dropout != 1.:
                match_vecs = tf.nn.dropout(match_vecs, self.dropout)
            _, hidden = tf.nn.dynamic_rnn(
                cell=self.second_gru,
                inputs=match_vecs,
                dtype=tf.float32,
                scope="second_gru")
            if self.dropout != 1.:
                hidden = tf.nn.dropout(hidden, self.dropout)

        with tf.name_scope("matching_prediction"):
            props = self.pred(hidden)

        with tf.name_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=props, labels=self.labels)
            loss = tf.reduce_mean(loss)
            optimizer = tf.train.AdamOptimizer(args.lr)
            self.train_op = optimizer.minimize(
                loss, global_step=self.global_step)

        with tf.name_scope("predictions"):
            predictions = tf.argmax(props, 1)
            corrects = tf.equal(predictions, self.labels)
            self.corrects = tf.reduce_mean(
                tf.cast(corrects, "float"), name="corrects")

        with tf.name_scope("summary"):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('corrects', self.corrects)
            self.merged = tf.summary.merge_all()

    def train_step(self, batch, sess, dropout):
        feed_dict = {
            self.utterances: batch.utterances,
            self.responses: batch.responses,
            self.labels: batch.labels,
            self.dropout: dropout
        }

        _, step, merged = sess.run(
            [self.train_op, self.global_step, self.merged], feed_dict)

        return step, merged

    def eval_step(self, batch, sess):
        feed_dict = {
            self.utterances: batch.utterances,
            self.responses: batch.responses,
            self.labels: batch.labels,
            self.dropout: 1.
        }

        _, corrects = sess.run(
            [self.global_step, self.corrects], feed_dict)

        return corrects
