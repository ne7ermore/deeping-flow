import tensorflow as tf
import numpy as np

from const import BOS, PAD


def multi_view_att(ori_memory_t, att_w, dec_hidden, *args):
    bsz, max_len, rnn_hsz = args

    dec_hidden_t = tf.transpose(dec_hidden, perm=[1, 0])
    dec_hidden_t = att_w(dec_hidden_t)

    flat_omt = tf.reshape(ori_memory_t, [-1, rnn_hsz])
    beta_is = tf.exp(tf.tanh(tf.matmul(flat_omt, dec_hidden_t)))
    beta_is = tf.reshape(beta_is, [max_len, bsz, rnn_hsz])

    beta_i_sum = tf.reduce_sum(beta_is, axis=0, keepdims=True)
    beta_i_sum = tf.tile(beta_i_sum, [max_len, 1, 1])
    beta_is = tf.div(beta_is, beta_i_sum)

    return tf.reduce_sum(beta_is * ori_memory_t, axis=0)


def pad_mask(seq, index, shape, dtype=tf.float32):
    indexs = tf.constant(index, shape=shape, dtype=tf.int64)
    return tf.cast(tf.not_equal(seq, indexs), dtype=dtype)


def gather_index(props, tgt, prev):
    tgt = tf.reshape(tgt, [-1, 1])
    index = tf.concat((prev, tgt), -1)
    return tf.gather_nd(props, index)


class Model(object):
    def __init__(self, args):
        self.args = args
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('init_variables'):
            self.original = tf.placeholder(
                tf.int64, [args.batch_size, args.max_ori_len], name="original")
            self.summary = tf.placeholder(
                tf.int64, [args.batch_size, args.max_sum_len], name="summary")
            self.label = tf.placeholder(
                tf.int64, [args.batch_size], name="label")
            self.dropout = tf.placeholder(tf.float32, name="dropout")
            self.lr = tf.placeholder(tf.float32, name="lr")

        with tf.variable_scope('init_layers'):
            self.emb = tf.keras.layers.Embedding(
                args.dict_size, args.emb_dim)
            self.fw_rnn_cell = tf.nn.rnn_cell.GRUCell(args.rnn_hsz)
            self.bw_rnn_cell = tf.nn.rnn_cell.GRUCell(args.rnn_hsz)
            self.dec_cell = tf.nn.rnn_cell.GRUCell(args.rnn_hsz)
            self.summ_att_w = tf.keras.layers.Dense(
                args.rnn_hsz, use_bias=False)
            self.cls_att_w = tf.keras.layers.Dense(
                args.rnn_hsz, use_bias=False)
            self.summ_gen = tf.keras.layers.Dense(args.dict_size)
            self.cls_pred = tf.keras.layers.Dense(args.label_size)

        self.init_graph()

    def init_graph(self):
        args = self.args
        bsz = args.batch_size

        with tf.variable_scope("encode"):
            ori_emb = self.emb(self.original)

            encodes, (fw_hidden, bw_hidden) = tf.nn.bidirectional_dynamic_rnn(
                self.fw_rnn_cell,
                self.bw_rnn_cell,
                ori_emb,
                dtype=tf.float32)

            ori_hidden = fw_hidden + bw_hidden
            ori_memory = encodes[0] + encodes[1]

            if self.dropout != 1.:
                ori_hidden = tf.nn.dropout(ori_hidden, self.dropout)
                ori_memory = tf.nn.dropout(ori_memory, self.dropout)

            ori_memory_t = tf.transpose(ori_memory, perm=[1, 0, 2])

        with tf.variable_scope("decode"):
            word = tf.constant(BOS, shape=[bsz, 1], dtype=tf.int64)
            v_ts, summ_props, summ_emb = [], [], self.emb(word)

            dec_hidden = ori_hidden
            for step in range(args.max_sum_len):
                _, dec_hidden = tf.nn.dynamic_rnn(
                    cell=self.dec_cell,
                    inputs=summ_emb,
                    initial_state=dec_hidden,
                    dtype=tf.float32)

                if self.dropout != 1.:
                    dec_hidden = tf.nn.dropout(dec_hidden, self.dropout)

                v_c = multi_view_att(ori_memory_t,
                                     self.summ_att_w,
                                     dec_hidden,
                                     bsz,
                                     args.max_ori_len,
                                     args.rnn_hsz)
                v_t = multi_view_att(ori_memory_t,
                                     self.cls_att_w,
                                     dec_hidden,
                                     bsz,
                                     args.max_ori_len,
                                     args.rnn_hsz)

                props = tf.nn.log_softmax(self.summ_gen(v_c))
                word = tf.expand_dims(
                    tf.cast(tf.argmax(props, -1), tf.int64), -1)

                v_ts.append(tf.expand_dims(v_t, 1))
                summ_props.append(tf.expand_dims(props, 1))
                summ_emb = self.emb(word)

            summ_props = tf.concat(summ_props, 1)
            v_ts = tf.concat(v_ts, 1)
            if self.dropout != 1.:
                v_ts = tf.nn.dropout(v_ts, self.dropout)

        with tf.variable_scope("classifier"):
            r = tf.concat([v_ts, ori_memory], 1)
            r = tf.nn.relu(r)
            r = tf.reduce_mean(r, axis=-1)
            l_props = self.cls_pred(r)
            predictions = tf.argmax(l_props, 1)

        with tf.name_scope("loss"):
            prev = tf.expand_dims(tf.constant(
                np.arange(bsz * args.max_sum_len, dtype=np.int64)), -1)
            summ_props = tf.reshape(props, [-1, args.dict_size])
            tgt_props = tf.reshape(gather_index(summ_props, self.summary, prev), [
                bsz, args.max_sum_len])
            mask = pad_mask(self.summary, PAD, [bsz, args.max_sum_len])
            summ_loss = -tf.reduce_sum(tgt_props * mask) / tf.reduce_sum(mask)

            one_hot = tf.one_hot(self.label, args.label_size, dtype=tf.int64)
            cls_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=l_props, labels=one_hot)
            cls_loss = tf.reduce_mean(cls_loss)

            loss = summ_loss + args.lamda * cls_loss

            optimizer = tf.train.AdamOptimizer(self.lr)
            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (
                        tf.clip_by_norm(grad, args.clip_norm), var)
            self.train_op = optimizer.apply_gradients(
                gradients, global_step=self.global_step)

            corrects = tf.equal(predictions, self.label)
            self.corrects = tf.reduce_sum(
                tf.cast(corrects, "float"), name="corrects")

        with tf.name_scope("summary"):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('summ_loss', summ_loss)
            tf.summary.scalar('cls_loss', cls_loss)
            tf.summary.scalar('corrects', self.corrects)
            self.merged = tf.summary.merge_all()

    def train_step(self, batch, sess, dropout, lr):
        feed_dict = {
            self.original: batch.original,
            self.summary: batch.summary,
            self.label: batch.label,
            self.dropout: dropout,
            self.lr: lr
        }

        _, step, merged = sess.run(
            [self.train_op, self.global_step, self.merged], feed_dict)

        return step, merged

    def eval_step(self, batch, sess):
        feed_dict = {
            self.original: batch.original,
            self.summary: batch.summary,
            self.label: batch.label,
            self.dropout: 1.
        }

        _, corrects = sess.run(
            [self.global_step, self.corrects], feed_dict)

        return corrects
