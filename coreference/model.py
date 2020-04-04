import tensorflow as tf
import numpy as np

from tqdm import tqdm

from layer import get_token_embeddings, position_wise, multi_head_attention, label_smoothing
import common
import const


class Transformer:
    def __init__(self, args):
        self.args = args
        self.tgt_max_len_int = 100

        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.initializer = tf.random_normal_initializer(
            stddev=const.INIT_RANGE)

        with tf.compat.v1.variable_scope('init_variables'):

            self.src_max_len = tf.compat.v1.placeholder(
                tf.int32, name="src_max_len")
            self.batch_size = tf.compat.v1.placeholder(
                tf.int32, name="batch_size")
            self.tgt_max_len = tf.compat.v1.placeholder(
                tf.int32, name="tgt_max_len")
            self.dropout_rate = tf.compat.v1.placeholder(
                tf.float32, name="dropout_keep_prob")

            self.src_tensor = tf.compat.v1.placeholder(
                tf.int64, [None, None], name="src_tensor")
            self.src_postion = tf.compat.v1.placeholder(
                tf.int64, [None, None], name="src_postion")
            self.turns_tensor = tf.compat.v1.placeholder(
                tf.int64, [None, None], name="turns_tensor")
            self.tgt_tensor = tf.compat.v1.placeholder(
                tf.int64, [None, None], name="tgt_tensor")
            self.tgt_postion = tf.compat.v1.placeholder(
                tf.int64, [None, None], name="tgt_postion")
            self.tgt_indexs_tensor = tf.compat.v1.placeholder(
                tf.int64, [None, None], name="tgt_indexs_tensor")

            self.pre_enc_output = tf.compat.v1.placeholder(
                tf.float32, [None, None, args.d_model], name="pre_enc_output")

            # distillation
            self.dist_encode = tf.compat.v1.placeholder(
                tf.float32, [None, None, args.dist_model], name="dist_encode")
            self.dist_decode = tf.compat.v1.placeholder(
                tf.float32, [None, None, args.dist_model], name="dist_decode")

        with tf.compat.v1.variable_scope('init_embedding'):
            self.turn_embedding = get_token_embeddings(
                args.turn_size, args.d_model, "turn_embedding")
            self.word_embedding = get_token_embeddings(
                args.vocab_size, args.d_model, "word_embedding")
            self.pos_embedding = get_token_embeddings(
                args.max_context_len, args.d_model, "pos_embedding", common.get_sinusoid_encoding_table)

        enc_output, non_pad_mask = self.encode(self.src_tensor,
                                               self.src_postion,
                                               self.turns_tensor,
                                               self.src_max_len)
        enc_output, enc_distill_loss = self.distillation(
            enc_output, self.dist_encode*non_pad_mask, "encode_distillation")
        enc_output *= non_pad_mask
        self.enc_output = tf.identity(enc_output, "enc_output")

        # train
        dec_output, non_pad_mask = self.decode(
            self.tgt_tensor, self.tgt_postion, self.tgt_max_len, self.src_tensor, enc_output)
        dec_output, dec_distill_loss = self.distillation(
            dec_output*non_pad_mask, self.dist_decode*non_pad_mask, "decode_distillation")
        dec_output *= non_pad_mask
        distributes, props = self.pointer_network(enc_output, dec_output)
        self.distributes = tf.identity(distributes, "distributes")

        # predict
        pre_dec_output, pre_non_pad_mask = self.decode(
            self.tgt_tensor, self.tgt_postion, self.tgt_max_len, self.src_tensor, self.pre_enc_output)
        pre_dec_output, _ = self.distillation(
            pre_dec_output*pre_non_pad_mask, self.dist_decode*pre_non_pad_mask, "decode_distillation")
        pre_dec_output *= pre_non_pad_mask
        pre_distributes, _ = self.pointer_network(
            self.pre_enc_output, pre_dec_output)
        self.pre_distributes = tf.identity(pre_distributes, "pre_distributes")

        entropy_loss = self._cross_entropy()

        self.distill_loss = args.dist_encode_rate*enc_distill_loss + \
            (1-args.dist_encode_rate)*dec_distill_loss
        self.loss = self.distill_loss * args.dist_rate + \
            entropy_loss * (1 - args.dist_rate)

        self.pre_train_op = tf.compat.v1.train.AdamOptimizer(
            args.pretrain_lr).minimize(self.distill_loss, global_step=self.global_step)
        self.train_op = tf.compat.v1.train.AdamOptimizer(
            args.learning_rate).minimize(self.loss, global_step=self.global_step)

    def encode(self, src_tensor, src_postion, turns_tensor, src_max_len):
        args = self.args

        with tf.compat.v1.variable_scope("encode", reuse=tf.compat.v1.AUTO_REUSE):
            # embedding
            enc_output = tf.nn.embedding_lookup(
                self.word_embedding, src_tensor)
            enc_output *= args.d_model**0.5

            enc_output += tf.nn.embedding_lookup(
                self.pos_embedding, src_postion)

            turn_enc_output = tf.nn.embedding_lookup(
                self.turn_embedding, turns_tensor)
            enc_output += turn_enc_output*(args.d_model**0.5)

            enc_output = tf.nn.dropout(enc_output, keep_prob=self.dropout_rate)

            # encode mask
            slf_attn_mask = common.get_attn_key_pad_mask(
                src_tensor, src_tensor, src_max_len)
            non_pad_mask = common.get_non_pad_mask(src_tensor)

            # encode
            for i in range(args.enc_stack_layers):
                with tf.compat.v1.variable_scope("num_blocks_{}".format(i), reuse=tf.compat.v1.AUTO_REUSE):
                    enc_output, enc_slf_attn = multi_head_attention(enc_output,
                                                                    enc_output,
                                                                    enc_output,
                                                                    slf_attn_mask,
                                                                    args.n_head,
                                                                    args.d_model,
                                                                    args.d_k,
                                                                    args.d_v,
                                                                    self.dropout_rate,
                                                                    self.initializer)
                    enc_output *= non_pad_mask

                    enc_output = position_wise(enc_output,
                                               args.d_model, args.d_ff, self.dropout_rate, self.initializer)
                    enc_output *= non_pad_mask

        return enc_output, non_pad_mask

    def distillation(self, output, dist, scope):
        args = self.args

        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            output = tf.layers.dense(
                output, args.dist_model, kernel_initializer=self.initializer)
            distill_loss = tf.losses.mean_squared_error(dist, output)
            output = tf.layers.dense(
                output, args.d_model, kernel_initializer=self.initializer)
            output = tf.nn.dropout(output, keep_prob=self.dropout_rate)

        return output, distill_loss

    def decode(self, tgt_tensor, tgt_postion, tgt_max_len, src_tensor, enc_output):
        args = self.args

        with tf.compat.v1.variable_scope("decode", reuse=tf.compat.v1.AUTO_REUSE):

            dec_output = tf.nn.embedding_lookup(
                self.word_embedding, tgt_tensor)
            dec_output *= args.d_model**0.5

            dec_output += tf.nn.embedding_lookup(
                self.pos_embedding, tgt_postion)

            dec_output = tf.nn.dropout(dec_output, keep_prob=self.dropout_rate)

            # decode mask
            non_pad_mask = common.get_non_pad_mask(tgt_tensor)
            slf_attn_mask_subseq = common.get_subsequent_mask(
                tgt_tensor, self.batch_size, tgt_max_len)
            slf_attn_mask_keypad = common.get_attn_key_pad_mask(
                tgt_tensor, tgt_tensor, tgt_max_len)
            slf_attn_mask = tf.math.greater(
                (slf_attn_mask_keypad + slf_attn_mask_subseq), 0)
            dec_enc_attn_mask = common.get_attn_key_pad_mask(
                src_tensor, tgt_tensor, tgt_max_len)

            for i in range(args.dec_stack_layers):
                with tf.compat.v1.variable_scope(f"num_blocks_{i}", reuse=tf.compat.v1.AUTO_REUSE):
                    dec_output, dec_slf_attn = multi_head_attention(dec_output,
                                                                    dec_output,
                                                                    dec_output,
                                                                    slf_attn_mask,
                                                                    args.n_head,
                                                                    args.d_model,
                                                                    args.d_k,
                                                                    args.d_v,
                                                                    self.dropout_rate,
                                                                    self.initializer,
                                                                    scope="self_attention")
                    dec_output *= non_pad_mask
                    m_dec_output = dec_output

                    dec_output, dec_enc_attn = multi_head_attention(dec_output,
                                                                    enc_output,
                                                                    enc_output,
                                                                    dec_enc_attn_mask,
                                                                    args.n_head,
                                                                    args.d_model,
                                                                    args.d_k,
                                                                    args.d_v,
                                                                    self.dropout_rate,
                                                                    self.initializer,
                                                                    scope="vanilla_attention")
                    dec_output *= non_pad_mask

                    dec_output = position_wise(
                        dec_output, args.d_model, args.d_ff, self.dropout_rate, self.initializer)
                    dec_output *= non_pad_mask

            dec_output = m_dec_output

        return dec_output, non_pad_mask

    def pointer_network(self, enc_output, dec_output):
        args = self.args

        with tf.compat.v1.variable_scope("pointer", reuse=tf.compat.v1.AUTO_REUSE):

            last_enc_output = tf.layers.dense(
                enc_output, args.d_model, use_bias=False, kernel_initializer=self.initializer)  # bsz slen dim
            last_enc_output = tf.expand_dims(
                last_enc_output, 1)  # bsz 1 slen dim

            dec_output = tf.layers.dense(
                dec_output, args.d_model, use_bias=False, kernel_initializer=self.initializer)  # bsz tlen dim
            dec_output = tf.expand_dims(dec_output, 2)  # bsz tlen 1 dim

            attn_encode = tf.nn.tanh(
                dec_output + last_enc_output)  # bsz tlen slen dim
            attn_encode = tf.layers.dense(
                attn_encode, 1, use_bias=False, kernel_initializer=self.initializer)  # bsz tlen slen 1
            attn_encode = tf.nn.dropout(tf.squeeze(
                attn_encode, 3), keep_prob=self.dropout_rate)  # bsz tlen slen
            distributes = tf.nn.log_softmax(attn_encode, axis=-1)+1e-9

        return distributes, attn_encode

    def _cross_entropy(self):

        gold = label_smoothing(tf.one_hot(
            self.tgt_indexs_tensor, depth=self.src_max_len), self.args.d_model)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.distributes, labels=gold)

        nonpadding = tf.cast(tf.not_equal(
            self.tgt_indexs_tensor, const.PAD), tf.float32)

        return tf.reduce_sum(loss * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-9)

    def bert_vecs(self, src_context, tgt_context, bert, src_max_len, tgt_max_len):
        src_tgt_context = np.asarray(
            [f"{t}{const.SPLIT}{s}" for s, t in zip(src_context, tgt_context)])
        _, dec_vecs = bert.encode(src_tgt_context)
        _, enc_vecs = bert.encode(src_context)

        return enc_vecs[:, :src_max_len], dec_vecs[:, :tgt_max_len]

    def pre_train(self, training_data, sess, bert, batch_size):
        total_loss = 0
        for (src_tensor, src_postion, turns_tensor), (tgt_tensor, tgt_postion), tgt_indexs_tensor, src_max_len, eos_indexs, tgt_max_len, src_context, tgt_context in training_data:
            enc_vecs, dec_vecs = self.bert_vecs(
                src_context, tgt_context, bert, src_max_len, tgt_max_len)

            feed_dict = {
                self.src_max_len: src_max_len,
                self.batch_size: batch_size,
                self.tgt_max_len: tgt_max_len,
                self.src_tensor: src_tensor,
                self.src_postion: src_postion,
                self.turns_tensor: turns_tensor,
                self.tgt_tensor: tgt_tensor,
                self.tgt_postion: tgt_postion,
                self.dropout_rate: self.args.dropout,
                self.dist_encode: enc_vecs,
                self.dist_decode: dec_vecs,
            }

            _, step, loss = sess.run(
                [self.global_step, self.pre_train_op, self.distill_loss], feed_dict)
            total_loss += loss
            training_data.set_description(
                f"pre-train Processing  (loss={loss:.4f})")

        return total_loss

    def train(self, training_data, sess, bert, batch_size):
        total_loss = total_correct = total_gold = rouge_scores = 0
        for (src_tensor, src_postion, turns_tensor), (tgt_tensor, tgt_postion), tgt_indexs_tensor, src_max_len, eos_indexs, tgt_max_len, src_context, tgt_context in training_data:
            enc_vecs, dec_vecs = self.bert_vecs(
                src_context, tgt_context, bert, src_max_len, tgt_max_len)

            feed_dict = {
                self.src_max_len: src_max_len,
                self.batch_size: batch_size,
                self.tgt_max_len: tgt_max_len,
                self.src_tensor: src_tensor,
                self.src_postion: src_postion,
                self.turns_tensor: turns_tensor,
                self.tgt_tensor: tgt_tensor,
                self.tgt_postion: tgt_postion,
                self.tgt_indexs_tensor: tgt_indexs_tensor,
                self.dropout_rate: self.args.dropout,
                self.dist_encode: enc_vecs,
                self.dist_decode: dec_vecs,
            }

            _, step, loss, distributes = sess.run(
                [self.global_step, self.train_op, self.loss, self.distributes], feed_dict)

            predict = np.argmax(distributes, axis=2)
            n_correct = ((predict == tgt_indexs_tensor)
                         * (tgt_indexs_tensor != 0)).sum()
            n_gold = (tgt_indexs_tensor != 0).sum()
            rouge_score = common.rouge_l(
                predict, tgt_indexs_tensor, eos_indexs)

            total_loss += loss
            total_correct += n_correct
            total_gold += n_gold
            rouge_scores += rouge_score

            training_data.set_description(
                f"train Processing  (loss={loss:.4f} correct={n_correct}/{n_gold})")

        return total_loss, total_correct, total_gold, rouge_scores

    def valid(self, validation_data, sess, bert, batch_size):
        total_loss = total_correct = total_gold = rouge_scores = 0
        for (src_tensor, src_postion, turns_tensor), (tgt_tensor, tgt_postion), tgt_indexs_tensor, src_max_len, eos_indexs, tgt_max_len, src_context, tgt_context in validation_data:
            enc_vecs, dec_vecs = self.bert_vecs(
                src_context, tgt_context, bert, src_max_len, tgt_max_len)

            feed_dict = {
                self.src_tensor: src_tensor,
                self.batch_size: batch_size,
                self.tgt_max_len: tgt_max_len,
                self.src_postion: src_postion,
                self.turns_tensor: turns_tensor,
                self.tgt_tensor: tgt_tensor,
                self.tgt_postion: tgt_postion,
                self.tgt_indexs_tensor: tgt_indexs_tensor,
                self.src_max_len: src_max_len,
                self.dropout_rate: 1.,
                self.dist_encode: enc_vecs,
                self.dist_decode: dec_vecs,
            }

            _, loss, distributes = sess.run(
                [self.global_step, self.loss, self.distributes], feed_dict)

            predict = np.argmax(distributes, axis=2)
            n_correct = ((predict == tgt_indexs_tensor)
                         * (tgt_indexs_tensor != 0)).sum()
            n_gold = (tgt_indexs_tensor != 0).sum()
            rouge_score = common.rouge_l(
                predict, tgt_indexs_tensor, eos_indexs)

            total_loss += loss
            total_correct += n_correct
            total_gold += n_gold
            rouge_scores += rouge_score

            validation_data.set_description(
                f"valid Processing  (loss={loss:.4f} correct={n_correct}/{n_gold})")

        return total_loss, total_correct, total_gold, rouge_scores

    def debug(self, training_data, sess, bert):
        (src_tensor, src_postion, turns_tensor), (tgt_tensor,
                                                  tgt_postion), tgt_indexs_tensor, src_max_len, eos_indexs, tgt_max_len, src_context, tgt_context = next(training_data)
        enc_vecs, dec_vecs = self.bert_vecs(
            src_context, tgt_context, bert, src_max_len, tgt_max_len)

        feed_dict = {
            self.src_max_len: src_max_len,
            self.batch_size: training_data.batch_size,
            self.tgt_max_len: tgt_max_len,
            self.src_tensor: src_tensor,
            self.src_postion: src_postion,
            self.turns_tensor: turns_tensor,
            self.tgt_tensor: tgt_tensor,
            self.tgt_postion: tgt_postion,
            self.tgt_indexs_tensor: tgt_indexs_tensor,
            self.dropout_rate: self.args.dropout,
        }

        _, distributes = sess.run(
            [self.global_step, self.distributes], feed_dict)
