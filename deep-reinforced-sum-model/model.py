import tensorflow as tf
import numpy as np
import time

from const import BOS
from rouge import rouge_l

def embedding(vocab_size, emb_dim, zero_pad=True):
    emb_matrix = tf.get_variable('emb_matrix',
                         dtype=tf.float32,
                         shape=[vocab_size, emb_dim],
                         initializer=tf.contrib.layers.xavier_initializer())
    if zero_pad:
        emb_matrix = tf.concat((tf.zeros(shape=[1, emb_dim]),
                                  emb_matrix[1:, :]), 0)

    return emb_matrix

def Encoder(docs, hidden_size, dropout, bsz):
    fw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    bw_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

    fw_hidden = fw_rnn_cell.zero_state(bsz, dtype=tf.float32)
    bw_hidden = bw_rnn_cell.zero_state(bsz, dtype=tf.float32)

    encodes, (fw_states, bw_states) = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, docs,initial_state_fw=fw_hidden,initial_state_bw=bw_hidden, dtype=tf.float32)  

    states = tf.contrib.rnn.LSTMStateTuple(
            c=tf.concat((fw_states.c, bw_states.c), -1),
            h=tf.concat((fw_states.h, bw_states.h), -1))

    encodes = tf.concat(encodes, -1)

    if dropout != 1.:
        encodes = tf.nn.dropout(encodes, dropout)

    return encodes, states

def intra_temp_atten(w_enc, enc_docs, dec_hidden, pre_enc_hiddens):
    # B x T x F --> T x B x F
    enc_docs = tf.unstack(
            tf.transpose(enc_docs, perm=[1, 0, 2]))

    # T x B x F
    # formulation 2
    e_ti = tf.concat([tf.expand_dims(tf.exp(tf.multiply(
            tf.matmul(dec_hidden, w_enc), feat)), 0) for feat in enc_docs], 0) 

    # formulation 3
    if len(pre_enc_hiddens) == 0:
        pre_enc_hiddens.append(e_ti)
    else:
        norm_e_tis = tf.reduce_sum(tf.concat(
            [tf.expand_dims(h, 0) for h in pre_enc_hiddens], 0), 0)
        e_ti = tf.div(e_ti, norm_e_tis)
        pre_enc_hiddens.append(e_ti)

    # formulation 4
    norm_e_tjs = tf.reduce_sum(e_ti, 0)
    alpha_e_tis = [tf.div(e_tj, norm_e_tjs) for e_tj in tf.unstack(e_ti)]

    # formulation 5
    enc_c_t = tf.reduce_sum(tf.concat([tf.expand_dims(tf.multiply(alpha_enc_ti, enc_h_i), 0) for alpha_enc_ti, enc_h_i in zip(alpha_e_tis, enc_docs)], 0), 0)

    return enc_c_t

def intra_decoder_atten(dec_hidden, dec_out, w_dec):    
    pre_hiddens = tf.unstack(
            tf.transpose(dec_out, perm=[1, 0, 2]))[:-1]

    # formulation 6
    d_tts = [tf.exp(tf.multiply(tf.matmul(
            dec_hidden, w_dec), h)) for h in pre_hiddens] 

    # formulation 7
    norm_d_tt = tf.reduce_sum(tf.concat([
        tf.expand_dims(d_tt, 0) for d_tt in d_tts], 0), 0)
    alpha_dec_tts = [tf.div(d_tt, norm_d_tt) for d_tt in d_tts]

    # formulation 8
    dec_c_t = tf.reduce_sum(tf.concat([tf.expand_dims(tf.multiply(alpha_dec_tt, pre_hidden), 0) for alpha_dec_tt, pre_hidden in zip(alpha_dec_tts, pre_hiddens)], 0), 0)

    return dec_c_t

class Summarizor(object):
    def __init__(self, args, batch_size):
        self.args = args
        self.batch_size = batch_size
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.norm_init = tf.truncated_normal_initializer(
                    stddev=args.n_scope, seed=args.seed)

        self.uniform_init = tf.random_uniform_initializer(
                    -args.u_scope, args.u_scope, seed=args.seed)

        print('initing graph')
        s_time = time.time()

        self.init_placeholders()
        self.init_vars()
        self.build()

        optimizer = tf.train.AdamOptimizer(
                self.args.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        print('graph inited, time cost - {:.2f}s'.format(time.time() - s_time))        

    def init_placeholders(self):
        args = self.args

        self.doc = tf.placeholder(
            tf.int64, [self.batch_size, args.d_max_len], name="doc")

        self.label = tf.placeholder(tf.float32, [
            self.batch_size, args.l_max_len, args.vocab_size], name="label")

        self.label_ids = tf.placeholder(
            tf.int64, [self.batch_size, args.l_max_len], name="label") 

        self.dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def init_vars(self):
        args = self.args

        with tf.variable_scope('init_variables'):
            self.emb_matrix = embedding(args.vocab_size, args.emb_dim) 

            self.w_enc = tf.get_variable("w_enc_atten", [args.dec_hsz,
                args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init)  

            self.w_dec = tf.get_variable("w_dec_atten", [args.dec_hsz,
                args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init) 

            self.w_proj = tf.get_variable("w_proj", [args.emb_dim,
                args.dec_hsz*3], dtype=tf.float32, initializer=self.uniform_init)

        
            self.w_out = tf.nn.tanh(
                tf.transpose(tf.matmul(self.emb_matrix, self.w_proj), perm=[1, 0]))

            self.b_out = tf.Variable(
                tf.constant(0.1, shape=[args.vocab_size]), name='b_out')

            self.w_base = tf.get_variable("w_base", [args.emb_dim,
                args.dec_hsz], dtype=tf.float32, initializer=self.uniform_init) 

            self.w_base_out = tf.nn.tanh(
                tf.transpose(tf.matmul(self.emb_matrix, self.w_base), perm=[1, 0]))

            self.b_base = tf.Variable(tf.constant(0.1, shape=[args.vocab_size]), name='b_base')

    def build(self):
        args = self.args
        emb_docs = tf.nn.embedding_lookup(self.emb_matrix, self.doc)

        with tf.variable_scope('encode'):
            enc_docs, encode_states = Encoder(emb_docs, args.enc_hsz, args.dropout, self.batch_size)

        with tf.variable_scope('decode'):
            pre_enc_hiddens = []
            dec_state = encode_states

            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(args.dec_hsz)

            summ = tf.constant(BOS, shape=[self.batch_size, 1], dtype=tf.int64)
            scores, base_lines, proj_ids, proj_scores = [], [], [], []

            for step in range(args.l_max_len):
                emb_summ = tf.nn.embedding_lookup(self.emb_matrix, summ)
                dec_out, dec_state = tf.nn.dynamic_rnn(
                                        cell=rnn_cell,
                                        inputs=emb_summ,
                                        initial_state=dec_state,
                                        dtype=tf.float32)
                dec_hidden = dec_state.h
                
                if args.dropout != 1.:
                    dec_out = tf.nn.dropout(dec_out, args.dropout)
                    dec_hidden = tf.nn.dropout(dec_hidden, args.dropout)

                enc_c_t = intra_temp_atten(self.w_enc, enc_docs, dec_hidden, pre_enc_hiddens)

                if step == 0:
                    # We set dec_c to a vector of zeros since the generated sequence is empty on the first decoding step   
                    dec_c_t = tf.constant(0., shape=[self.batch_size, args.dec_hsz])
                else:
                    dec_c_t = intra_decoder_atten(
                            dec_hidden, dec_out, self.w_dec)

                # formulation 9
                out = tf.concat((dec_hidden, enc_c_t, dec_c_t), -1)
                token_gen = tf.nn.xw_plus_b(
                    out, self.w_out, self.b_out, name='token_generation') 
                token_gen = tf.nn.log_softmax(token_gen)

                # formulation 12
                # no implement for pointer mechanism
                next_id = tf.expand_dims(tf.argmax(token_gen, 1), 1)
                summ = tf.concat((summ, next_id), 1)

                base_line = tf.nn.xw_plus_b(
                    dec_hidden, self.w_base_out, self.b_base, name='base_line')
                base_line = tf.expand_dims(tf.argmax(base_line, 1), 1)

                proj_score = tf.reduce_max(token_gen, axis=-1, keep_dims=True)

                # for ml loss
                scores.append(tf.expand_dims(token_gen, 1))

                # for rl loss
                proj_ids.append(next_id)
                base_lines.append(base_line)
                proj_scores.append(proj_score)

        scores = tf.concat(scores, 1)
        proj_ids = tf.concat(proj_ids, 1)
        base_lines = tf.concat(base_lines, 1)
        proj_scores = tf.reduce_sum(tf.concat(proj_scores, 1), 1)

        with tf.variable_scope('loss'):
            # formulation 14
            ml_loss = tf.nn.softmax_cross_entropy_with_logits(
                                logits=scores, labels=self.label)
            ml_loss = tf.reduce_sum(ml_loss, -1)
            tf.summary.scalar('ml_loss', ml_loss)

            # formulation 15
            bl_reward = tf.py_func(
                    rouge_l, [base_lines, self.label_ids], tf.float32)
            p_reward = tf.py_func(
                    rouge_l, [proj_ids, self.label_ids], tf.float32)
            rl_loss = (bl_reward - p_reward) * proj_scores
            tf.summary.scalar('rl_loss', rl_loss)

            # formulation 16
            self.loss = tf.reduce_mean(
                    args.gamma * rl_loss + (1. - args.gamma) * ml_loss)
            
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('corrects'):
            corrects = tf.equal(proj_ids, self.label_ids)
            self.corrects = tf.reduce_sum(
                    tf.cast(corrects, "float"), name="corrects")  
            tf.summary.scalar('corrects', self.corrects)      

        self.merged = tf.summary.merge_all()    

    def train_step(self, batch, sess):
        feed_dict = {
            self.doc: batch.data,
            self.label: batch.label,
            self.label_ids: batch.label_ids,
            self.dropout: self.args.dropout
        }

        _, step, merged = sess.run([
                self.train_op, self.global_step, self.merged], feed_dict)

        return merged, step


         
        


            
