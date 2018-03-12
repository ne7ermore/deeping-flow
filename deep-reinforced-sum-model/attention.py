import tensorflow as tf

def intra_temp_atten(w_enc, enc_docs, dec_hidden, d_max_len, pre_enc_hiddens):
    """
    Args:
        w_enc: w_enc_atten, size - [dec_hsz*dec_hsz]
        enc_docs: encode docs, size - [bsz*d_max_len*feat]
        dec_hidden: decode hidden/time, size - [bsz*dec_hsz]
        d_max_len: doc man length
        pre_enc_hiddens: cache

    Return:
        enc_c_t: doc vector, size - [bsz*dec_hsz]
    """

    # B x T x F --> T x B x F
    enc_docs = tf.transpose(enc_docs, perm=[1, 0, 2])

    # formulation 2 | T x B x F
    e_ti = tf.exp(tf.multiply(tf.tile(tf.expand_dims(
            tf.matmul(dec_hidden, w_enc), 0), [d_max_len, 1, 1]), enc_docs))

    # formulation 3
    if len(pre_enc_hiddens) == 0:
        pre_enc_hiddens.append(e_ti)

    else:
        e_ti = tf.div(e_ti, pre_enc_hiddens[0])
        pre_enc_hiddens[0] = tf.add(pre_enc_hiddens[0], e_ti)

    # formulation 4
    norm_e_tjs = tf.tile(tf.reduce_sum(
            e_ti, 0, keep_dims=True), [d_max_len, 1, 1])
    alpha_e_tis = tf.div(e_ti, norm_e_tjs)

    # formulation 5
    enc_c_t = tf.reduce_sum(tf.multiply(alpha_e_tis, enc_docs), 0)

    return enc_c_t

def intra_decoder_atten(w_dec, dec_hidden, dec_out):
    """
    Args:
        w_dec: w_dec_atten, size - [dec_hsz*dec_hsz]
        dec_hidden: decode hidden/time, size - [bsz*dec_hsz]
        dec_out: decode out, size - [bsz*time*dec_hsz]

    Return:
        dec_c_t: doc vector, size - [bsz*dec_hsz]
    """
    pre_hiddens = tf.transpose(dec_out, perm=[1, 0, 2])
    times = tf.shape(dec_out)[1]

     # formulation 6
    d_tts = tf.exp(tf.multiply(tf.tile(tf.expand_dims(
        tf.matmul(dec_hidden, w_dec), 0), [times, 1, 1]), pre_hiddens))

    # formulation 7
    norm_d_tt = tf.tile(tf.reduce_sum(d_tts, 0, keep_dims=True), [times, 1, 1])
    alpha_dec_tts = tf.div(d_tts, norm_d_tt)

    # formulation 8
    dec_c_t = tf.reduce_sum(tf.multiply(alpha_dec_tts, pre_hiddens), 0)

    return dec_c_t