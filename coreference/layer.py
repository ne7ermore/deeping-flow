import numpy as np
import tensorflow as tf


def layer_norm(input_tensor, name=None):
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def get_token_embeddings(vocab_size, num_units, scope, init_func=None):
    with tf.compat.v1.variable_scope(scope):
    if init_func is not None:
    embeddings = tf.Variable(tf.constant(
        init_func(vocab_size, num_units), dtype=tf.float32))
    else:
    embeddings = tf.compat.v1.get_variable(scope,
                                           dtype=tf.float32,
                                           shape=(vocab_size, num_units),
                                           initializer=tf.contrib.layers.xavier_initializer())

    return tf.concat((tf.zeros(shape=[1, num_units]), embeddings[1:, :]), 0)


def scaled_dot_product_attention(q, k, v, d_k, mask, dropout_rate,
                                 scope="scaled_dot_product_attention"):

    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):

    attn = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
    attn = attn / (d_k ** 0.5)
    attn = attn + tf.cast(mask, tf.float32) * (-2 ** 32 + 1)
    attn = tf.nn.softmax(attn)
    attn = tf.nn.dropout(attn, keep_prob=dropout_rate)
    output = tf.matmul(attn, v)

    return output, attn


def multi_head_attention(q, k, v, mask, n_head, d_model, d_k, d_v, dropout_rate, initializer,
                         scope="multi_head_attention"):

    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
    residual = q

    q = layer_norm(q)

    q = tf.layers.dense(q, n_head * d_k, kernel_initializer=initializer)
    k = tf.layers.dense(k, n_head * d_k, kernel_initializer=initializer)
    v = tf.layers.dense(v, n_head * d_k, kernel_initializer=initializer)

    q = tf.concat(tf.split(q, n_head, axis=2), axis=0)
    k = tf.concat(tf.split(k, n_head, axis=2), axis=0)
    v = tf.concat(tf.split(v, n_head, axis=2), axis=0)

    mask = tf.tile(mask, [n_head, 1, 1])

    output, attn = scaled_dot_product_attention(
        q, k, v, d_k, mask, dropout_rate)

    output = tf.concat(tf.split(output, n_head, axis=0), axis=2)

    output = tf.layers.dense(output, d_model, kernel_initializer=initializer)
    output = tf.nn.dropout(output, keep_prob=dropout_rate)

    output = output + residual

    return output, attn


def position_wise(inputs, d_model, d_ff, dropout_rate, initializer, scope="position_wise"):

    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):

    outputs = tf.layers.dense(
        inputs, d_ff, activation=tf.nn.relu, kernel_initializer=initializer)
    outputs = tf.layers.dense(outputs, d_model, kernel_initializer=initializer)
    outputs = tf.nn.dropout(outputs, keep_prob=dropout_rate)
    outputs = layer_norm(outputs+inputs)

    return outputs


def label_smoothing(inputs, d_model, epsilon=0.1):
    return ((1-epsilon) * inputs) + (epsilon / d_model)
