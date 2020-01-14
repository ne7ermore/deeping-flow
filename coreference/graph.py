import json
import logging
from termcolor import colored
import modeling
import tensorflow as tf
import os


def set_logger(context, verbose=False):
    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).5s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt='%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


def optimize_graph(output_dir, config_name, max_seq_len, checkpoint_name, graph_file, logger=None, verbose=False):
    if not logger:
    logger = set_logger(colored('BERT_VEC', 'yellow'), verbose)
    try:
        # we don't need GPU for optimizing the graph
    from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
    tf.gfile.MakeDirs(output_dir)

    config_fp = config_name
    logger.info('model config: %s' % config_fp)

    # 加载bert配置文件
    with tf.gfile.GFile(config_fp, 'r') as f:
    bert_config = modeling.BertConfig.from_dict(json.load(f))

    logger.info('build graph...')
    # input placeholders, not sure if they are friendly to XLA
    input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
    input_mask = tf.placeholder(tf.int32, (None, max_seq_len), 'input_mask')
    input_type_ids = tf.placeholder(
        tf.int32, (None, max_seq_len), 'input_type_ids')

    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

    with jit_scope():
    input_tensors = [input_ids, input_mask, input_type_ids]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=False)

    # 获取所有要训练的变量
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
        tvars, checkpoint_name)
    tf.train.init_from_checkpoint(checkpoint_name, assignment_map)

    # 共享卷积核
    with tf.variable_scope("pooling"):
        # 如果只有一层，就只取对应那一层的weight
    encoder_layer = model.get_sequence_output()

    def mul_mask(x, m): return x * tf.expand_dims(m, axis=-1)

    def masked_reduce_mean(x, m): return tf.reduce_sum(
        mul_mask(x, m), axis=1) / (tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

    input_mask = tf.cast(input_mask, tf.float32)
    # 以下代码是句向量的生成方法，可以理解为做了一个卷积的操作，但是没有把结果相加, 卷积核是input_mask
    encoder_layer = tf.identity(encoder_layer, 'sentence_encodes')
    pooled = masked_reduce_mean(encoder_layer, input_mask)
    pooled = tf.identity(pooled, 'final_encodes')

    output_tensors = [pooled, encoder_layer]
    tmp_g = tf.get_default_graph().as_graph_def()

    # allow_soft_placement:自动选择运行设备
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
    logger.info('load parameters from checkpoint...')
    sess.run(tf.global_variables_initializer())
    logger.info('freeze...')
    tmp_g = tf.graph_util.convert_variables_to_constants(
        sess, tmp_g, [n.name[:-2] for n in output_tensors])
    dtypes = [n.dtype for n in input_tensors]
    logger.info('optimize...')
    tmp_g = optimize_for_inference(
        tmp_g,
        [n.name[:-2] for n in input_tensors],
        [n.name[:-2] for n in output_tensors],
        [dtype.as_datatype_enum for dtype in dtypes],
        False)

    tmp_file = graph_file
    logger.info('write graph to a tmp file: %s' % tmp_file)
    with tf.gfile.GFile(tmp_file, 'wb') as f:
    f.write(tmp_g.SerializeToString())
    return tmp_file
    except Exception as e:
    logger.error('fail to optimize the graph!')
    logger.error(e)
