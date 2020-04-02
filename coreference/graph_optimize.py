
import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

import collections
import re


def _get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)
    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def graph_optimize(model_args, model_class, graph_output_dir, checkpoint_name, output_tensors, input_datatype, input_tensors):
    tf.gfile.MakeDirs(graph_output_dir)
    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

    with jit_scope():
        model_class(model_args)
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = _get_assignment_map_from_checkpoint(
            tvars, checkpoint_name)
        tf.train.init_from_checkpoint(checkpoint_name, assignment_map)
        tmp_g = tf.get_default_graph().as_graph_def()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_g = tf.graph_util.convert_variables_to_constants(
            sess, tmp_g, output_tensors)
        tmp_g = optimize_for_inference(
            tmp_g, input_tensors, output_tensors, input_datatype, False)

    with tf.gfile.GFile(f"{graph_output_dir}/graph", 'wb') as f:
        f.write(tmp_g.SerializeToString())


if __name__ == "__main__":
    from common import middle_load
    from model import Transformer

    class CorpusArgs(object):
        def __init__(self, _args):
            for k, v in _args.items():
                self.__setattr__(k, v)

    data = middle_load("weights/corpus")
    cargs = CorpusArgs(data["args"])

    graph_optimize(cargs, Transformer, "weights", "weights/model", ["enc_output", "pre_distributes"], [tf.int32.as_datatype_enum, tf.int32.as_datatype_enum, tf.int32.as_datatype_enum, tf.int64.as_datatype_enum, tf.int64.as_datatype_enum, tf.int64.as_datatype_enum, tf.float32.as_datatype_enum, tf.int64.as_datatype_enum, tf.int64.as_datatype_enum, tf.float32.as_datatype_enum], [
                   "init_variables/src_max_len", "init_variables/batch_size", "init_variables/tgt_max_len", "init_variables/src_tensor", "init_variables/src_postion", "init_variables/turns_tensor", "init_variables/dropout_keep_prob", "init_variables/tgt_tensor", "init_variables/tgt_postion", "init_variables/pre_enc_output"])
