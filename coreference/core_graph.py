import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from model import Transformer
from common import middle_load
import collections
import re


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
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


class CorpusArgs(object):
    def __init__(self, _args):
        for k, v in _args.items():
            self.__setattr__(k, v)


class CoreGraphOptimize:

    output_tensors = ["enc_output", "pre_distributes"]
    input_tensors = ["init_variables/src_max_len", "init_variables/batch_size", "init_variables/tgt_max_len", "init_variables/src_tensor", "init_variables/src_postion",
                     "init_variables/turns_tensor", "init_variables/dropout_keep_prob", "init_variables/tgt_tensor", "init_variables/tgt_postion", "init_variables/pre_enc_output"]
    input_datatype = [tf.int32.as_datatype_enum, tf.int32.as_datatype_enum, tf.int32.as_datatype_enum, tf.int64.as_datatype_enum, tf.int64.as_datatype_enum,
                      tf.int64.as_datatype_enum, tf.float32.as_datatype_enum, tf.int64.as_datatype_enum, tf.int64.as_datatype_enum, tf.float32.as_datatype_enum]

    output_dir = "weights"
    checkpoint_name = "weights/model"
    graph_file = "weights/graph"

    def __init__(self, corpus_path="weights/corpus"):
        data = middle_load(corpus_path)
        cargs = CorpusArgs(data["args"])

        tf.gfile.MakeDirs(self.output_dir)

        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

        with jit_scope():
            model = Transformer(cargs)
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(
                tvars, self.checkpoint_name)
            tf.train.init_from_checkpoint(self.checkpoint_name, assignment_map)
            tmp_g = tf.get_default_graph().as_graph_def()

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            print('load parameters from checkpoint...')
            sess.run(tf.global_variables_initializer())
            print('freeze...')
            tmp_g = tf.graph_util.convert_variables_to_constants(
                sess, tmp_g, self.output_tensors)
            print('optimize...')
            tmp_g = optimize_for_inference(
                tmp_g, self.input_tensors, self.output_tensors, self.input_datatype, False)

        print('write graph to a tmp file: %s' % self.graph_file)
        with tf.gfile.GFile(self.graph_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())


if __name__ == "__main__":
    CoreGraphOptimize()
