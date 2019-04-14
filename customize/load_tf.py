import tensorflow as tf


class Predict(object):
    def __init__(self, data, _path, model_path="model/"):
        self.dict = data['dict']['src']
        self.max_q_len = data['max_q_len']
        self.max_i_len = data['max_i_len']

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.import_meta_graph(_path + "epoch_6.meta")
        saver.restore(sess, tf.train.latest_checkpoint(_path))
        graph = tf.get_default_graph()
        self.info = graph.get_tensor_by_name("question:0")
        self.question = graph.get_tensor_by_name("info:0")
        self.dropout = graph.get_tensor_by_name("dropout_keep_prob:0")
        self.op_predictions = graph.get_tensor_by_name("predict/predictions:0")
        self.op_softmax = graph.get_tensor_by_name("predict/softmax:0")
        self.session = sess

    def divine(self, contexts):
        infos, ques = contexts

        feed_dict = {
            self.info: infos,
            self.question: ques,
            self.dropout: 1.
        }

        pred_labels, pred_props = self.session.run(
            [self.op_predictions, self.op_softmax], feed_dict)
