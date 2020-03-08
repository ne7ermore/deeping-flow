import copy
import os
import collections

import tensorflow as tf
import numpy as np

from const import *
import common


def post_check(contexts, rewrite_context):
    last_context = contexts[-1]

    for w in rewrite_context:
        if last_context.find(w) == -1:
            break
    else:
        return last_context

    context_counts = collections.Counter("".join(contexts))

    rewrite_counts = {}
    for w in rewrite_context:
        if w not in rewrite_counts:
            rewrite_counts[w] = 1
        else:
            if rewrite_counts[w] >= context_counts[w]:
                return last_context
            rewrite_counts[w] += 1

    return rewrite_context


class Predict(object):
    def __init__(self, model_path="model/", beam_size=4, rewrite_len=100, debug=False, use_beam_serch=False):
        self.model_path = model_path
        self.beam_size = beam_size
        self.rewrite_len = rewrite_len
        self.debug = debug
        self.use_beam_serch = use_beam_serch

    def load_model(self):
        data = common.middle_load(os.path.join(self.model_path, "corpus"))
        self.dict = data["word2idx"]
        self.turn_size = data["turn_size"]
        self.max_context_len = data["max_context_len"]

        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.import_meta_graph(
            os.path.join(self.model_path, "model.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
        graph = tf.get_default_graph()

        self.src_max_len = graph.get_tensor_by_name(
            "init_variables/src_max_len:0")
        self.batch_size = graph.get_tensor_by_name(
            "init_variables/batch_size:0")
        self.tgt_max_len = graph.get_tensor_by_name(
            "init_variables/tgt_max_len:0")

        self.src = graph.get_tensor_by_name("init_variables/src_tensor:0")
        self.postion = graph.get_tensor_by_name("init_variables/src_postion:0")
        self.turns = graph.get_tensor_by_name("init_variables/turns_tensor:0")
        self.dropout_rate = graph.get_tensor_by_name(
            "init_variables/dropout_keep_prob:0")
        self.enc_output = graph.get_tensor_by_name("enc_output:0")

        self.tgt = graph.get_tensor_by_name("init_variables/tgt_tensor:0")
        self.tgt_postion = graph.get_tensor_by_name(
            "init_variables/tgt_postion:0")
        self.pre_enc_output = graph.get_tensor_by_name(
            "init_variables/pre_enc_output:0")
        self.distributes = graph.get_tensor_by_name("pre_distributes:0")

        self.session = sess

    def preprocess(self, sentences):
        assert isinstance(sentences, list)

        while sum([len(q) for q in sentences]) > self.max_context_len-1:
            sentences.pop(0)

        q_words, replaced_qs, turns = [], [], []

        turns_count = len(sentences)-1
        for turn, query in enumerate(sentences):
            if turn == turns_count:
                q_words += [WORD[EOS]]
                replaced_qs += [WORD[EOS]]
                turns += [turn]

            q_word, repalced_q = common.split_char(query)
            assert len(q_word) == len(repalced_q)

            replaced_qs += repalced_q
            q_words += q_word
            turns += [turn+1]*(len(q_word))

        assert len(q_words) == len(replaced_qs) == len(turns)

        idx = np.asarray(
            [[self.dict[w] if w in self.dict else UNK for w in replaced_qs]])
        position = np.asarray(
            [[pos_i+1 if w_i != PAD else 0 for pos_i, w_i in enumerate(idx[0])]])
        turns = np.asarray([turns])

        self.word = q_words

        return idx, position, turns

    def widx2didx(self, widx):
        word = self.word[widx]
        return self.dict[word] if word in self.dict else UNK

    def beam_search(self, w_scores, end_seqs, top_seqs):
        max_idxs = np.argsort(w_scores, axis=-1)[:, ::-1][:, :self.beam_size]

        all_seqs, seen = [], []
        for index, seq in enumerate(top_seqs):
            seq_idxs, word_index, seq_score = seq
            if seq_idxs[-1] == EOS:
                all_seqs += [(seq, seq_score, True)]
                continue

            for widx in max_idxs[index]:
                score = w_scores[index][widx]
                idx = self.widx2didx(widx)
                seq_idxs, word_index, seq_score = copy.deepcopy(seq)
                seq_score += score
                seq_idxs += [idx]
                word_index += [widx]
                if word_index not in seen:
                    seen.append(word_index)
                    all_seqs += [((seq_idxs, word_index, seq_score),
                                  seq_score, idx == EOS)]

        all_seqs += [((seq[0], seq[1], seq[-1]), seq[-1], True)
                     for seq in end_seqs]
        top_seqs = sorted(all_seqs, key=lambda seq: seq[1], reverse=True)[
            :self.beam_size]

        all_done, done_nums = self.check_all_done(top_seqs)
        top_seqs = [seq for seq, _, _ in top_seqs]

        return top_seqs, all_done, self.beam_size-done_nums

    def check_all_done(self, seqs):
        done_nums = len([s for s in seqs if s[-1]])
        return done_nums == self.beam_size, done_nums

    def init_input(self, beam_size):
        return np.asarray([[1]]*beam_size), np.asarray([[BOS]]*beam_size)

    def update_input(self, top_seqs):
        end_seqs, un_end_seqs, input_data = [], [], []
        for seq in top_seqs:
            if seq[0][-1] != EOS:
                un_end_seqs.append(seq)
                input_data.append(seq[0])
            else:
                end_seqs.append(seq)

        return np.asarray(input_data), end_seqs, un_end_seqs

    def update_state(self, step, src_seq, enc_output, un_dones):
        input_pos = np.expand_dims(np.arange(1, step+1), 0)
        input_pos = np.tile(input_pos, (un_dones, 1))

        src_seq_beam = np.tile(src_seq, (un_dones, 1))
        enc_output_beam = np.tile(enc_output, (un_dones, 1, 1))

        return input_pos, src_seq_beam, enc_output_beam

    def encode(self, inp, position, turns):
        feed_dict = {
            self.src_max_len: position.shape[1],
            self.batch_size: 1,
            self.src: inp,
            self.postion: position,
            self.turns: turns,
            self.dropout_rate: 1.
        }
        return self.session.run([self.enc_output], feed_dict)[0]

    def decode(self, input_data, input_pos, src_beam, enc_output_beam):
        feed_dict = {
            self.tgt_max_len: input_data.shape[1],
            self.batch_size: 1,
            self.tgt: input_data,
            self.tgt_postion: input_pos,
            self.src: src_beam,
            self.pre_enc_output: enc_output_beam,
            self.dropout_rate: 1.
        }

        return self.session.run([self.distributes], feed_dict)[0]

    def no_beam_divine(self, sentences):
        src, position, turns = self.preprocess(sentences)

        enc_output = self.encode(src, position, turns)
        words = [[BOS]]
        idxs = []

        for step in range(1, self.rewrite_len):
            input_pos = np.arange(1, step+1)[np.newaxis, :]
            input_data = np.array(words)
            s = time.time()
            dec_output = self.decode(input_data, input_pos, src, enc_output)
            print(time.time()-s)
            widx = np.argmax(dec_output[:, -1, :])
            idx = self.widx2didx(widx)
            if idx == EOS:
                break
            words[0].append(idx)
            idxs.append(widx)

        return ["".join([self.word[idx] for idx in idxs])]

    def divine(self, sentences):
        def length_penalty(step, len_penalty_w=1.):
            return (np.log(np.array([5 + step], dtype=np.float32)) - np.log([6]))*len_penalty_w

        src, position, turns = self.preprocess(sentences)

        top_seqs = [([BOS], [], 0)] * self.beam_size
        enc_output = self.encode(src, position, turns)

        src_beam = np.tile(src, (self.beam_size, 1))

        enc_output_beam = np.tile(enc_output, (self.beam_size, 1, 1))
        input_pos, input_data = self.init_input(self.beam_size)
        end_seqs = []

        for step in range(1, self.rewrite_len):
            dec_output = self.decode(
                input_data, input_pos, src_beam, enc_output_beam)
            out = dec_output[:, -1, :]
            lp = length_penalty(step)
            top_seqs, all_done, un_dones = self.beam_search(
                out+lp, end_seqs, top_seqs)
            if all_done:
                break

            input_data, end_seqs, top_seqs = self.update_input(top_seqs)
            input_pos, src_seq_beam, enc_output_beam = self.update_state(
                step+1, src, enc_output, un_dones)
            src_beam = np.tile(src, (un_dones, 1))

        tgts = []
        for (cor_idxs, word_index, score) in top_seqs:
            cor_idxs = word_index[: -1]
            tgts += [("".join([self.word[idx] for idx in cor_idxs]), score)]
        return tgts

    def Trains(self, sentences):
        if self.use_beam_serch:
            answers = self.divine(sentences)
            answer = answers[0][0]
        else:
            answers = self.no_beam_divine(sentences)
            answer = answers[0]
        if self.debug:
            print(answers)

        return post_check(sentences, answer)
