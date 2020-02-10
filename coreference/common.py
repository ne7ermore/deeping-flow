import re
import pickle
import logging

import tensorflow as tf
import numpy as np

import const


def set_logger(context, verbose=False, useFile=None):
    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).5s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt='%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if useFile:
        file_handle = logging.FileHandler(useFile)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)

    return logger


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return sinusoid_table


def get_non_pad_mask(seq):
    assert len(seq.get_shape()) == 2

    mask = tf.math.not_equal(seq, const.PAD)
    mask = tf.expand_dims(mask, -1)
    return tf.cast(mask, tf.float32)


def get_attn_key_pad_mask(seq_k, seq_q, len_q):
    padding_mask = tf.math.equal(seq_k, const.PAD)
    padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [1, len_q, 1])

    return tf.cast(padding_mask, tf.float32)


def get_subsequent_mask(seq, sz_b, len_s):
    subsequent_mask = 1 - \
        tf.linalg.LinearOperatorLowerTriangular(
            tf.ones((len_s, len_s))).to_dense()
    subsequent_mask = tf.tile(tf.expand_dims(subsequent_mask, 0), [sz_b, 1, 1])
    return tf.cast(subsequent_mask, tf.float32)


def is_chinese_char(c):
    if ((c >= 0x4E00 and c <= 0x9FFF) or
        (c >= 0x3400 and c <= 0x4DBF) or
        (c >= 0x20000 and c <= 0x2A6DF) or
        (c >= 0x2A700 and c <= 0x2B73F) or
        (c >= 0x2B740 and c <= 0x2B81F) or
        (c >= 0x2B820 and c <= 0x2CEAF) or
        (c >= 0xF900 and c <= 0xFAFF) or
            (c >= 0x2F800 and c <= 0x2FA1F)):

        return True

    return False


def split_char(text):
    text = "".join([w for w in text.split()])
    step, words, replaced_words = 0, [], []
    un_chinese = ""
    while step < len(text):
        if is_chinese_char(ord(text[step])):
            words.append(text[step])
            replaced_words.append(text[step])
            step += 1
        else:
            while step < len(text):
                if is_chinese_char(ord(text[step])):
                    words.append(un_chinese.lower())
                    replaced_words.append(const.WORD[const.UNC])
                    un_chinese = ""
                    break
                un_chinese += text[step]
                step += 1

    if un_chinese:
        return words + [un_chinese.lower()], replaced_words + [const.WORD[const.UNC]]

    return words, replaced_words


def texts2idx(texts, word2idx):
    return [[word2idx[word] if word in word2idx else const.UNK for word in text] for text in texts]


def find_index(text, word):
    stop_index = text.index(const.WORD[const.EOS])
    if word in text[stop_index:]:
        idx = text.index(word, stop_index)
    else:
        idx = text.index(word)
        text[idx] = "@@@"
    return idx


def find_text_index(q_words, new_tgt_words):
    word_map, q_words = {}, q_words.copy()
    t_index = np.zeros(len(new_tgt_words), dtype=int)
    for index, word in enumerate(new_tgt_words):
        if word in q_words:
            pointer = find_index(q_words, word)
            t_index[index] = pointer
            word_map[word] = pointer
        elif word in word_map:
            t_index[index] = word_map[word]
        else:
            raise Exception(
                f"invalid word {word} from {''.join(q_words)} {''.join(new_tgt_words)}")
    return t_index


def middle_save(obj, file_):
    pickle.dump(obj, open(file_, "wb"), True)


def middle_load(file_):
    return pickle.load(open(file_, "rb"))


def longest_common_seq(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (m+1) for _ in range(n+1)]
    if n == 0 or m == 0:
        return 0, m, n

    for j in range(1, n+1):
        for i in range(1, m+1):
            if x[i-1] == y[j-1]:
                dp[j][i] = dp[j-1][i-1] + 1
            else:
                dp[j][i] = max([dp[j][i-1], dp[j-1][i]])

    return dp[-1][-1], m, n


def split_by_eos(datas, eos_idx):
    for idx, d in enumerate(datas):
        if d == eos_idx:
            return datas[:idx]
    return datas


def rouge_l(evals, refs, eos_idxs):
    assert evals.shape == refs.shape

    scores = []
    for eva, ref, eos_idx in zip(evals, refs, eos_idxs):
        eva = split_by_eos(eva, eos_idx)
        ref = split_by_eos(ref, eos_idx)
        same_len, eva_len, ref_len = map(float, longest_common_seq(eva, ref))
        r_lcs = same_len/ref_len if ref_len else 0
        p_lcs = same_len/eva_len if eva_len else 0

        beta = p_lcs / (r_lcs + 1e-12)
        f_lcs = ((1 + (beta**2)) * r_lcs * p_lcs) / \
            (r_lcs + ((beta**2) * p_lcs) + 1e-12)
        scores.append(f_lcs)

    return np.asarray(scores, dtype=np.float32).sum()


if __name__ == '__main__':
    # data = np.asarray([[3,4,3,0,1,0],[2,4,1,4,0,0]], dtype=np.int64)
    # label = np.asarray([[3,4,2,3,9,0],[2,4,2,9,1,0]], dtype=np.int64)
    # eos_idxs = np.asarray([1, 9], dtype=np.int64)

    # print(rouge_l(data, label, eos_idxs))
    print(split_char("lg v30详情你看嘛配置详情lgv30配置详情asd"))
