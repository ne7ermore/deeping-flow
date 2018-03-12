import const
import pickle
import re

import numpy as np
from const import *

def prepare(doc):
    doc = utterance_preprocess(doc)
    doc = remove_urls(doc)
    doc = remove_emails(doc)
    doc = remove_images(doc)
    return zh_abc_n(doc)

def corpora2idx(sents, ind2idx):
    return [[ind2idx[w] if w in ind2idx else const.UNK for w in s] for s in sents]

def glorot_uniform(in_dim, out_dim):
    return np.sqrt(6.0 / (in_dim + out_dim))

def middle_save(obj, file_):
    pickle.dump(obj, open(file_, "wb"), True)

def middle_load(file_):
    return pickle.load(open(file_, "rb"))

def load_pre_w2c(file_, dict_):
    w2c_dict = {}
    for line in open(file_):
        temp = line.strip().split(" ")
        if len(temp) < 10: continue
        w2c_dict[temp[0]] = list(map(float, temp[1:]))

        if "len_" not in locals():
            len_ = len(temp[1:])

    emb_mx = np.random.rand(len(dict_), len_)
    for word, idx in sorted(dict_.items(), key=lambda x: x[1]):
        if word in w2c_dict:
            emb_mx[idx] = np.asarray(w2c_dict[word])

    return emb_mx

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

if __name__ == "__main__":
    # obj = {"a": 1}
    # print(middle_load("test"))

    print(load_pre_w2c("data/pre-train.w2v", middle_load("data/dssm_middle")))
