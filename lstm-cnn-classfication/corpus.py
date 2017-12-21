import logging
import argparse
import os
import pickle
import re

from const import *

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def middle_save(obj, file_):
    pickle.dump(obj, open(file_, "wb"), True)

def middle_load(file_):
    return pickle.load(open(file_, "rb"))

def word2idx(sents, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in sents]

class Dictionary(object):
    def __init__(self, word2idx={}, idx_num=0):
        self.word2idx = word2idx
        self.idx = idx_num

    def _add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def _convert(self):
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return self.idx

    def __str__(self):
        return "%s(size = %d)".format(self.__class__.__name__, len(self.idx))

class Words(Dictionary):
    def __init__(self):
        word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK
        }
        super().__init__(word2idx=word2idx, idx_num=len(word2idx))

    def __call__(self, sents):
        words = set([word for sent in sents for word in sent])
        for word in words:
            self._add(word)

class Labels(Dictionary):
    def __init__(self):
        super().__init__()

    def __call__(self, labels):
        _labels = set(labels)
        for label in _labels:
            self._add(label)

class Corpus(object):
    def __init__(self, path, save_data, max_len=16):
        self.train = os.path.join(path, "train")
        self.valid = os.path.join(path, "valid")
        self._save_data = save_data

        self.w = Words()
        self.l = Labels()
        self.max_len = max_len

    def parse_data(self, _file, is_train=True, fine_grained=False):
        """
        fine_grained: Whether to use the fine-grained (50-class) version of TREC
                or the coarse grained (6-class) version.
        """
        _sents, _labels = [], []
        ignore_nums = 0
        for sentence in open(_file):
            label, _, _words = sentence.replace('\xf0', ' ').partition(' ')
            label = label.split(":")[0] if not fine_grained else label

            words = [normalizeString(w) for w in _words.strip().split()]

            if len(words) > self.max_len:
                words = words[:self.max_len]
                ignore_nums += 1

            _sents += [words]
            _labels += [label]
        if is_train:
            self.w(_sents)
            self.l(_labels)
            self.train_sents = _sents
            self.train_labels = _labels
            print('train data - ignore lines - [{}]'.format(ignore_nums))
        else:
            self.valid_sents = _sents
            self.valid_labels = _labels
            print('test data - ignore lines - [{}]'.format(ignore_nums))

    def save(self):
        self.parse_data(self.train)
        self.parse_data(self.valid, False)

        data = {
            'max_len': self.max_len,
            'dict': {
                'train': self.w.word2idx,
                'vocab_size': len(self.w),
                'label': self.l.word2idx,
                'label_size': len(self.l),
            },
            'train': {
                'src': word2idx(self.train_sents, self.w.word2idx),
                'label': [self.l.word2idx[l] for l in self.train_labels]
            },
            'valid': {
                'src': word2idx(self.valid_sents, self.w.word2idx),
                'label': [self.l.word2idx[l] for l in self.valid_labels]
            }
        }

        middle_save(data, self._save_data)
        print('Finish dumping the data to file - [{}]'.format(self._save_data))
        print('words length - [{}]'.format(len(self.w)))
        print('label size - [{}]'.format(len(self.l)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM CNN Classification')
    parser.add_argument('--file-path', type=str, default="./data")
    parser.add_argument('--save-data', type=str, default="./data/corpus")
    parser.add_argument('--max-lenth', type=int, default=16)
    args = parser.parse_args()
    corpus = Corpus(args.file_path, args.save_data, args.max_lenth)
    corpus.save()
