import re
import pickle

import numpy as np

from const import *


def word2idx(sents, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in sents]


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def middle_save(obj, file_):
    pickle.dump(obj, open(file_, "wb"), True)


def middle_load(file_):
    return pickle.load(open(file_, "rb"))


class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK,
            WORD[BOS]: BOS,
            WORD[EOS]: EOS
        }
        self.idx = len(self.word2idx)

    def add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def __call__(self, sents, min_count):
        words = [word for sent in sents for word in sent]
        word_count = {w: 0 for w in set(words)}
        for w in words:
            word_count[w] += 1

        ignored_word_count = 0
        for word, count in word_count.items():
            if count <= min_count:
                ignored_word_count += 1
                continue
            self.add(word)

        return ignored_word_count

    def __len__(self):
        return self.idx

    def __str__(self):
        return "%s(size = %d)".format(self.__class__.__name__, len(self.idx))


class Corpus(object):
    def __init__(self, fuel_path, save_data, max_w_len, max_l_len, min_word_count=2, easy=True):
        self._fuel_path = fuel_path
        self._save_data = save_data
        self.max_w_len = max_w_len
        self.max_l_len = max_l_len - 1
        self._min_word_count = min_word_count
        self.easy = easy
        self.src_dict = Dictionary()
        self.tgt_dict = Dictionary()

    def parse_file(self, _file, is_label=True):
        sents, ignore_count = [], 0
        for sentence in open(_file):
            sentence = normalizeString(sentence)
            words = sentence.strip().split()

            if self.easy:
                if is_label and len(words) > 53:
                    continue

                if not is_label and len(words) > 632:
                    continue

            if is_label:
                if len(words) > self.max_l_len:
                    ignore_count += 1
                    words = words[:self.max_l_len]
                sents.append(words + [WORD[EOS]])

            if not is_label:
                if len(words) > self.max_w_len:
                    ignore_count += 1
                    words = words[:self.max_w_len]
                sents.append(words)

        return sents, ignore_count

    def parse_files(self):
        self.t_fuel, tf_ignore = self.parse_file(
            self._fuel_path + 'train_fuel', False)
        self.t_label, tl_ignore = self.parse_file(
            self._fuel_path + 'train_label')
        self.v_fuel, _ = self.parse_file(self._fuel_path + 'valid_fuel', False)
        self.v_label, _ = self.parse_file(self._fuel_path + 'valid_label')

        print(
            "Doc`s length out of {} - [{}]".format(self.max_w_len, tf_ignore))
        print(
            "Label`s length out of {} - [{}]".format(self.max_l_len, tl_ignore))

        src_word_ignore = self.src_dict(self.t_fuel, self._min_word_count)
        tgt_word_ignore = self.tgt_dict(self.t_label, self._min_word_count)

        if src_word_ignore != 0:
            print("Ignored src word counts - [{}]".format(src_word_ignore))
        if tgt_word_ignore != 0:
            print("Ignored tgt word counts - [{}]".format(tgt_word_ignore))

    def save(self):
        data = {
            'max_w_len': self.max_w_len,
            'max_l_len': self.max_l_len + 1,
            'dict': {
                'src': self.src_dict.word2idx,
                'src_size': len(self.src_dict),
                'tgt': self.tgt_dict.word2idx,
                'tgt_size': len(self.tgt_dict),
                'id2word': {v: k for k, v in self.tgt_dict.word2idx.items()}
            },
            'train': {
                'data': word2idx(self.t_fuel, self.src_dict.word2idx),
                'label': word2idx(self.t_label, self.tgt_dict.word2idx),
            },
            'valid': {
                'data': word2idx(self.v_fuel, self.src_dict.word2idx),
                'label': word2idx(self.v_label, self.tgt_dict.word2idx),
            }
        }

        middle_save(data, self._save_data)
        print('src word length - [{}]'.format(len(self.src_dict)))
        print('tgt word length - [{}]'.format(len(self.tgt_dict)))

    def process(self):
        self.parse_files()
        self.save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='drsm')
    parser.add_argument('--save_data', type=str, default='data/corpus')
    parser.add_argument('--fuel_path', type=str, default='data/')
    parser.add_argument('--max_w_len', type=int, default=512)
    parser.add_argument('--max_l_len', type=int, default=40)
    parser.add_argument('--min_word_count', type=int, default=2)
    args = parser.parse_args()

    corpus = Corpus(args.fuel_path, args.save_data,
                    args.max_w_len, args.max_l_len, args.min_word_count)
    corpus.process()
