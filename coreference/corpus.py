import numpy as np

import common
from const import *


class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            WORD[BOS]: BOS,
            WORD[EOS]: EOS,
            WORD[PAD]: PAD,
            WORD[UNK]: UNK,
            WORD[UNC]: UNC,
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
    def __init__(self, save_data=f"{DATAPATH}/corpus", min_word_count=1, max_len=200):
        self._save_data = save_data
        self.min_word_count = min_word_count
        self.max_len = max_len
        self.dict = Dictionary()
        self.parse()
        self.save()

    def parse(self):
        def parse_file(inf):
            src_texts, src_turn, tgt_indexs, tgt_texts, eos_indexs, src_contexts, tgt_contexts = [
            ], [], [], [], [], [], []
            with open(inf, encoding="utf8") as contexts:
                for line in contexts:
                    contexts = line.strip().split("\t")
                    querys, target = contexts[:-1], contexts[-1]

                    # 不能长于最大长度
                    while sum([len(q) for q in querys]) > self.max_len-1:
                        querys.pop(0)

                    turns_count = len(querys)-1
                    turns, q_words, replaced_qs = [], [], []
                    for turn, query in enumerate(querys):
                        # last index
                        if turn == turns_count:
                            eos_index = len(q_words)
                            q_words += [WORD[EOS]]
                            replaced_qs += [WORD[EOS]]
                            turns += [turn]

                        q_word, repalced_q = common.split_char(query)
                        assert len(q_word) == len(repalced_q)

                        replaced_qs += repalced_q
                        q_words += q_word
                        turns += [turn+1]*(len(q_word))

                    tgt_words, replaced_tgt = common.split_char(target)
                    assert len(tgt_words) == len(replaced_tgt)

                    new_tgt_words, replaced_new_tgt_words = [], []
                    for word, replaced_word in zip(tgt_words, replaced_tgt):
                        if word in q_words:
                            new_tgt_words.append(word)
                            replaced_new_tgt_words.append(replaced_word)

                    new_tgt_words = new_tgt_words + [WORD[EOS]]
                    t_index = common.find_text_index(q_words, new_tgt_words)

                    # 保存
                    # step 1 - 替换后的q
                    src_contexts.append("".join(replaced_qs))

                    assert len(replaced_qs) == len(turns)
                    src_texts.append(replaced_qs)
                    src_turn.append(turns)

                    # step 2 - 截止符位置
                    eos_indexs.append(eos_index)

                    # step 3 - tgt
                    tgt_texts.append(([WORD[BOS]]+replaced_new_tgt_words))
                    tgt_contexts.append(
                        "".join([WORD[BOS]]+replaced_new_tgt_words))

                    # step 4 - index
                    tgt_indexs.append(t_index)

            return src_texts, src_turn, tgt_indexs, tgt_texts, eos_indexs, src_contexts, tgt_contexts

        src_texts, src_turn, tgt_indexs, tgt_texts, eos_indexs, src_context, tgt_context = parse_file(
            f"{DATAPATH}/data")
        print(
            f"Ignored word counts - {self.dict(src_texts, self.min_word_count)}")

        src_context = np.asarray(src_context)
        tgt_context = np.asarray(tgt_context)
        src_texts = np.asarray(common.texts2idx(src_texts, self.dict.word2idx))
        src_turn = np.asarray(src_turn)
        tgt_indexs = np.asarray(tgt_indexs)
        eos_indexs = np.asarray(eos_indexs)
        tgt_texts = np.asarray(common.texts2idx(tgt_texts, self.dict.word2idx))

        assert src_texts.shape == src_turn.shape
        assert tgt_indexs.shape == tgt_texts.shape

        index = np.arange(tgt_texts.shape[0])
        np.random.shuffle(index)
        src_context = src_context[index]
        tgt_context = tgt_context[index]
        src_texts = src_texts[index]
        src_turn = src_turn[index]
        tgt_indexs = tgt_indexs[index]
        tgt_texts = tgt_texts[index]
        eos_indexs = eos_indexs[index]

        split_count = 2000
        self.src_context_train = src_context[split_count:]
        self.tgt_context_train = tgt_context[split_count:]
        self.src_texts_train = src_texts[split_count:]
        self.src_turn_train = src_turn[split_count:]
        self.tgt_indexs_train = tgt_indexs[split_count:]
        self.tgt_texts_train = tgt_texts[split_count:]
        self.eos_indexs_train = eos_indexs[split_count:]

        self.src_context_test = src_context[:split_count]
        self.tgt_context_test = tgt_context[:split_count]
        self.src_texts_test = src_texts[:split_count]
        self.src_turn_test = src_turn[:split_count]
        self.tgt_indexs_test = tgt_indexs[:split_count]
        self.tgt_texts_test = tgt_texts[:split_count]
        self.eos_indexs_test = eos_indexs[:split_count]

    def save(self):
        data = {
            'word2idx':  self.dict.word2idx,
            'max_len':  self.max_len,
            'train': {
                'src_context': self.src_context_train,
                'tgt_context': self.tgt_context_train,
                'src_texts': self.src_texts_train,
                'src_turn': self.src_turn_train,
                'tgt_indexs': self.tgt_indexs_train,
                'tgt_texts':  self.tgt_texts_train,
                'eos_indexs':  self.eos_indexs_train,
            },
            'valid': {
                'src_context': self.src_context_test,
                'tgt_context': self.tgt_context_test,
                'src_texts': self.src_texts_test,
                'src_turn': self.src_turn_test,
                'tgt_indexs': self.tgt_indexs_test,
                'tgt_texts':  self.tgt_texts_test,
                'eos_indexs':  self.eos_indexs_test,
            }
        }
        common.middle_save(data, self._save_data)
        print(f'corpora length - {len(self.dict)}')


if __name__ == "__main__":
    Corpus()
