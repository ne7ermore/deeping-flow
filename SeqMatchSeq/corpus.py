from utils import prepare, corpora2idx, middle_save, load_pre_w2c
import const

class Dictionary(object):
    def __init__(self):
        self.ind2idx = {
            const.WORD[const.PAD]: const.PAD,
            const.WORD[const.UNK]: const.UNK
        }
        self.idx2ind = {
            const.PAD: const.WORD[const.PAD],
            const.UNK: const.WORD[const.UNK]
        }
        self.idx = 2

    def add(self, ind):
        if self.ind2idx.get(ind) is None:
            self.ind2idx[ind] = self.idx
            self.idx2ind[self.idx] = ind
            self.idx += 1

    def build_idx(self, sents, min_count):
        corpora = [cor for sent in sents for cor in sent]
        word_count = {w: 0 for w in set(corpora)}
        for w in corpora: word_count[w]+=1

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
    def __init__(self, train_src="data/train",
                       valid_src="data/val",
                       save_data="data/corpus",
                       max_length=16,
                       min_word_count=3):
        self._train_src = train_src
        self._valid_src = valid_src
        self._save_data = save_data
        self.max_length = max_length
        self._min_word_count = min_word_count
        self.src_sents = None
        self.tgt_sents = None
        self.src_valid_sents = None
        self.tgt_valid_sents = None
        self.dict = Dictionary()

    def parse_train(self):
        src_sents, tgt_sents, labels  = [], [], []
        ignore_len = 0

        for sentences in open(self._train_src):
            sentence = sentences.strip().split('\t')
            if len(sentence) != 3: continue
            src_sent, tgt_sent, label = sentence

            src_corpora = src_sent.strip().split()
            if len(src_corpora) > self.max_length:
                ignore_len += 1
                src_corpora = src_corpora[:self.max_length]

            tgt_corpora = tgt_sent.strip().split()
            if len(tgt_corpora) > self.max_length:
                ignore_len += 1
                tgt_corpora = tgt_corpora[:self.max_length]

            src_sents.append(src_corpora)
            tgt_sents.append(tgt_corpora)
            labels.append(int(label))

        ignore = self.dict.build_idx(src_sents+tgt_sents, self._min_word_count)

        if ignore_len != 0:
            print("Ignored max length - [{}]".format(ignore_len))

        if ignore != 0:
            print("Ignored corpus counts - [{}]".format(ignore))

        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.labels = labels

    def parse_valid(self, by_word=True):
        src_sents, tgt_sents, valid_labels = [], [], []
        for sentences in open(self._valid_src):
            sentence = sentences.strip().split('\t')
            if len(sentence) != 3: continue
            src_sent, tgt_sent, label = sentence

            src_corpora = src_sent.strip().split()
            if len(src_corpora) > self.max_length:
                src_corpora = src_corpora[:self.max_length]

            tgt_corpora = src_sent.strip().split()
            if len(tgt_corpora) > self.max_length:
                tgt_corpora = tgt_corpora[:self.max_length]

            src_sents.append(src_corpora)
            tgt_sents.append(tgt_corpora)
            valid_labels.append(int(label))

        self.src_valid_sents = src_sents
        self.tgt_valid_sents = tgt_sents
        self.valid_labels = valid_labels

    def save(self):
        data = {
            'max_lenth_src': self.max_length,
            'dict': {
                'src': self.dict.ind2idx,
                'src_size': len(self.dict),
            },
            'train': {
                'src': corpora2idx(self.src_sents, self.dict.ind2idx),
                'tgt': corpora2idx(self.tgt_sents, self.dict.ind2idx),
                'label': self.labels
            },
            'valid': {
                'src': corpora2idx(self.src_valid_sents, self.dict.ind2idx),
                'tgt': corpora2idx(self.tgt_valid_sents, self.dict.ind2idx),
                'label': self.valid_labels
            }
        }

        print('Finish dumping the corora data to file - [{}]'.format(self._save_data))
        print('corpora length - [{}]'.format(len(self.dict)))
        middle_save(data, self._save_data)

    def process(self):
        self.parse_train()
        if self._valid_src is not None:
            self.parse_valid()
        self.save()

if __name__ == "__main__":
    corpus = Corpus()
    corpus.process()
