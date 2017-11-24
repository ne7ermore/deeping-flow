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
    def __init__(self, train_src, save_data, valid_src, max_length, min_word_count):
        self._train_src = train_src
        self._valid_src = valid_src
        self._save_data = save_data
        self.max_length = max_length
        self._min_word_count = min_word_count
        self.src_sents = None
        self.tgt_sents = None
        self.src_valid_sents = None
        self.tgt_valid_sents = None

        self.jb = Jieba("../segmenter_dicts", useSynonym=True, HMM=False) # unuse hmm
        self.swf = StopwordFilter("../segmenter_dicts/stopwords.txt")
        # self.corpus_dict = Dictionary()
        # self.word_dict = Dictionary()
        self.dict = Dictionary()

    def sent2corpora(self, sentence, synonym=False):
        sentence = prepare(sentence)
        corpora = [e for e in self.jb.segment_search(sentence) if self.swf.filter(e)]
        new_corpora = []
        for corpus in corpora:
            if synonym and corpus in self.jb.synonym:
                corpus = self.jb.synonym[corpus]
            new_corpora.append(corpus)
        return new_corpora

    def parse_train(self):
        src_sents, tgt_sents, labels  = [], [], []
        for sentences in sopen(self._train_src):
            sentence = sentences.strip().split()
            if len(sentence) != 3: continue
            src_sent, tgt_sent, label = sentence

            src_corpora = self.sent2corpora(src_sent)
            if len(src_corpora) > self.max_length:
                src_corpora = src_corpora[:self.max_length]

            tgt_corpora = self.sent2corpora(tgt_sent)
            if len(tgt_corpora) > self.max_length:
                tgt_corpora = tgt_corpora[:self.max_length]

            src_sents.append(src_corpora)
            tgt_sents.append(tgt_corpora)
            labels.append(int(label))

        ignore = self.dict.build_idx(src_sents+tgt_sents, self._min_word_count)

        if ignore != 0:
            logging.info("Ignored corpus counts - [{}]".format(ignore))


        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.labels = labels

    def parse_valid(self, by_word=True):
        src_sents, tgt_sents, valid_labels = [], [], []
        for sentences in sopen(self._valid_src):
            sentence = sentences.strip().split()
            if len(sentence) != 3: continue
            src_sent, tgt_sent, label = sentence

            if by_word:
                src_corpora = [word for word in src_sent if self.swf.filter(word)]
                if len(src_corpora) > self.max_length:
                    src_corpora = src_corpora[:self.max_length]
            else:
                src_corpora = self.sent2corpora(src_sent)
                if len(src_corpora) > self.max_length:
                    src_corpora = src_corpora[:self.max_length]

            tgt_corpora = self.sent2corpora(tgt_sent)
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
            'max_lenth_src': self.max_length
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

        logging.info('Finish dumping the corora data to file - [{}]'.format(self._save_data))
        logging.info('corpora length - [{}]'.format(len(self.dict)))
        middle_save(data, self._save_data)

    def process(self):
        self.parse_train()
        if self._valid_src is not None:
            self.parse_valid()
        self.save()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='seq2sqe corpora handle')
    parser.add_argument('--train-src', type=str, default="data/fuel_train",
                        help='train file')
    parser.add_argument('--save-data', type=str, default="data/dssm_middle",
                        help='path to save processed data')
    parser.add_argument('--valid-src', type=str, default="./data/fuel_valid",
                        help='valid file')
    parser.add_argument('--max-lenth', type=int, default=16,
                        help='max length of sentence')
    parser.add_argument('--min-word-count', type=int, default='1',
                        help='min corpora count to discard')
    args = parser.parse_args()
    corpus = Corpus(args.train_src, args.save_data, args.valid_src,
                    args.max_lenth, args.min_word_count)
    corpus.process()
