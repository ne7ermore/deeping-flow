import numpy as np
import const

class DataLoader(object):
    def __init__(self, src_sents, tgt_sents, label, max_len,
                batch_size=64, shuffle=True):
        self.sents_size = len(src_sents)
        self._step = 0
        self.stop_step = self.sents_size // batch_size

        self._batch_size = batch_size
        self._max_len = max_len
        self._src_sents = np.asarray(src_sents)
        self._tgt_sents = np.asarray(tgt_sents)
        self._label = np.asarray(label)
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)
        self._src_sents = self._src_sents[indices]
        self._tgt_sents = self._tgt_sents[indices]
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        def pad2np(insts):
            return np.array([inst + [const.PAD] * (self._max_len - len(inst)) for inst in insts])

        def parse_label(labels):
            return [[0, 1] if l == 1 else [0, 1] for l in labels]


        if self._step == self.stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        _bsz = min(self._batch_size, self.sents_size-_start)
        self._step += 1
        src = pad2np(self._src_sents[_start:_start+_bsz])
        tgt = pad2np(self._tgt_sents[_start:_start+_bsz])
        label = np.asarray([[0., 1.] if l == 1 else [1., 0.] for l in self._label[_start:_start+_bsz]])
        return src, tgt, label

if __name__ == "__main__":
    from utils import middle_load
    data = middle_load("./data/corpus")
    training_data = DataLoader(
                 data['train']['src'],
                 data['train']['tgt'],
                 data['train']['label'],
                 data["max_lenth_src"],
                 batch_size=8)

    id2word = {v: k for k, v in data["dict"]["src"].items()}
    q, d, label = next(training_data)
    print(label)
    print(q)
    for doc in q:
        print([id2word[w] for w in doc])
    print("="*30)
    for doc in d:
        print([id2word[w] for w in doc])

