import numpy as np
from collections import namedtuple
import const

def data_pad(sents, max_len):
    return np.array([s + [const.PAD] * (
            max_len - len(s)) for s in sents], dtype=np.int64)

def label_pad(labels, max_len, size):
    def fill_one(label):
        l = []
        for w in label + [const.PAD] * (max_len - len(label)):
            temp = [0]*size
            temp[w] = 1
            l.append(temp)
        return l

    pad_labels = map(fill_one, [label for label in labels])
    label_ids = np.array([l + [const.PAD] * (
            max_len - len(l)) for l in labels], dtype=np.int64)

    return np.asarray(list(pad_labels), dtype=np.float32), label_ids

class DataLoader(object):
    def __init__(self, datas, labels, d_max_len, l_max_len, tgt_vs, batch_size=64, shuffle=True):
        self.data_size = len(datas)
        self._step = 0
        self.stop_step = self.data_size // batch_size

        self._batch_size = batch_size
        self._d_max_len = d_max_len
        self._l_max_len = l_max_len
        self._tgt_vs = tgt_vs
        self._datas = np.asarray(datas)
        self._label = np.asarray(labels)
        self.nt = namedtuple('dataloader', ['data', 'label', 'label_ids'])
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._datas.shape[0])
        np.random.shuffle(indices)
        self._datas = self._datas[indices]
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        if self._step == self.stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        _bsz = min(self._batch_size, self.data_size-_start)

        self._step += 1
        data = data_pad(self._datas[_start:_start+_bsz], self._d_max_len)
        label, label_ids = label_pad(self._label[_start:_start+_bsz], self._l_max_len, self._tgt_vs)

        return self.nt._make([data, label, label_ids])

if __name__ == '__main__':
    from corpus import middle_load

    data = middle_load('./data/corpus')

    i2w = {v: k for k, v in data['dict']['src'].items()}
    training_data = DataLoader(
             data['train']['data'], data['train']['label'], data['max_w_len'], data['max_l_len'], data['dict']['src_size'], 100)
    print(training_data.data_size)
    for batch in training_data:
        print("")

