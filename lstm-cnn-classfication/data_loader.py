from collections import namedtuple

import numpy as np

import const

def data_pad(sents, max_len):
    return np.array([s + [const.PAD] * (max_len - len(s)) for s in sents])

def label_pad(labels, label_size):
    def fill_one(args):
        label, index = args
        label[index] = 1
        return label

    _labels = map(fill_one, [([0]*label_size, label) for _, label in enumerate(labels)])
    return np.asarray(list(_labels))

class DataLoader(object):
    def __init__(self, src_sents, label, max_len, label_size, batch_size=64, shuffle=True):
        self.sents_size = len(src_sents)
        self._step = 0
        self.stop_step = self.sents_size // batch_size

        self._batch_size = batch_size
        self._max_len = max_len
        self.label_size = label_size
        self._src_sents = np.asarray(src_sents)
        self._label = np.asarray(label)
        self.nt = namedtuple('dataloader', ['data', 'label'])
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)
        self._src_sents = self._src_sents[indices]
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        if self._step == self.stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        _bsz = min(self._batch_size, self.sents_size-_start)

        self._step += 1
        data = data_pad(self._src_sents[_start:_start+_bsz], self._max_len)
        label = label_pad(self._label[_start:_start+_bsz], self.label_size)

        return self.nt._make([data, label])

if __name__ == '__main__':
    from corpus import middle_load

    data = middle_load('./data/corpus')
    print(data['dict']['label'])

    i2w = {v: k for k, v in data['dict']['train'].items()}

    training_data = DataLoader(
             data['train']['src'], data['train']['label'], 16, 6, 8)    

    dt = next(training_data)

    print(dt.data)
    print(dt.label)

