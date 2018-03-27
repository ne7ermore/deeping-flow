import numpy as np
from collections import namedtuple
import const


def label_pad(labels, max_len):
    label_ids = np.array([l + [const.PAD] * (
        max_len - len(l)) for l in labels], dtype=np.int64)

    return label_ids


def data_pad(sents, max_len):
    return np.array([s + [const.PAD] * (
        max_len - len(s)) for s in sents], dtype=np.int64)


class DataLoader(object):
    def __init__(self, src, tgt, max_len, batch_size=64):
        self.data_size = len(src)
        self._step = 0
        self.stop_step = self.data_size // batch_size

        self._batch_size = batch_size
        self._max_len = max_len
        self._src = np.asarray(src)
        self._tgt = np.asarray(tgt)
        self.nt = namedtuple('dataloader', ['data', 'label'])

    def __iter__(self):
        return self

    def __next__(self):
        if self._step == self.stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step * self._batch_size
        _bsz = min(self._batch_size, self.data_size - _start)

        self._step += 1
        data = data_pad(self._src[_start:_start + _bsz], self._max_len)
        label = label_pad(
            self._tgt[_start:_start + _bsz], self._max_len)

        return self.nt._make([data, label])


if __name__ == '__main__':
    from corpus import middle_load

    data = middle_load('./data/corpus')

    i2w = {v: k for k, v in data['dict']['src'].items()}
    training_data = DataLoader(
        data['train']['data'], data['train']['label'], data['max_len'], 100)
    print(training_data.data_size)
    # for batch in training_data:
    #     print("")
