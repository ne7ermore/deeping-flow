from collections import namedtuple

import numpy as np

import const


class DataLoader(object):
    def __init__(self,
                 original,
                 summary,
                 label,
                 max_ori_len,
                 max_sum_len,
                 bsz=64,
                 shuffle=True):
        self.sents_size = len(original)
        self.step = 0
        self.stop_step = self.sents_size // bsz

        self.bsz = bsz
        self.max_ori_len = max_ori_len
        self.max_sum_len = max_sum_len
        self.original = np.asarray(original)
        self.summary = np.asarray(summary)
        self.label = label
        self.nt = namedtuple('dataloader', ['original', 'summary', 'label'])
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self.original.shape[0])
        np.random.shuffle(indices)
        self.original = self.original[indices]
        self.summary = self.summary[indices]
        self.label = self.label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        def data_pad(sents, max_len):
            return np.array([s + [const.PAD] * (max_len - len(s)) for s in sents])

        if self.step == self.stop_step:
            self.step = 0
            raise StopIteration()

        start = self.step * self.bsz
        bsz = min(self.bsz, self.sents_size - start)

        self.step += 1
        original = data_pad(self.original[start:start + bsz], self.max_ori_len)
        summary = data_pad(self.summary[start:start + bsz], self.max_sum_len)
        label = self.label[start:start + bsz]

        return self.nt._make([original, summary, label])


if __name__ == '__main__':
    from corpus import middle_load

    data = middle_load('./data/corpus')

    training_data = DataLoader(
        data['train']['original'],
        data['train']['summary'],
        data['train']['label'],
        data['max_ori_len'],
        data['max_sum_len'],
        bsz=2)

    dict = data["dict"]["dict"]
    idx2word = {v: k for k, v in dict.items()}
    dt = next(training_data)

    print([idx2word[idx] for idx in dt.original[0]])
    print([idx2word[idx] for idx in dt.summary[0]])
    print(dt.label[0])
    # print(dt.original.shape)
    # print(dt.summary.shape)
    # print(dt.label.shape)
