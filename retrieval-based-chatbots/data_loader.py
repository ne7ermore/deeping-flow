from collections import namedtuple

import numpy as np

from const import *


def reps_pad(responses, max_len):
    return np.array([resp + [PAD] * (max_len - len(resp)) for resp in responses])


def uttes_pad(utterances, max_cont_len, max_utte_len):
    pad_utte = [[PAD] * max_utte_len]
    utterances = [[u + [PAD] * (max_utte_len - len(u))
                   for u in utte] for utte in utterances]
    utterances = [pad_utte * (max_cont_len - len(utte)) +
                  utte for utte in utterances]

    return np.array(utterances)


class DataLoader(object):
    def __init__(self,
                 utterances,
                 responses,
                 labels,
                 max_cont_len,
                 max_utte_len,
                 bsz=64,
                 shuffle=True):
        self.sents_size = len(utterances)
        self.step = 0
        self.stop_step = self.sents_size // bsz
        self.bsz = bsz
        self.max_cont_len = max_cont_len
        self.max_utte_len = max_utte_len
        self.utterances = np.asarray(utterances)
        self.responses = np.asarray(responses)
        self.labels = np.asarray(labels)
        self.nt = namedtuple(
            'dataloader', ['utterances', 'responses', 'labels'])

        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self.utterances.shape[0])
        np.random.shuffle(indices)
        self.utterances = self.utterances[indices]
        self.responses = self.responses[indices]
        self.labels = self.labels[indices]

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.stop_step:
            self.step = 0
            raise StopIteration()

        start = self.step * self.bsz
        bsz = min(self.bsz, self.sents_size - start)
        self.step += 1

        utterances = uttes_pad(
            self.utterances[start:start + bsz], self.max_cont_len, self.max_utte_len)
        responses = reps_pad(
            self.responses[start:start + bsz], self.max_utte_len)
        labels = self.labels[start:start + bsz]

        return self.nt._make([utterances, responses, labels])


if __name__ == '__main__':
    from corpus import middle_load

    data = middle_load('./data/corpus')

    training_data = DataLoader(
        data['train']['utterances'],
        data['train']['responses'],
        data['train']['labels'],
        data['max_cont_len'],
        data['max_utte_len'],
        bsz=4, shuffle=False)

    dict = data["dict"]["dict"]
    idx2word = {v: k for k, v in dict.items()}
    dt = next(training_data)

    print([idx2word[idx] for idx in dt.responses[3]])
    print(dt.labels[3])
    for utte in dt.utterances[3]:
        print([idx2word[idx] for idx in utte])
